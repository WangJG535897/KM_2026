"""后处理和结果整合 - 支持二分类和多类"""
import numpy as np
import cv2
from typing import List, Dict, Tuple
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from .preprocess import auto_detect_roi, crop_roi
from .segmentation import (process_binary_segmentation, process_multiclass_segmentation,
                           filter_components_by_shape, extract_skeleton_from_mask, refine_masks)
from .color_refine import separate_curves_by_connectivity
from .trace import trace_multiple_curves
from .km_constraints import apply_km_constraints_batch
from .mapping import CoordinateMapper


def convert_roi_paths_to_global(paths_roi: List[np.ndarray], roi: Tuple[int, int, int, int]) -> List[np.ndarray]:
    """将ROI局部坐标路径转换为全图坐标

    Args:
        paths_roi: ROI局部坐标路径列表
        roi: (x1, y1, x2, y2) ROI坐标

    Returns:
        全图坐标路径列表
    """
    x1, y1, x2, y2 = roi
    paths_global = []

    for path in paths_roi:
        if len(path) == 0:
            paths_global.append(path)
            continue

        path_global = path.copy()
        path_global[:, 0] += x1  # x坐标偏移
        path_global[:, 1] += y1  # y坐标偏移
        paths_global.append(path_global)

    return paths_global


def estimate_curve_count_from_prob_map(prob_map: np.ndarray, min_prob: float = 0.2,
                                       min_distance: int = 15) -> int:
    """从概率图左侧估计曲线数量

    Args:
        prob_map: 概率图 [H, W]
        min_prob: 最小概率阈值
        min_distance: 峰值间最小距离

    Returns:
        估计的曲线数量
    """
    h, w = prob_map.shape

    # 分析左侧15%区域
    left_width = max(int(w * 0.15), 10)
    left_region = prob_map[:, :left_width]

    # 对每列检测峰值
    peak_counts = []
    for x in range(left_width):
        col = left_region[:, x]
        col_smooth = gaussian_filter1d(col, sigma=1.5)

        peaks, _ = find_peaks(col_smooth,
                             height=min_prob,
                             prominence=0.05,
                             distance=min_distance)

        if len(peaks) > 0:
            peak_counts.append(len(peaks))

    if len(peak_counts) == 0:
        return 0

    # 使用中位数作为估计（更稳健）
    estimated = int(np.median(peak_counts))
    print(f"[估计曲线数] 左侧{left_width}列峰值统计: min={min(peak_counts)}, "
          f"median={estimated}, max={max(peak_counts)}, mean={np.mean(peak_counts):.1f}")

    return estimated


class KMPipeline:
    """KM曲线提取完整管线 - 支持二分类和多类"""

    def __init__(self, x_max: float = 48.0, y_range: Tuple[float, float] = (0.0, 100.0)):
        self.x_max = x_max
        self.y_range = y_range
        self.roi = None
        self.mapper = None

    def process(self, image: np.ndarray, pred_result: dict,
                roi: Tuple[int, int, int, int] = None) -> Dict:
        """完整处理流程 - ROI优先，区分二分类和多类

        Args:
            image: 原始图像
            pred_result: 模型推理结果
            roi: (x1, y1, x2, y2) ROI坐标

        Returns:
            结果字典
        """

        # 1. 确定ROI
        if roi is None:
            roi = auto_detect_roi(image)
        self.roi = roi

        # 获取ROI图像
        roi_image = crop_roi(image, roi)

        # 2. 根据模型类型处理
        mode = pred_result.get('mode', 'multiclass')

        if mode == 'binary':
            return self._process_binary(image, roi_image, pred_result, roi)
        else:
            return self._process_multiclass(image, roi_image, pred_result, roi)

    def _process_binary(self, original_image: np.ndarray, roi_image: np.ndarray,
                       pred_result: dict, roi: Tuple) -> Dict:
        """二分类模型处理流程 - Ridge主链优先"""
        import logging
        logger = logging.getLogger(__name__)

        print(f"\n[Binary后处理] 开始处理...")
        logger.info(f"\n[postprocess] === 二分类后处理开始 ===")

        # 2. 获取概率图
        prob_map = pred_result['prob_map']

        print(f"[Binary后处理] 概率图统计: max={prob_map.max():.3f}, min={prob_map.min():.3f}, mean={prob_map.mean():.3f}")
        logger.info(f"[postprocess] 概率图统计: max={prob_map.max():.3f}, min={prob_map.min():.3f}, mean={prob_map.mean():.3f}")

        # 估计曲线数量（仅供参考）- 强制限制在1-3
        estimated_count_raw = estimate_curve_count_from_prob_map(prob_map, min_prob=0.2, min_distance=15)
        estimated_count = max(1, min(estimated_count_raw, 3))  # 强制限制
        print(f"[Binary后处理] 估计曲线数: {estimated_count_raw} -> 限制为 {estimated_count}")
        logger.info(f"[postprocess] 估计曲线数: {estimated_count_raw} -> 限制为 {estimated_count}")

        # === 主链：Ridge脊线提取 ===
        logger.info(f"\n[postprocess] === 主链：Ridge脊线提取 ===")
        from .fallback_trace import trace_from_prob_map_ridge

        ridge_paths, ridge_debug = trace_from_prob_map_ridge(
            prob_map,
            num_curves=estimated_count,  # 使用限制后的数量
            min_prob=0.15  # 从0.2降到0.15
        )
        print(f"[Binary后处理] Ridge主链提取: {len(ridge_paths)} 条路径")
        logger.info(f"[postprocess] Ridge主链提取: {len(ridge_paths)} 条路径")

        # === 辅助链：Binary分割追踪（降级为参考） ===
        logger.info(f"\n[postprocess] === 辅助链：Binary分割追踪 ===")
        seg_result = process_binary_segmentation(pred_result, roi_image)
        binary_mask = seg_result['binary_mask']

        print(f"[Binary后处理] 最佳阈值: {seg_result['best_threshold']}, 前景占比: {seg_result['fg_ratio']:.4f}")
        logger.info(f"[postprocess] Binary阈值: {seg_result['best_threshold']}, fg_ratio: {seg_result['fg_ratio']:.4f}")

        # 形状过滤
        component_masks = filter_components_by_shape(binary_mask, min_area=300, min_width=50, prob_map=prob_map)
        print(f"[Binary后处理] 形状过滤: 保留 {len(component_masks)} 个组件")

        if len(component_masks) == 0:
            print(f"[Binary后处理] 形状过滤后无组件，尝试更低阈值...")
            fallback_mask = (prob_map > 0.15).astype(np.uint8) * 255
            component_masks = filter_components_by_shape(fallback_mask, min_area=200, min_width=40, prob_map=prob_map)
            print(f"[Binary后处理] 降低阈值后: 保留 {len(component_masks)} 个组件")

        # 提取skeleton
        skeletons = []
        for i, mask in enumerate(component_masks):
            skeleton = extract_skeleton_from_mask(mask)
            skeleton_pixels = np.sum(skeleton > 0)
            if skeleton_pixels > 0:
                skeletons.append(skeleton)

        # 分离多条曲线
        separated_masks = []
        for skeleton in skeletons:
            curves = separate_curves_by_connectivity(skeleton, min_size=100)
            separated_masks.extend(curves)

        # 路径追踪
        regular_paths = trace_multiple_curves(separated_masks, enable_smooth=False)
        print(f"[Binary后处理] Regular辅助链: {len(regular_paths)} 条路径")
        logger.info(f"[postprocess] Regular辅助链: {len(regular_paths)} 条路径")

        # 检查路径覆盖率
        h, w = prob_map.shape
        valid_regular_paths = []
        for path in regular_paths:
            coverage = len(path) / w
            if coverage >= 0.35:
                valid_regular_paths.append(path)

        regular_paths = valid_regular_paths
        print(f"[Binary后处理] Regular覆盖率过滤后: {len(regular_paths)} 条路径")
        logger.info(f"[postprocess] Regular覆盖率过滤后: {len(regular_paths)} 条路径")

        # === 融合决策 ===
        logger.info(f"\n[postprocess] === 融合决策 ===")
        selected_method = 'ridge_primary'
        final_paths = ridge_paths
        fallback_triggered = False

        # 仅当ridge提取结果质量太差时，考虑fallback到regular
        if len(ridge_paths) == 0 and len(regular_paths) > 0:
            selected_method = 'regular_fallback'
            final_paths = regular_paths
            fallback_triggered = True
            logger.info(f"[postprocess] Ridge失败，使用Regular fallback")
            print(f"[Binary后处理] Ridge失败，使用Regular fallback")
        elif len(ridge_paths) == 1 and len(regular_paths) >= 2:
            # Hybrid：如果ridge只提1条但regular提2条，考虑合并
            # 检查regular路径质量
            from .fallback_trace import _score_path
            all_valid = all(_score_path(p, prob_map, w)[1] for p in regular_paths)
            if all_valid:
                selected_method = 'hybrid'
                final_paths = ridge_paths + regular_paths
                logger.info(f"[postprocess] Hybrid模式：Ridge {len(ridge_paths)} + Regular {len(regular_paths)}")
                print(f"[Binary后处理] Hybrid模式：Ridge {len(ridge_paths)} + Regular {len(regular_paths)}")
            else:
                logger.info(f"[postprocess] 使用Ridge primary（Regular质量不足）")
                print(f"[Binary后处理] 使用Ridge primary（Regular质量不足）")
        else:
            logger.info(f"[postprocess] 使用Ridge primary")
            print(f"[Binary后处理] 使用Ridge primary")

        logger.info(f"[postprocess] 最终方法: {selected_method}")
        logger.info(f"[postprocess] 最终曲线数: {len(final_paths)}")
        print(f"[Binary后处理] 最终方法: {selected_method}, 曲线数: {len(final_paths)}")

        # 失败诊断
        if len(final_paths) == 0:
            logger.warning(f"[postprocess] ALERT: 未检测到曲线!")
            logger.warning(f"  Ridge失败原因: {len(ridge_paths)} 条路径")
            logger.warning(f"  Regular失败原因: {len(regular_paths)} 条路径")
            logger.warning(f"  建议检查: prob_map统计、水平线抑制、起点检测")
            print(f"[Binary后处理] ⚠️ ALERT: 未检测到曲线!")

        # 应用KM约束
        constrained_paths_roi = apply_km_constraints_batch(final_paths)
        print(f"[Binary后处理] KM约束后: {len(constrained_paths_roi)} 条路径")
        logger.info(f"[postprocess] KM约束后: {len(constrained_paths_roi)} 条路径")

        # 转换为全图坐标
        constrained_paths_global = convert_roi_paths_to_global(constrained_paths_roi, roi)

        # 坐标映射（基于ROI局部坐标）
        x1, y1, x2, y2 = roi
        self.mapper = CoordinateMapper(
            roi=(0, 0, x2 - x1, y2 - y1),
            x_range=(0.0, self.x_max),
            y_range=self.y_range
        )

        chart_coords = self.mapper.batch_paths_to_chart(constrained_paths_roi)

        # 合并skeleton用于可视化
        combined_skeleton = np.zeros_like(binary_mask)
        for sk in skeletons:
            combined_skeleton = np.maximum(combined_skeleton, sk)

        print(f"[Binary后处理] ✓ 最终提取: {len(chart_coords)} 条曲线 (方法: {selected_method})\n")
        logger.info(f"[postprocess] ✓ 完成，最终提取: {len(chart_coords)} 条曲线\n")

        # 保存调试图像
        from pathlib import Path
        output_dir = Path(__file__).parent.parent.parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        # 保存Ridge调试图像
        if ridge_debug:
            import cv2
            # 保存抑制后的概率图
            suppressed_vis = (ridge_debug['suppressed_prob_map'] * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / "prob_map_suppressed.png"), suppressed_vis)

            # 保存水平线抑制mask
            suppression_vis = (ridge_debug['suppression_mask'] * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / "horizontal_suppression_mask.png"), suppression_vis)

            # 保存Ridge路径可视化
            ridge_paths_vis = cv2.cvtColor(suppressed_vis, cv2.COLOR_GRAY2BGR)
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            for idx, path in enumerate(ridge_paths):
                color = colors[idx % len(colors)]
                path_int = path.astype(int)
                for i in range(len(path_int) - 1):
                    pt1 = tuple(path_int[i])
                    pt2 = tuple(path_int[i + 1])
                    cv2.line(ridge_paths_vis, pt1, pt2, color, 2)
                # 标记起点
                cv2.circle(ridge_paths_vis, tuple(path_int[0]), 5, (255, 255, 255), -1)
            cv2.imwrite(str(output_dir / "ridge_paths.png"), ridge_paths_vis)

            logger.info(f"[postprocess] 调试图像已保存到 {output_dir}")

        return {
            'mode': 'binary',
            'roi': roi,
            'original_image': original_image,
            'roi_image': roi_image,
            'prob_map': prob_map,
            'binary_mask': binary_mask,
            'best_threshold': seg_result['best_threshold'],
            'component_masks': component_masks,
            'skeleton': combined_skeleton,
            'separated_masks': separated_masks,
            'pixel_paths_roi': constrained_paths_roi,
            'pixel_paths_global': constrained_paths_global,
            'pixel_paths': constrained_paths_roi,  # 兼容旧代码
            'chart_coords': chart_coords,
            'num_curves': len(chart_coords),
            'fg_ratio': seg_result['fg_ratio'],
            'estimated_curve_count': len(final_paths),  # 使用实际提取数
            'fallback_triggered': fallback_triggered,
            'selected_method': selected_method,
            'ridge_debug': ridge_debug,  # 新增：Ridge调试信息
            'ridge_paths_count': len(ridge_paths),  # 新增
            'regular_paths_count': len(regular_paths)  # 新增
        }

    def _process_multiclass(self, original_image: np.ndarray, roi_image: np.ndarray,
                           pred_result: dict, roi: Tuple) -> Dict:
        """多类模型处理流程"""
        # 2. 处理分割结果
        seg_result = process_multiclass_segmentation(pred_result, roi_image)
        class_masks = seg_result['class_masks']

        # 3. 精修masks
        refined_masks = refine_masks(class_masks)

        # 4. 分离曲线（如果有重叠）
        separated_masks = []
        for mask in refined_masks:
            curves = separate_curves_by_connectivity(mask, min_size=300)
            separated_masks.extend(curves)

        # 5. 路径追踪（ROI局部坐标）
        traced_paths_roi = trace_multiple_curves(separated_masks)

        # 6. 应用KM约束
        constrained_paths_roi = apply_km_constraints_batch(traced_paths_roi)

        # 7. 转换为全图坐标
        constrained_paths_global = convert_roi_paths_to_global(constrained_paths_roi, roi)

        # 8. 坐标映射（基于ROI局部坐标）
        x1, y1, x2, y2 = roi
        self.mapper = CoordinateMapper(
            roi=(0, 0, x2 - x1, y2 - y1),
            x_range=(0.0, self.x_max),
            y_range=self.y_range
        )

        chart_coords = self.mapper.batch_paths_to_chart(constrained_paths_roi)

        return {
            'mode': 'multiclass',
            'roi': roi,
            'original_image': original_image,
            'roi_image': roi_image,
            'class_masks': class_masks,
            'refined_masks': refined_masks,
            'separated_masks': separated_masks,
            'pixel_paths_roi': constrained_paths_roi,
            'pixel_paths_global': constrained_paths_global,
            'pixel_paths': constrained_paths_roi,  # 兼容旧代码
            'chart_coords': chart_coords,
            'num_curves': len(chart_coords)
        }
