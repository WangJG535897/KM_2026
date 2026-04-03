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


def suppress_non_curve_regions(prob_map: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    轻量级预清洗：抑制明显非主曲线区域

    处理目标：
    1. 右上图例区域
    2. 下方统计文本区域
    3. 细长竖线型删失标记

    Returns:
        (cleaned_prob_map, debug_info)
    """
    H, W = prob_map.shape
    cleaned = prob_map.copy()
    suppression_mask = np.zeros_like(prob_map, dtype=bool)

    debug_info = {
        'legend_region': None,
        'bottom_text_region': None,
        'vertical_lines_count': 0
    }

    print(f"\n[预清洗] 开始抑制非曲线区域...")

    # 1. 右上图例区域抑制（启发式：右上角20%x20%区域）
    legend_h = int(H * 0.20)
    legend_w_start = int(W * 0.75)
    legend_region = cleaned[:legend_h, legend_w_start:]
    if legend_region.mean() > 0.15:  # 如果该区域有明显响应
        cleaned[:legend_h, legend_w_start:] *= 0.3  # 降权70%
        suppression_mask[:legend_h, legend_w_start:] = True
        debug_info['legend_region'] = (0, legend_w_start, legend_h, W)
        print(f"[预清洗] 抑制右上图例区: y=[0,{legend_h}], x=[{legend_w_start},{W}], mean_prob={legend_region.mean():.3f}")

    # 2. 下方统计文本区域抑制（启发式：底部15%区域）
    bottom_h_start = int(H * 0.85)
    bottom_region = cleaned[bottom_h_start:, :]
    if bottom_region.mean() > 0.15:  # 如果该区域有明显响应
        cleaned[bottom_h_start:, :] *= 0.3  # 降权70%
        suppression_mask[bottom_h_start:, :] = True
        debug_info['bottom_text_region'] = (bottom_h_start, 0, H, W)
        print(f"[预清洗] 抑制下方文本区: y=[{bottom_h_start},{H}], mean_prob={bottom_region.mean():.3f}")

    # 3. 细长竖线型删失标记抑制
    # 检测每列的垂直连续性
    vertical_lines_suppressed = 0
    for x in range(W):
        col = prob_map[:, x]
        high_prob_pixels = np.sum(col > 0.3)

        # 如果某列有很多高概率像素，但宽度很窄，可能是竖线
        if high_prob_pixels > H * 0.15:  # 超过15%高度
            # 检查左右邻域是否也有类似响应（真曲线会有宽度）
            left_support = 0
            right_support = 0

            if x > 0:
                left_col = prob_map[:, x-1]
                left_support = np.sum(left_col > 0.3)

            if x < W - 1:
                right_col = prob_map[:, x+1]
                right_support = np.sum(right_col > 0.3)

            # 如果左右支撑都很弱，判定为孤立竖线
            if left_support < H * 0.05 and right_support < H * 0.05:
                cleaned[:, x] *= 0.2  # 降权80%
                suppression_mask[:, x] = True
                vertical_lines_suppressed += 1

    if vertical_lines_suppressed > 0:
        debug_info['vertical_lines_count'] = vertical_lines_suppressed
        print(f"[预清洗] 抑制竖线删失标记: {vertical_lines_suppressed} 列")

    suppression_ratio = np.sum(suppression_mask) / (H * W)
    print(f"[预清洗] ✓ 完成，抑制像素占比: {suppression_ratio:.1%}\n")

    return cleaned, debug_info


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

        # 2. 获取概率图 - 判断坐标系
        prob_map_full = pred_result['prob_map']
        transform_info = pred_result.get('transform_info', {})
        x1, y1, x2, y2 = roi

        # 关键修复：判断prob_map是整图坐标还是ROI局部坐标
        # 如果推理时传入了roi，则prob_map已经是ROI局部坐标
        if transform_info.get('roi') is not None:
            # ROI模式：prob_map已经是ROI局部坐标，直接使用
            prob_map = prob_map_full
            print(f"[Binary后处理] 检测到ROI模式，prob_map已是局部坐标，尺寸: {prob_map.shape}")
            logger.info(f"[postprocess] ROI模式，prob_map shape: {prob_map.shape}")
        else:
            # 整图模式：prob_map是全图坐标，需要裁剪
            prob_map = prob_map_full[y1:y2, x1:x2]
            print(f"[Binary后处理] 检测到整图模式，裁剪prob_map到ROI，尺寸: {prob_map.shape}")
            logger.info(f"[postprocess] 整图模式，裁剪后prob_map shape: {prob_map.shape}")

        # 防御性检查：确保prob_map非空
        if prob_map.size == 0:
            raise ValueError(f"prob_map为空数组！prob_map_full.shape={prob_map_full.shape}, roi={roi}, transform_info={transform_info}")

        print(f"[Binary后处理] 概率图统计: max={prob_map.max():.3f}, min={prob_map.min():.3f}, mean={prob_map.mean():.3f}")
        logger.info(f"[postprocess] 概率图统计: max={prob_map.max():.3f}, min={prob_map.min():.3f}, mean={prob_map.mean():.3f}")

        # === 预清洗：抑制非曲线区域 ===
        prob_map_cleaned, clean_debug = suppress_non_curve_regions(prob_map)
        print(f"[Binary后处理] 预清洗后统计: max={prob_map_cleaned.max():.3f}, mean={prob_map_cleaned.mean():.3f}")

        # 估计曲线数量（仅供参考）- 强制限制在1-3
        estimated_count_raw = estimate_curve_count_from_prob_map(prob_map_cleaned, min_prob=0.2, min_distance=15)
        estimated_count = max(1, min(estimated_count_raw, 3))  # 强制限制
        print(f"[Binary后处理] 估计曲线数: {estimated_count_raw} -> 限制为 {estimated_count}")
        logger.info(f"[postprocess] 估计曲线数: {estimated_count_raw} -> 限制为 {estimated_count}")

        # === 主链：Ridge脊线提取 ===
        logger.info(f"\n[postprocess] === 主链：Ridge脊线提取 ===")
        from .fallback_trace import trace_from_prob_map_ridge

        ridge_paths, ridge_debug = trace_from_prob_map_ridge(
            prob_map_cleaned,  # 使用清洗后的prob_map
            num_curves=estimated_count,  # 使用限制后的数量
            min_prob=0.15  # 从0.2降到0.15
        )
        print(f"[Binary后处理] Ridge主链提取: {len(ridge_paths)} 条路径")
        logger.info(f"[postprocess] Ridge主链提取: {len(ridge_paths)} 条路径")

        # === 辅助链：直接从prob_map追踪 ===
        logger.info(f"\n[postprocess] === 辅助链：直接prob_map追踪 ===")
        # 直接将prob_map二值化并追踪
        direct_mask = (prob_map > 0.20).astype(np.uint8) * 255

        # 尝试分离多条曲线
        from .color_refine import separate_curves_by_connectivity
        from .segmentation import extract_skeleton_from_mask
        skeleton = extract_skeleton_from_mask(direct_mask)
        separated_direct_masks = separate_curves_by_connectivity(skeleton, min_size=100)
        print(f"[Binary后处理] Direct分离: {len(separated_direct_masks)} 个组件")

        from .trace import trace_multiple_curves
        direct_paths = trace_multiple_curves(separated_direct_masks, enable_smooth=False)
        print(f"[Binary后处理] Direct追踪: {len(direct_paths)} 条路径")
        logger.info(f"[postprocess] Direct追踪: {len(direct_paths)} 条路径")

        # 检查Direct路径覆盖率 - 改进：保底策略
        h, w = prob_map.shape
        valid_direct_paths = []
        direct_path_scores = []  # 记录(path, coverage, score)

        for path in direct_paths:
            coverage = len(path) / w
            # 计算路径平均概率
            path_int = path.astype(int)
            path_int[:, 0] = np.clip(path_int[:, 0], 0, w - 1)
            path_int[:, 1] = np.clip(path_int[:, 1], 0, h - 1)
            path_probs = prob_map[path_int[:, 1], path_int[:, 0]]
            avg_prob = np.mean(path_probs)

            # 综合评分
            score = coverage * 100 + avg_prob * 50

            direct_path_scores.append((path, coverage, score))

            if coverage >= 0.20:  # 从0.25降到0.20
                valid_direct_paths.append(path)
                print(f"[Binary后处理] Direct路径: 长度={len(path)}, 覆盖={coverage:.1%}, prob={avg_prob:.3f}, score={score:.1f} ✓")
            else:
                print(f"[Binary后处理] Direct路径: 长度={len(path)}, 覆盖={coverage:.1%}, prob={avg_prob:.3f}, score={score:.1f} ✗")

        # 保底策略：如果Ridge=0且所有Direct都被过滤，保留top-2候选作为DEBUG候选
        # 注意：这些候选不会直接成为最终输出，需要通过后续完整性验证
        debug_direct_candidates = []
        if len(ridge_paths) == 0 and len(valid_direct_paths) == 0 and len(direct_path_scores) > 0:
            print(f"[Binary后处理] Direct保底策略：Ridge=0且所有Direct被过滤，保留top-2作为DEBUG候选（非最终输出）")
            direct_path_scores.sort(key=lambda x: x[2], reverse=True)  # 按score排序
            for path, coverage, score in direct_path_scores[:2]:
                debug_direct_candidates.append(path)
                print(f"[Binary后处理] Direct DEBUG候选: 长度={len(path)}, 覆盖={coverage:.1%}, score={score:.1f} (仅供调试)")

        direct_paths = valid_direct_paths
        print(f"[Binary后处理] Direct覆盖率过滤后: {len(direct_paths)} 条路径 (DEBUG候选: {len(debug_direct_candidates)} 条)")
        logger.info(f"[postprocess] Direct覆盖率过滤后: {len(direct_paths)} 条路径")

        # === 辅助链：Binary分割追踪（作为主要方法） ===
        logger.info(f"\n[postprocess] === 辅助链：Binary分割追踪 ===")
        # 使用ROI-cropped prob_map创建临时pred_result
        temp_pred_result = {'prob_map': prob_map, 'mode': 'binary'}
        seg_result = process_binary_segmentation(temp_pred_result, roi_image)
        binary_mask = seg_result['binary_mask']

        print(f"[Binary后处理] 最佳阈值: {seg_result['best_threshold']}, 前景占比: {seg_result['fg_ratio']:.4f}")
        logger.info(f"[postprocess] Binary阈值: {seg_result['best_threshold']}, fg_ratio: {seg_result['fg_ratio']:.4f}")

        # 形状过滤
        component_masks = filter_components_by_shape(binary_mask, min_area=300, min_width=50, prob_map=prob_map)
        print(f"[Binary后处理] 形状过滤: 保留 {len(component_masks)} 个组件")

        # 改进：如果形状过滤后无组件，根据fg_ratio决定方向
        if len(component_masks) == 0:
            fg_ratio = seg_result['fg_ratio']
            if fg_ratio > 0.50:
                # fg_ratio过高，尝试更高阈值收紧
                print(f"[Binary后处理] fg_ratio={fg_ratio:.3f}过高，尝试更高阈值收紧...")
                for higher_thresh in [0.30, 0.35, 0.40, 0.45]:
                    if higher_thresh <= seg_result['best_threshold']:
                        continue
                    fallback_mask = (prob_map > higher_thresh).astype(np.uint8) * 255
                    component_masks = filter_components_by_shape(fallback_mask, min_area=200, min_width=40, prob_map=prob_map)
                    print(f"[Binary后处理] 尝试阈值{higher_thresh}: 保留 {len(component_masks)} 个组件")
                    if len(component_masks) > 0:
                        break
            else:
                # fg_ratio不高，尝试更低阈值
                print(f"[Binary后处理] fg_ratio={fg_ratio:.3f}不高，尝试更低阈值...")
                fallback_mask = (prob_map > 0.10).astype(np.uint8) * 255
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

        # 检查路径覆盖率 - 改进：保底策略
        h, w = prob_map.shape
        valid_regular_paths = []
        regular_path_scores = []  # 记录(path, coverage, score)

        for path in regular_paths:
            coverage = len(path) / w
            # 计算路径平均概率
            path_int = path.astype(int)
            path_int[:, 0] = np.clip(path_int[:, 0], 0, w - 1)
            path_int[:, 1] = np.clip(path_int[:, 1], 0, h - 1)
            path_probs = prob_map[path_int[:, 1], path_int[:, 0]]
            avg_prob = np.mean(path_probs)

            # 综合评分
            score = coverage * 100 + avg_prob * 50

            regular_path_scores.append((path, coverage, score))

            if coverage >= 0.30:  # 从0.35降到0.30
                valid_regular_paths.append(path)
                print(f"[Binary后处理] Regular路径: 长度={len(path)}, 覆盖={coverage:.1%}, prob={avg_prob:.3f}, score={score:.1f} ✓")
            else:
                print(f"[Binary后处理] Regular路径: 长度={len(path)}, 覆盖={coverage:.1%}, prob={avg_prob:.3f}, score={score:.1f} ✗")

        # 保底策略：如果Ridge=0且Direct=0且所有Regular都被过滤，保留top-2作为DEBUG候选
        # 注意：这些候选不会直接成为最终输出，需要通过后续完整性验证
        debug_regular_candidates = []
        if len(ridge_paths) == 0 and len(direct_paths) == 0 and len(valid_regular_paths) == 0 and len(regular_path_scores) > 0:
            print(f"[Binary后处理] Regular保底策略：Ridge=0且Direct=0且所有Regular被过滤，保留top-2作为DEBUG候选（非最终输出）")
            regular_path_scores.sort(key=lambda x: x[2], reverse=True)  # 按score排序
            for path, coverage, score in regular_path_scores[:2]:
                debug_regular_candidates.append(path)
                print(f"[Binary后处理] Regular DEBUG候选: 长度={len(path)}, 覆盖={coverage:.1%}, score={score:.1f} (仅供调试)")

        regular_paths = valid_regular_paths
        print(f"[Binary后处理] Regular覆盖率过滤后: {len(regular_paths)} 条路径 (DEBUG候选: {len(debug_regular_candidates)} 条)")
        logger.info(f"[postprocess] Regular覆盖率过滤后: {len(regular_paths)} 条路径")

        # === 融合决策 ===
        logger.info(f"\n[postprocess] === 融合决策 ===")
        selected_method = 'ridge_primary'
        final_paths = ridge_paths
        fallback_triggered = False
        output_confidence = 'high'  # 新增：输出置信度

        # 优先级：Ridge > Direct > Regular
        if len(ridge_paths) == 0 and len(direct_paths) > 0:
            selected_method = 'direct_fallback'
            final_paths = direct_paths
            fallback_triggered = True
            output_confidence = 'medium'
            logger.info(f"[postprocess] Ridge失败，使用Direct fallback")
            print(f"[Binary后处理] Ridge失败，使用Direct fallback")
        elif len(ridge_paths) == 0 and len(regular_paths) > 0:
            selected_method = 'regular_fallback'
            final_paths = regular_paths
            fallback_triggered = True
            output_confidence = 'medium'
            logger.info(f"[postprocess] Ridge和Direct失败，使用Regular fallback")
            print(f"[Binary后处理] Ridge和Direct失败，使用Regular fallback")
        elif len(ridge_paths) > 0:
            logger.info(f"[postprocess] 使用Ridge primary")
            print(f"[Binary后处理] 使用Ridge primary")
        else:
            logger.info(f"[postprocess] 所有方法都失败")
            print(f"[Binary后处理] 所有方法都失败")
            output_confidence = 'failed'

        logger.info(f"[postprocess] 最终方法: {selected_method}")
        logger.info(f"[postprocess] 最终曲线数: {len(final_paths)}")
        print(f"[Binary后处理] 最终方法: {selected_method}, 曲线数: {len(final_paths)}, 置信度: {output_confidence}")

        # === 最终输出完整性验证 ===
        # 只有通过完整性验证的路径才能成为最终输出
        validated_paths = []
        for idx, path in enumerate(final_paths):
            coverage = len(path) / w
            # 最低完整性要求：覆盖率至少30%
            if coverage >= 0.30:
                validated_paths.append(path)
                print(f"[Binary后处理] 路径{idx+1}通过完整性验证: 覆盖={coverage:.1%} ✓")
            else:
                print(f"[Binary后处理] 路径{idx+1}未通过完整性验证: 覆盖={coverage:.1%} < 30% ✗ (拒绝输出)")
                output_confidence = 'low'

        final_paths = validated_paths
        print(f"[Binary后处理] 完整性验证后: {len(final_paths)} 条路径")

        # 失败诊断
        if len(final_paths) == 0:
            logger.warning(f"[postprocess] ALERT: 未检测到完整曲线!")
            logger.warning(f"  Ridge失败原因: {len(ridge_paths)} 条路径")
            logger.warning(f"  Direct失败原因: {len(direct_paths)} 条路径")
            logger.warning(f"  Regular失败原因: {len(regular_paths)} 条路径")
            logger.warning(f"  建议检查: prob_map统计、水平线抑制、起点检测")
            print(f"[Binary后处理] ⚠️ ALERT: 未检测到完整曲线! 输出置信度: {output_confidence}")
            print(f"[Binary后处理] 可用DEBUG候选: Direct={len(debug_direct_candidates) if 'debug_direct_candidates' in locals() else 0}, Regular={len(debug_regular_candidates) if 'debug_regular_candidates' in locals() else 0}")

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

        print(f"[Binary后处理] ✓ 最终提取: {len(chart_coords)} 条曲线 (方法: {selected_method}, 置信度: {output_confidence})\n")
        logger.info(f"[postprocess] ✓ 完成，最终提取: {len(chart_coords)} 条曲线，置信度: {output_confidence}\n")

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
            'output_confidence': output_confidence,  # 新增
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
