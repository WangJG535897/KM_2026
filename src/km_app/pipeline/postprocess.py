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
        """二分类模型处理流程 - 增强版with fallback"""
        print(f"\n[Binary后处理] 开始处理...")

        # 2. 处理分割结果
        seg_result = process_binary_segmentation(pred_result, roi_image)
        prob_map = seg_result['prob_map']
        binary_mask = seg_result['binary_mask']

        print(f"[Binary后处理] 概率图统计: max={prob_map.max():.3f}, min={prob_map.min():.3f}, mean={prob_map.mean():.3f}")
        print(f"[Binary后处理] 最佳阈值: {seg_result['best_threshold']}, 前景占比: {seg_result['fg_ratio']:.4f}")

        # 打印所有候选阈值的统计
        for cand in seg_result['candidates']:
            print(f"  阈值{cand['threshold']}: fg_pixels={cand['fg_pixels']}, fg_ratio={cand['fg_ratio']:.4f}")

        # 估计曲线数量
        estimated_count = estimate_curve_count_from_prob_map(prob_map, min_prob=0.2, min_distance=15)
        print(f"[Binary后处理] 估计曲线数: {estimated_count}")

        # 3. 形状过滤
        component_masks = filter_components_by_shape(binary_mask, min_area=300, min_width=50, prob_map=prob_map)
        print(f"[Binary后处理] 形状过滤: 保留 {len(component_masks)} 个组件")

        if len(component_masks) == 0:
            print(f"[Binary后处理] 形状过滤后无组件，尝试更低阈值...")
            fallback_mask = (prob_map > 0.15).astype(np.uint8) * 255
            component_masks = filter_components_by_shape(fallback_mask, min_area=200, min_width=40, prob_map=prob_map)
            print(f"[Binary后处理] 降低阈值后: 保留 {len(component_masks)} 个组件")

        # 4. 提取skeleton
        skeletons = []
        for i, mask in enumerate(component_masks):
            skeleton = extract_skeleton_from_mask(mask)
            skeleton_pixels = np.sum(skeleton > 0)
            print(f"[Binary后处理] 组件{i+1} skeleton像素数: {skeleton_pixels}")
            if skeleton_pixels > 0:
                skeletons.append(skeleton)

        print(f"[Binary后处理] 有效skeleton数量: {len(skeletons)}")

        # 5. 分离多条曲线（如果有重叠）
        separated_masks = []
        for skeleton in skeletons:
            curves = separate_curves_by_connectivity(skeleton, min_size=100)
            separated_masks.extend(curves)

        print(f"[Binary后处理] 连通域分离后: {len(separated_masks)} 个mask")

        # 6. 路径追踪（ROI局部坐标）- 不启用平滑以保留KM台阶
        traced_paths_roi = trace_multiple_curves(separated_masks, enable_smooth=False)
        print(f"[Binary后处理] 常规追踪得到: {len(traced_paths_roi)} 条路径")

        # 检查路径覆盖率
        h, w = prob_map.shape
        valid_paths = []
        for path in traced_paths_roi:
            coverage = len(path) / w
            if coverage >= 0.35:
                valid_paths.append(path)
            else:
                print(f"[Binary后处理] 路径覆盖率{coverage:.1%}太低，丢弃")

        traced_paths_roi = valid_paths
        print(f"[Binary后处理] 覆盖率过滤后: {len(traced_paths_roi)} 条路径")

        # 7. 决策是否启用fallback
        fallback_triggered = False
        selected_method = "regular"

        # 触发条件：
        # 1. 常规流程得到0条路径
        # 2. 或者路径数量明显少于估计数量（且估计数>=2）
        should_fallback = (
            len(traced_paths_roi) == 0 or
            (estimated_count >= 2 and len(traced_paths_roi) < estimated_count - 1)
        )

        if should_fallback:
            print(f"[Binary后处理] ⚠️ 触发fallback (常规={len(traced_paths_roi)}, 估计={estimated_count})")
            from .fallback_trace import trace_from_prob_map_ridge

            fallback_paths = trace_from_prob_map_ridge(prob_map, num_curves=max(3, estimated_count), min_prob=0.2)
            print(f"[Binary后处理] Fallback提取到: {len(fallback_paths)} 条路径")

            if len(fallback_paths) > len(traced_paths_roi):
                # Fallback更好，使用fallback结果
                traced_paths_roi = fallback_paths
                fallback_triggered = True
                selected_method = "fallback"
                print(f"[Binary后处理] ✓ 使用fallback结果")
            elif len(traced_paths_roi) > 0:
                # 常规流程有结果，保留常规
                print(f"[Binary后处理] ✓ 保留常规结果")
            else:
                # 都失败了，至少用fallback
                traced_paths_roi = fallback_paths
                fallback_triggered = True
                selected_method = "fallback"
                print(f"[Binary后处理] ✓ 使用fallback结果（常规失败）")

        # 8. 应用KM约束
        constrained_paths_roi = apply_km_constraints_batch(traced_paths_roi)
        print(f"[Binary后处理] KM约束后: {len(constrained_paths_roi)} 条路径")

        # 9. 转换为全图坐标
        constrained_paths_global = convert_roi_paths_to_global(constrained_paths_roi, roi)

        # 10. 坐标映射（基于ROI局部坐标）
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
            'estimated_curve_count': estimated_count,
            'fallback_triggered': fallback_triggered,
            'selected_method': selected_method
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
