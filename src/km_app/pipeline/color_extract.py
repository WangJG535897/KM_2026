"""Color-first曲线提取器 - 唯一目标：从图片提取彩色线的像素轨迹"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from skimage.morphology import skeletonize
from .trace import trace_curve_column_scan, trace_multiple_curves
from .color_refine import separate_curves_by_connectivity


def detect_image_type(roi_image: np.ndarray) -> Tuple[str, Dict]:
    """
    判断图像类型：彩色图 vs 灰度图

    策略：
    1. 计算ROI内像素的饱和度统计
    2. 如果大部分像素饱和度很低，判定为灰度图
    3. 否则判定为彩色图

    Returns:
        (image_type, debug_info)
        image_type: 'color' or 'grayscale'
    """
    # 转换到HSV空间
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # 统计饱和度
    mean_sat = np.mean(saturation)
    median_sat = np.median(saturation)
    low_sat_ratio = np.sum(saturation < 30) / saturation.size

    # 判定阈值：如果80%以上像素饱和度<30，或平均饱和度<25，判定为灰度图
    is_grayscale = (low_sat_ratio > 0.80) or (mean_sat < 25)

    image_type = 'grayscale' if is_grayscale else 'color'

    debug_info = {
        'mean_saturation': mean_sat,
        'median_saturation': median_sat,
        'low_sat_ratio': low_sat_ratio,
        'image_type': image_type
    }

    print(f"[图像类型判定] 平均饱和度={mean_sat:.1f}, 中位数={median_sat:.1f}, 低饱和度占比={low_sat_ratio:.1%}")
    print(f"[图像类型判定] 判定结果: {image_type.upper()}")

    return image_type, debug_info


def detect_plot_bbox_simple(roi_image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    在ROI内检测真实plot box边界

    策略：保守估计，去掉明显的边缘区域
    - 左侧：去掉15%（y轴区）
    - 右侧：去掉20%（图例区）
    - 顶部：去掉5%（标题区）
    - 底部：去掉15%（x轴和文本区）

    Returns:
        (x1, y1, x2, y2) plot box坐标（相对于ROI）
    """
    H, W = roi_image.shape[:2]

    x1 = int(W * 0.15)
    x2 = int(W * 0.80)
    y1 = int(H * 0.05)
    y2 = int(H * 0.85)

    print(f"[plot_bbox检测] plot_bbox=({x1},{y1},{x2},{y2}), 尺寸={(x2-x1)}x{(y2-y1)}")

    return (x1, y1, x2, y2)


def extract_thin_line_foreground(roi_image: np.ndarray, plot_bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    提取细线候选前景（针对灰度图）

    策略：
    1. 只在plot_bbox内处理
    2. 提取暗色细线结构
    3. 使用top-hat变换增强细线
    4. 抑制大块填充区域

    Returns:
        foreground mask (uint8, 0-255)
    """
    x1, y1, x2, y2 = plot_bbox
    plot_region = roi_image[y1:y2, x1:x2]

    # 转灰度
    gray = cv2.cvtColor(plot_region, cv2.COLOR_BGR2GRAY)

    # 1. 基础暗线提取
    _, dark_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 2. Top-hat变换增强细线
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_line)
    _, tophat_mask = cv2.threshold(tophat, 10, 255, cv2.THRESH_BINARY)

    # 3. 合并
    foreground = cv2.bitwise_or(dark_mask, tophat_mask)

    # 4. 去除大块区域（文字、轴线等）
    # 使用连通域分析，过滤掉宽度或高度过大的组件
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)

    filtered = np.zeros_like(foreground)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # 保留条件：
        # - 宽度不能太小（至少20像素，避免竖线删失标记）
        # - 高度不能太大（不超过plot高度的30%，避免大块文字）
        # - 宽高比要合理（宽度应该明显大于高度）
        plot_h = y2 - y1
        if w >= 20 and h < plot_h * 0.30 and w > h * 1.5:
            component_mask = (labels == i).astype(np.uint8) * 255
            filtered = cv2.bitwise_or(filtered, component_mask)

    # 5. 轻量形态学清理
    kernel_clean = np.ones((2, 2), np.uint8)
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel_clean)

    # 6. 扩展到完整ROI尺寸
    full_foreground = np.zeros(roi_image.shape[:2], dtype=np.uint8)
    full_foreground[y1:y2, x1:x2] = filtered

    fg_pixels = np.sum(filtered > 0)
    plot_pixels = (x2 - x1) * (y2 - y1)
    fg_ratio = fg_pixels / plot_pixels if plot_pixels > 0 else 0

    print(f"[细线前景提取] plot内前景像素: {fg_pixels}, 占比: {fg_ratio:.1%}")

    return full_foreground


def extract_colored_curves(image: np.ndarray,
                          roi: Tuple[int,int,int,int] = None,
                          n_colors: int = 5) -> Dict:
    """
    从图片提取彩色曲线像素轨迹

    Args:
        image: 输入图像 (BGR)
        roi: (x1,y1,x2,y2) ROI区域，None则自动
        n_colors: 颜色聚类数

    Returns:
        {
            'roi': (x1,y1,x2,y2),
            'roi_image': np.ndarray,
            'original_image': np.ndarray,
            'color_masks': List[np.ndarray],
            'separated_masks': List[np.ndarray],
            'pixel_paths_roi': List[np.ndarray],  # ROI局部坐标
            'pixel_paths_global': List[np.ndarray],  # 全图坐标
            'colors': List[np.ndarray],  # BGR颜色
            'num_curves': int,
            'stats': Dict
        }
    """
    H, W = image.shape[:2]

    print(f"\n{'='*60}")
    print(f"[ColorExtract] 开始提取彩色曲线")
    print(f"[ColorExtract] 原始图像尺寸: {W}x{H}")

    # 1. ROI裁剪（简单保守）
    if roi is None:
        # 默认去掉边缘10%
        roi = (int(W*0.1), int(H*0.1), int(W*0.9), int(H*0.9))
        print(f"[ColorExtract] ROI模式: 自动")
    else:
        print(f"[ColorExtract] ROI模式: 手动")

    x1, y1, x2, y2 = roi
    roi_image = image[y1:y2, x1:x2].copy()
    print(f"[ColorExtract] ROI: {roi}, 尺寸: {roi_image.shape[:2][::-1]}")

    # 2. 图像类型判定
    image_type, type_debug = detect_image_type(roi_image)

    # 3. 检测plot_bbox
    plot_bbox = detect_plot_bbox_simple(roi_image)
    plot_x1, plot_y1, plot_x2, plot_y2 = plot_bbox

    # 4. 根据图像类型选择前景提取策略
    if image_type == 'grayscale':
        print(f"[ColorExtract] 灰度图分支：使用细线提取")
        foreground = extract_thin_line_foreground(roi_image, plot_bbox)
    else:
        print(f"[ColorExtract] 彩色图分支：使用颜色聚类")
        # 原有的简单阈值方法（仅用于彩色图）
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        foreground = cv2.bitwise_not(binary)

        # 限制在plot_bbox内
        mask_outside_plot = np.ones_like(foreground) * 255
        mask_outside_plot[plot_y1:plot_y2, plot_x1:plot_x2] = 0
        foreground = cv2.bitwise_and(foreground, cv2.bitwise_not(mask_outside_plot))

    # 5. 颜色聚类或灰度合并
    pixels = roi_image[foreground > 0]
    if len(pixels) < 100:
        print(f"[ColorExtract] ✗ 前景像素不足: {len(pixels)}")
        return {
            'roi': roi,
            'roi_image': roi_image,
            'original_image': image,
            'pixel_paths_roi': [],
            'pixel_paths_global': [],
            'color_masks': [],
            'separated_masks': [],
            'num_curves': 0,
            'stats': {'error': 'no_foreground', 'image_type': image_type}
        }

    print(f"[ColorExtract] 前景像素: {len(pixels)}")

    if image_type == 'grayscale':
        # 灰度图：不做颜色聚类，直接把foreground当成单一mask
        print(f"[ColorExtract] 灰度图：跳过颜色聚类，直接使用前景mask")
        color_masks = [foreground]
        colors_bgr = [np.array([0, 0, 0])]  # 黑色
    else:
        # 彩色图：颜色聚类
        print(f"[ColorExtract] 颜色聚类...")
        pixels_lab = cv2.cvtColor(pixels.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3)
        n_clusters = min(n_colors, max(2, len(pixels)//100))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels_lab)

        print(f"[ColorExtract] 颜色中心数: {kmeans.n_clusters}")

        # 为每个颜色生成mask
        roi_lab = cv2.cvtColor(roi_image, cv2.COLOR_BGR2LAB)
        color_masks = []
        colors_bgr = []

        for i, center in enumerate(kmeans.cluster_centers_):
            diff = roi_lab.astype(np.float32) - center.reshape(1,1,3)
            distance = np.sqrt(np.sum(diff**2, axis=2))
            color_mask = (distance < 40).astype(np.uint8) * 255

            # 过滤小噪声
            kernel = np.ones((3,3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

            pixel_count = np.sum(color_mask > 0)
            if pixel_count > 500:  # 至少500像素
                color_masks.append(color_mask)
                # 转回BGR用于可视化
                center_bgr = cv2.cvtColor(center.reshape(1,1,3).astype(np.uint8),
                                         cv2.COLOR_LAB2BGR)[0,0]
                colors_bgr.append(center_bgr)
                print(f"  颜色{len(color_masks)}: {pixel_count} 像素, BGR={center_bgr}")

    print(f"[ColorExtract] 有效颜色数: {len(color_masks)}")

    # 6. 连通域分离
    print(f"[ColorExtract] 连通域分离...")
    all_separated_masks = []
    for i, mask in enumerate(color_masks):
        separated = separate_curves_by_connectivity(mask, min_size=300)
        print(f"  颜色{i+1}: {len(separated)} 个连通域")
        all_separated_masks.extend(separated)

    print(f"[ColorExtract] 总连通域数: {len(all_separated_masks)}")

    # 7. 骨架化 + 追踪
    print(f"[ColorExtract] 骨架化 + 追踪...")
    pixel_paths_roi = []

    # 使用trace_multiple_curves统一处理
    traced_paths = trace_multiple_curves(all_separated_masks, enable_smooth=False)

    # 完整性检查
    roi_w = x2 - x1
    for path in traced_paths:
        if len(path) < 50:
            print(f"  路径过短: {len(path)} 个点 ✗")
            continue

        # 横向覆盖率检查
        x_span = path[:, 0].max() - path[:, 0].min()
        coverage = x_span / roi_w
        if coverage < 0.15:  # 至少覆盖15%宽度
            print(f"  路径覆盖不足: {len(path)} 个点, x范围=[{path[:,0].min()},{path[:,0].max()}], 覆盖={coverage:.1%} ✗")
            continue

        pixel_paths_roi.append(path)
        print(f"  路径{len(pixel_paths_roi)}: {len(path)} 个点, x范围=[{path[:,0].min()},{path[:,0].max()}], 覆盖={coverage:.1%} ✓")

    print(f"[ColorExtract] 最终曲线数: {len(pixel_paths_roi)}")

    # 8. 转换为全图坐标
    pixel_paths_global = []
    for path in pixel_paths_roi:
        path_global = path.copy()
        path_global[:, 0] += x1
        path_global[:, 1] += y1
        pixel_paths_global.append(path_global)

    print(f"{'='*60}\n")

    return {
        'roi': roi,
        'roi_image': roi_image,
        'original_image': image,
        'color_masks': color_masks,
        'separated_masks': all_separated_masks,
        'pixel_paths_roi': pixel_paths_roi,
        'pixel_paths_global': pixel_paths_global,
        'pixel_paths': pixel_paths_roi,  # 兼容旧代码
        'colors': colors_bgr,
        'num_curves': len(pixel_paths_roi),
        'stats': {
            'image_type': image_type,
            'n_colors': len(color_masks),
            'n_components': len(all_separated_masks),
            'n_curves': len(pixel_paths_roi),
            'foreground_pixels': len(pixels),
            'plot_bbox': plot_bbox
        }
    }
