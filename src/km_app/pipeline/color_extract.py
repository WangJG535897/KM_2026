"""Color-first曲线提取器 - 唯一目标：从图片提取彩色线的像素轨迹"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from .trace import trace_curve_column_scan
from .color_refine import separate_curves_by_connectivity


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

    # 2. 去除明显非曲线元素（轻量）
    print(f"[ColorExtract] 前景提取...")
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_not(binary)

    # 3. 颜色聚类
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
            'stats': {'error': 'no_foreground'}
        }

    print(f"[ColorExtract] 前景像素: {len(pixels)}")
    print(f"[ColorExtract] 颜色聚类...")

    pixels_lab = cv2.cvtColor(pixels.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3)
    n_clusters = min(n_colors, max(2, len(pixels)//100))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels_lab)

    print(f"[ColorExtract] 颜色中心数: {kmeans.n_clusters}")

    # 4. 为每个颜色生成mask
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

    # 5. 连通域分离
    print(f"[ColorExtract] 连通域分离...")
    all_separated_masks = []
    for i, mask in enumerate(color_masks):
        separated = separate_curves_by_connectivity(mask, min_size=300)
        print(f"  颜色{i+1}: {len(separated)} 个连通域")
        all_separated_masks.extend(separated)

    print(f"[ColorExtract] 总连通域数: {len(all_separated_masks)}")

    # 6. 逐列追踪
    print(f"[ColorExtract] 逐列追踪...")
    pixel_paths_roi = []
    for i, mask in enumerate(all_separated_masks):
        path = trace_curve_column_scan(mask, max_jump=20)
        if len(path) > 50:  # 至少50个点
            pixel_paths_roi.append(path)
            print(f"  路径{len(pixel_paths_roi)}: {len(path)} 个点, x范围=[{path[:,0].min()},{path[:,0].max()}]")

    print(f"[ColorExtract] 最终曲线数: {len(pixel_paths_roi)}")

    # 7. 转换为全图坐标
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
            'n_colors': len(color_masks),
            'n_components': len(all_separated_masks),
            'n_curves': len(pixel_paths_roi),
            'foreground_pixels': len(pixels)
        }
    }
