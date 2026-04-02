"""颜色精修和曲线分离"""
import cv2
import numpy as np
from typing import List, Tuple
from sklearn.cluster import KMeans
from scipy.ndimage import label


def extract_curve_colors(image: np.ndarray, mask: np.ndarray, n_colors: int = 5) -> List[np.ndarray]:
    """从mask区域提取主要曲线颜色"""
    # 获取mask区域的像素
    pixels = image[mask > 0]

    if len(pixels) < 10:
        return []

    # 转换到LAB空间
    pixels_lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)

    # KMeans聚类
    n_clusters = min(n_colors, len(pixels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels_lab)

    # 返回聚类中心（LAB空间）
    return kmeans.cluster_centers_


def refine_mask_by_color(image: np.ndarray, mask: np.ndarray,
                         target_color_lab: np.ndarray, threshold: float = 30.0) -> np.ndarray:
    """基于颜色精修mask"""
    # 转换到LAB空间
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 计算颜色距离
    diff = image_lab.astype(np.float32) - target_color_lab.reshape(1, 1, 3)
    distance = np.sqrt(np.sum(diff ** 2, axis=2))

    # 颜色匹配mask
    color_mask = (distance < threshold).astype(np.uint8)

    # 与原mask取交集
    refined = cv2.bitwise_and(mask, color_mask)

    return refined


def separate_curves_by_connectivity(mask: np.ndarray, min_size: int = 500) -> List[np.ndarray]:
    """通过连通域分离曲线

    Args:
        mask: 输入mask
        min_size: 最小连通域大小（像素数），过滤小噪声
    """
    # 连通域标记
    labeled, num_features = label(mask)

    curves = []
    for i in range(1, num_features + 1):
        curve_mask = (labeled == i).astype(np.uint8)
        area = np.sum(curve_mask)

        # 过滤太小的区域（噪声）
        if area >= min_size:
            curves.append(curve_mask)

    return curves


def filter_non_curve_regions(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """过滤非曲线区域（坐标轴、文字等）"""
    # 形态学操作去除小噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 去除过粗的区域（可能是坐标轴）
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thick_regions = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
    thick_regions = cv2.dilate(thick_regions, kernel_large, iterations=2)

    # 保留细长区域
    cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(thick_regions))

    return cleaned
