"""可视化工具"""
import cv2
import numpy as np
from typing import List, Tuple


def draw_curves_on_image(image: np.ndarray, paths: List[np.ndarray],
                         colors: List[Tuple[int, int, int]] = None,
                         thickness: int = 2) -> np.ndarray:
    """在图像上绘制曲线"""
    result = image.copy()

    if colors is None:
        # 默认颜色
        colors = [
            (255, 0, 0),    # 蓝
            (0, 255, 0),    # 绿
            (0, 0, 255),    # 红
            (255, 255, 0),  # 青
            (255, 0, 255),  # 品红
        ]

    for i, path in enumerate(paths):
        if len(path) == 0:
            continue

        color = colors[i % len(colors)]

        # 绘制路径
        for j in range(len(path) - 1):
            pt1 = tuple(path[j].astype(int))
            pt2 = tuple(path[j + 1].astype(int))
            cv2.line(result, pt1, pt2, color, thickness)

    return result


def create_mask_visualization(masks: List[np.ndarray], image_shape: Tuple[int, int]) -> np.ndarray:
    """创建mask可视化"""
    h, w = image_shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]

    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        vis[mask > 0] = color

    return vis


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """创建叠加图"""
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
    return overlay
