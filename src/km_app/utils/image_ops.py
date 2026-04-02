"""图像操作工具"""
import cv2
import numpy as np


def resize_keep_aspect(image: np.ndarray, target_size: int) -> np.ndarray:
    """保持比例resize"""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """确保图像是uint8类型"""
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return image
