"""ROI处理"""
from typing import Tuple
import numpy as np


def validate_roi(roi: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """验证和修正ROI"""
    x, y, w, h = roi
    img_h, img_w = image_shape[:2]

    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    return (x, y, w, h)
