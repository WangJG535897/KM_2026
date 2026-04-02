"""几何工具"""
import numpy as np
from typing import Tuple


def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算两点距离"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_in_rect(point: Tuple[int, int], rect: Tuple[int, int, int, int]) -> bool:
    """判断点是否在矩形内"""
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh
