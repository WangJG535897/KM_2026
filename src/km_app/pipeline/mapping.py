"""坐标映射"""
import numpy as np
from typing import Tuple, Dict


class CoordinateMapper:
    """像素坐标到图表坐标的映射器"""

    def __init__(self, roi: Tuple[int, int, int, int],
                 x_range: Tuple[float, float],
                 y_range: Tuple[float, float] = (0.0, 100.0)):
        """
        Args:
            roi: (x1, y1, x2, y2) ROI区域
            x_range: (x_min, x_max) X轴范围（时间）
            y_range: (y_min, y_max) Y轴范围（生存率）
        """
        self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = roi
        self.roi_w = self.roi_x2 - self.roi_x1
        self.roi_h = self.roi_y2 - self.roi_y1
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range

    def pixel_to_chart(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """像素坐标转图表坐标（输入为ROI局部坐标）"""
        # 归一化到[0, 1]
        norm_x = pixel_x / self.roi_w
        norm_y = 1.0 - (pixel_y / self.roi_h)  # Y轴翻转

        # 映射到图表坐标
        chart_x = self.x_min + norm_x * (self.x_max - self.x_min)
        chart_y = self.y_min + norm_y * (self.y_max - self.y_min)

        return chart_x, chart_y

    def path_to_chart_coords(self, path: np.ndarray) -> np.ndarray:
        """将路径转换为图表坐标"""
        if len(path) == 0:
            return np.array([])

        chart_coords = []
        for px, py in path:
            cx, cy = self.pixel_to_chart(px, py)
            chart_coords.append([cx, cy])

        return np.array(chart_coords)

    def batch_paths_to_chart(self, paths: list) -> list:
        """批量转换路径"""
        return [self.path_to_chart_coords(path) for path in paths]
