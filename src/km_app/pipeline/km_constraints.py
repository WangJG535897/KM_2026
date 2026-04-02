"""KM曲线约束 - 修正图像坐标系方向

关键原则：
- 图像坐标系：y越小=越靠上=生存率越高，y越大=越靠下=生存率越低
- KM曲线随时间（x增加）：y应持平或增大（向下或水平），不应明显减小（向上跳）
- 允许水平平台，允许向下台阶，不允许明显回升
"""
import numpy as np
from typing import List


def enforce_monotonic_decreasing(path: np.ndarray, tolerance: int = 3) -> np.ndarray:
    """强制单调递减（图像坐标系：y不减小）

    在图像坐标系中，KM曲线应该y持平或增大，不应明显减小
    """
    if len(path) < 2:
        return path

    corrected = path.copy()

    for i in range(1, len(corrected)):
        # 如果当前点y明显小于前一点（向上跳），修正它
        if corrected[i, 1] < corrected[i-1, 1] - tolerance:
            corrected[i, 1] = corrected[i-1, 1]

    return corrected


def enforce_start_at_top(path: np.ndarray, top_percentile: float = 0.1) -> np.ndarray:
    """强制起点在顶部区域"""
    if len(path) == 0:
        return path

    constrained = path.copy()

    # 获取图像高度范围
    min_y = path[:, 1].min()
    max_y = path[:, 1].max()
    y_range = max_y - min_y

    # 起点应该在顶部10%区域
    top_threshold = min_y + y_range * top_percentile

    if constrained[0, 1] > top_threshold:
        # 起点太低，调整到顶部
        constrained[0, 1] = int(min_y)

    return constrained


def enforce_step_like(path: np.ndarray, platform_threshold: int = 5) -> np.ndarray:
    """强制step-like结构

    KM曲线应该是水平平台+向下台阶的组合
    """
    if len(path) < 3:
        return path

    corrected = path.copy()

    # 识别平台和台阶
    i = 0
    while i < len(corrected) - 1:
        # 找到下一个明显变化点
        j = i + 1
        while j < len(corrected) and abs(corrected[j, 1] - corrected[i, 1]) < platform_threshold:
            j += 1

        # 将i到j之间的点设为平台
        if j > i + 1:
            avg_y = np.mean(corrected[i:j, 1])
            corrected[i:j, 1] = int(avg_y)

        i = j

    return corrected


def remove_outliers(path: np.ndarray, window_size: int = 5, threshold: float = 2.0) -> np.ndarray:
    """移除异常点"""
    if len(path) < window_size:
        return path

    corrected = path.copy()

    for i in range(window_size // 2, len(path) - window_size // 2):
        window = path[i - window_size // 2: i + window_size // 2 + 1, 1]
        median = np.median(window)
        std = np.std(window)

        if abs(path[i, 1] - median) > threshold * std:
            corrected[i, 1] = int(median)

    return corrected


def fill_gaps(path: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """填充小gap"""
    if len(path) < 2:
        return path

    filled = [path[0]]

    for i in range(1, len(path)):
        gap = path[i, 0] - path[i-1, 0]

        if gap > 1 and gap <= max_gap:
            # 线性插值填充
            for x in range(path[i-1, 0] + 1, path[i, 0]):
                alpha = (x - path[i-1, 0]) / gap
                y = int(path[i-1, 1] * (1 - alpha) + path[i, 1] * alpha)
                filled.append([x, y])

        filled.append(path[i])

    return np.array(filled)


def apply_km_constraints(path: np.ndarray) -> np.ndarray:
    """应用KM约束到单条路径"""
    if len(path) < 2:
        return path

    # 1. 移除异常点
    path = remove_outliers(path, window_size=5, threshold=2.0)

    # 2. 强制起点在顶部
    path = enforce_start_at_top(path, top_percentile=0.1)

    # 3. 强制单调（y不减小）
    path = enforce_monotonic_decreasing(path, tolerance=3)

    # 4. 强制step-like
    path = enforce_step_like(path, platform_threshold=5)

    # 5. 填充小gap
    path = fill_gaps(path, max_gap=10)

    return path


def apply_km_constraints_batch(paths: List[np.ndarray]) -> List[np.ndarray]:
    """批量应用KM约束"""
    return [apply_km_constraints(path) for path in paths if len(path) > 0]
