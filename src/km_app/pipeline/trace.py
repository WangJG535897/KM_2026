"""曲线路径追踪 - 增强版"""
import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def extract_skeleton(mask: np.ndarray) -> np.ndarray:
    """提取骨架"""
    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    return skeleton


def trace_curve_column_scan(mask: np.ndarray, start_x: Optional[int] = None, max_jump: int = 15) -> np.ndarray:
    """列扫描追踪曲线路径 - 增强版"""
    h, w = mask.shape

    if start_x is None:
        start_x = 0

    # 存储路径点
    path_points = []

    # 初始化：找第一列的起点
    col = mask[:, start_x]
    ys = np.where(col > 0)[0]

    if len(ys) == 0:
        # 尝试向右找第一个有点的列
        for x in range(start_x, min(start_x + 50, w)):
            col = mask[:, x]
            ys = np.where(col > 0)[0]
            if len(ys) > 0:
                start_x = x
                break

    if len(ys) == 0:
        return np.array([])

    # 选择最上方的点作为起点（KM曲线从高生存率开始）
    current_y = ys[0]
    path_points.append([start_x, current_y])

    # 逐列扫描
    gap_count = 0
    max_gap = 5  # 允许最多5列gap

    for x in range(start_x + 1, w):
        col = mask[:, x]
        ys = np.where(col > 0)[0]

        if len(ys) == 0:
            # 当前列没有点
            gap_count += 1
            if gap_count > max_gap:
                break  # gap太大，停止
            # 保持当前y继续
            path_points.append([x, current_y])
            continue

        # 重置gap计数
        gap_count = 0

        # 找距离当前y最近的点
        distances = np.abs(ys - current_y)
        best_idx = np.argmin(distances)
        next_y = ys[best_idx]

        # 限制跳变幅度
        if abs(next_y - current_y) > max_jump:
            # 跳变过大，在小范围内搜索
            search_range = range(max(0, current_y - max_jump),
                               min(h, current_y + max_jump + 1))
            candidates = [y for y in search_range if col[y] > 0]

            if candidates:
                next_y = min(candidates, key=lambda y: abs(y - current_y))
            else:
                # 没有合适的点，可能是跳到另一条曲线了，停止
                break

        path_points.append([x, next_y])
        current_y = next_y

    return np.array(path_points) if len(path_points) > 10 else np.array([])


def trace_curve_ridge(mask: np.ndarray) -> np.ndarray:
    """基于距离变换的ridge追踪"""
    # 距离变换
    dist = distance_transform_edt(mask)

    # 找ridge（局部最大值）
    ridge = np.zeros_like(mask)
    h, w = mask.shape

    for x in range(w):
        col_dist = dist[:, x]
        if col_dist.max() > 0:
            # 找该列的最大值位置
            max_y = np.argmax(col_dist)
            ridge[max_y, x] = 1

    # 提取路径点
    ys, xs = np.where(ridge > 0)
    if len(xs) == 0:
        return np.array([])

    # 按x排序
    sorted_indices = np.argsort(xs)
    path_points = np.stack([xs[sorted_indices], ys[sorted_indices]], axis=1)

    return path_points if len(path_points) > 10 else np.array([])


def smooth_path(path: np.ndarray, window_size: int = 3, aggressive: bool = False) -> np.ndarray:
    """平滑路径 - 保守版，仅修复明显噪声

    Args:
        path: 路径点 [N, 2]
        window_size: 窗口大小
        aggressive: 是否激进平滑（破坏KM台阶）

    Returns:
        平滑后的路径
    """
    if len(path) < window_size:
        return path

    if not aggressive:
        # 保守模式：只修复明显的单点跳变
        smoothed = path.copy().astype(np.int32)

        for i in range(1, len(path) - 1):
            y_prev = path[i - 1, 1]
            y_curr = path[i, 1]
            y_next = path[i + 1, 1]

            # 检测单点跳变（前后都不同，且跳变幅度>5）
            if abs(y_curr - y_prev) > 5 and abs(y_curr - y_next) > 5:
                # 如果前后y相近，说明是噪声
                if abs(y_prev - y_next) <= 3:
                    smoothed[i, 1] = (y_prev + y_next) // 2

        return smoothed
    else:
        # 激进模式：移动平均（破坏KM台阶）
        smoothed = path.copy().astype(np.float32)

        for i in range(len(path)):
            start = max(0, i - window_size // 2)
            end = min(len(path), i + window_size // 2 + 1)
            smoothed[i, 1] = np.mean(path[start:end, 1])

        return smoothed.astype(np.int32)


def separate_overlapping_paths(paths: List[np.ndarray], min_distance: int = 10) -> List[np.ndarray]:
    """分离重叠路径 - 确保多条曲线之间有最小间距"""
    if len(paths) <= 1:
        return paths

    # 按平均y坐标排序（从上到下）
    paths_with_avg_y = [(path, np.mean(path[:, 1])) for path in paths]
    paths_with_avg_y.sort(key=lambda x: x[1])

    separated = [paths_with_avg_y[0][0]]

    for i in range(1, len(paths_with_avg_y)):
        current_path = paths_with_avg_y[i][0]
        prev_path = separated[-1]

        # 检查是否太接近
        # 计算重叠x范围内的平均y距离
        x_min = max(current_path[:, 0].min(), prev_path[:, 0].min())
        x_max = min(current_path[:, 0].max(), prev_path[:, 0].max())

        if x_max > x_min:
            # 有重叠x范围
            overlap_x = range(x_min, x_max + 1)
            distances = []
            for x in overlap_x:
                y1_candidates = current_path[current_path[:, 0] == x, 1]
                y2_candidates = prev_path[prev_path[:, 0] == x, 1]
                if len(y1_candidates) > 0 and len(y2_candidates) > 0:
                    distances.append(abs(y1_candidates[0] - y2_candidates[0]))

            if len(distances) > 0:
                avg_distance = np.mean(distances)
                if avg_distance < min_distance:
                    # 太接近，跳过
                    continue

        separated.append(current_path)

    return separated


def trace_multiple_curves(masks: List[np.ndarray], enable_smooth: bool = False) -> List[np.ndarray]:
    """追踪多条曲线 - 增强版

    Args:
        masks: mask列表
        enable_smooth: 是否启用平滑（默认False以保留KM台阶）

    Returns:
        路径列表
    """
    all_paths = []

    for mask in masks:
        # 先提取skeleton
        skeleton = extract_skeleton(mask)

        if np.sum(skeleton) < 50:
            # skeleton太小，直接用原mask
            skeleton = mask

        # 尝试列扫描
        path = trace_curve_column_scan(skeleton, max_jump=15)

        if len(path) == 0:
            # 如果失败，尝试ridge方法
            path = trace_curve_ridge(mask)

        if len(path) > 0:
            # 仅在启用时才平滑（保守模式）
            if enable_smooth:
                path = smooth_path(path, window_size=3, aggressive=False)
            all_paths.append(path)

    # 分离重叠路径
    separated_paths = separate_overlapping_paths(all_paths, min_distance=10)

    return separated_paths
