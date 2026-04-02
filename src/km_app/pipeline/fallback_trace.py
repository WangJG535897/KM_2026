"""Ridge路径追踪 - 从概率热图直接提取脊线（主力方案）"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import logging

logger = logging.getLogger(__name__)


def suppress_horizontal_reference_lines(
    prob_map: np.ndarray,
    min_horizontal_coverage: float = 0.5,
    y_std_threshold: float = 5.0,
    prob_threshold: float = 0.15,
    decay_factor: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    抑制长水平干扰线（如 50% 参考虚线、x轴等）

    输入：
      prob_map (H, W): float32 [0,1]，ROI内的sigmoid输出
      min_horizontal_coverage: 如果某行高概率覆盖 > W * 此比例，认为是水平线
      y_std_threshold: 如果某行周围±10像素的y方差 < 此值，认为是水平
      prob_threshold: 计算覆盖时的概率阈值
      decay_factor: 对水平线的概率衰减系数

    输出：
      suppressed_prob_map (H, W): 被衰减的概率图
      suppression_mask (H, W): bool，被抑制的像素位置（True = 被抑制）
    """
    H, W = prob_map.shape
    suppressed_prob_map = prob_map.copy()
    suppression_mask = np.zeros_like(prob_map, dtype=bool)

    suppressed_rows = []
    coverage_before = []
    coverage_after = []

    logger.info(f"[suppress_horizontal] 开始检测水平参考线...")

    for y in range(H):
        # 1. 计算该行高概率像素的连续覆盖长度
        row_prob = prob_map[y, :]
        high_prob_mask = row_prob > prob_threshold

        # 计算连续覆盖长度（最长连续段）
        max_continuous = 0
        current_continuous = 0
        for val in high_prob_mask:
            if val:
                current_continuous += 1
                max_continuous = max(max_continuous, current_continuous)
            else:
                current_continuous = 0

        coverage_ratio = max_continuous / W

        # 2. 如果覆盖率超过阈值，进入候选
        if coverage_ratio > min_horizontal_coverage:
            # 3. 检查周围±10行的y方向方差
            y_min = max(0, y - 10)
            y_max = min(H, y + 10)
            region = prob_map[y_min:y_max, :]

            # 计算该区域每列的加权中心y坐标
            weighted_y_list = []
            for x in range(W):
                col_region = region[:, x]
                if col_region.sum() > 0:
                    weights = col_region / col_region.sum()
                    weighted_y = np.sum(weights * np.arange(len(col_region)))
                    weighted_y_list.append(weighted_y)

            if len(weighted_y_list) > 0:
                y_std = np.std(weighted_y_list)
            else:
                y_std = 0

            # 4. 如果y_std小于阈值，确认为水平线
            if y_std < y_std_threshold:
                # 抑制该行
                suppressed_prob_map[y, :] *= decay_factor
                suppression_mask[y, :] = True
                suppressed_rows.append(y)

                coverage_before.append(coverage_ratio)
                coverage_after.append(coverage_ratio * decay_factor)

                logger.info(f"  抑制行 y={y}: coverage={coverage_ratio:.2%}, y_std={y_std:.2f}, "
                          f"avg_prob_before={row_prob.mean():.3f}, avg_prob_after={suppressed_prob_map[y, :].mean():.3f}")

    # 日志输出
    if suppressed_rows:
        logger.info(f"[suppress_horizontal] ✓ 抑制了 {len(suppressed_rows)} 行水平线")
        logger.info(f"  行号: {suppressed_rows}")
        logger.info(f"  覆盖率 before: min={min(coverage_before):.2%}, max={max(coverage_before):.2%}, mean={np.mean(coverage_before):.2%}")
        logger.info(f"  覆盖率 after: min={min(coverage_after):.2%}, max={max(coverage_after):.2%}, mean={np.mean(coverage_after):.2%}")
    else:
        logger.info(f"[suppress_horizontal] 未检测到明显水平线")

    return suppressed_prob_map, suppression_mask


def _find_all_start_seeds(prob_map: np.ndarray, used_mask: np.ndarray,
                          left_width: int, min_prob: float,
                          min_distance: int) -> List[Tuple[int, int, float]]:
    """
    在左侧left_width宽度范围内，找所有候选起点(col, y, prob)
    返回按概率倒序的列表：[(col1, y1, prob1), (col2, y2, prob2), ...]
    """
    H, W = prob_map.shape
    candidates = []

    # 对左侧每一列
    for col in range(min(left_width, W)):
        col_prob = prob_map[:, col] * (1 - used_mask[:, col].astype(float))
        col_smooth = gaussian_filter1d(col_prob, sigma=1.5)

        # 在该列找所有局部最大值
        peaks, properties = find_peaks(
            col_smooth,
            height=min_prob,
            prominence=0.05,
            distance=min_distance
        )

        # 对每个峰，记录(col, y, prob)
        for peak_idx in peaks:
            candidates.append((col, peak_idx, col_prob[peak_idx]))

    # 按概率倒序排列
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def _dp_trace_single_path(prob_map: np.ndarray, used_mask: np.ndarray,
                          start_col: int, start_y: int,
                          penalties: Dict[str, float],
                          max_y_jump: int, width_expand: int) -> Optional[np.ndarray]:
    """
    从(start_col, start_y)出发，向右DP追踪单条路径。
    返回路径 (N, 2) array [[col1, y1], [col2, y2], ...]
    如果追踪失败返回None。
    """
    H, W = prob_map.shape

    # DP表：dp[y] = 最小成本, prev[y] = 前驱y
    INF = float('inf')
    dp = [INF] * H
    prev = [-1] * H

    # 初始化起点
    dp[start_y] = 0

    # 记录每列的DP状态
    dp_history = [dp.copy()]
    prev_history = [prev.copy()]

    # 从start_col向右逐列推进
    for col in range(start_col, min(W, start_col + width_expand)):
        new_dp = [INF] * H
        new_prev = [-1] * H

        for y in range(H):
            if dp[y] == INF:
                continue

            # 从(col, y)考虑跳到(col+1, y')
            for dy in range(-max_y_jump, max_y_jump + 1):
                y_next = y + dy
                if not (0 <= y_next < H):
                    continue

                # 如果下一列超出范围，跳过
                if col + 1 >= W:
                    continue

                # 计算代价
                prob_val = max(prob_map[y_next, col + 1], 0.01)
                prob_cost = -np.log(prob_val) * penalties['low_prob']

                jump_cost = abs(dy) * penalties['jump'] if dy != 0 else 0

                # 水平倾向：if dy==0, 加penalty（KM曲线应该下降）
                horiz_cost = penalties['horizontal'] if dy == 0 else 0

                # 避开used区域
                avoid_cost = INF if used_mask[y_next, col + 1] else 0

                total_cost = dp[y] + prob_cost + jump_cost + horiz_cost + avoid_cost

                if total_cost < new_dp[y_next]:
                    new_dp[y_next] = total_cost
                    new_prev[y_next] = y

        dp = new_dp
        prev = new_prev
        dp_history.append(dp.copy())
        prev_history.append(prev.copy())

    # 回溯找最优终点和路径
    end_y = min(range(H), key=lambda y: dp[y])
    if dp[end_y] == INF:
        return None

    # 反向回溯构建路径
    path = []
    y = end_y
    for col_idx in range(len(dp_history) - 1, -1, -1):
        col = start_col + col_idx
        if col >= W:
            continue
        path.append([col, y])
        if col_idx == 0:
            break
        y = prev_history[col_idx][y]
        if y == -1:
            break

    path.reverse()
    return np.array(path, dtype=np.float32) if len(path) > 0 else None


def _score_path(path: np.ndarray, prob_map: np.ndarray, roi_width: int) -> Tuple[Dict, bool]:
    """
    对路径进行多维度评分。
    返回 (score_dict, is_valid_bool)
    """
    if len(path) == 0:
        return {}, False

    H, W = prob_map.shape

    # 计算覆盖宽度
    x_min, x_max = path[:, 0].min(), path[:, 0].max()
    width_coverage = (x_max - x_min + 1) / roi_width

    # 计算平均概率
    path_int = path.astype(int)
    path_int[:, 0] = np.clip(path_int[:, 0], 0, W - 1)
    path_int[:, 1] = np.clip(path_int[:, 1], 0, H - 1)
    path_probs = prob_map[path_int[:, 1], path_int[:, 0]]
    avg_prob = np.mean(path_probs)

    # Y方向统计
    y_range = path[:, 1].max() - path[:, 1].min()
    y_std = np.std(path[:, 1])

    # 检查是否几乎完全水平（坏特征）
    dy_list = np.diff(path[:, 1])
    horizontal_count = np.sum(np.abs(dy_list) < 0.5)
    horizontal_ratio = horizontal_count / len(dy_list) if len(dy_list) > 0 else 0

    score_dict = {
        'width_coverage': width_coverage,
        'avg_prob': avg_prob,
        'y_range': y_range,
        'y_std': y_std,
        'horizontal_ratio': horizontal_ratio,
        'path_length': len(path)
    }

    # 质量检查
    is_valid = (
        width_coverage >= 0.35 and          # 覆盖至少35%宽度
        avg_prob >= 0.15 and               # 平均概率足够
        y_range >= 5 and                   # Y方向不能完全水平
        horizontal_ratio < 0.7             # 不能有超过70%的点是水平的
    )

    score_dict['final_score'] = (
        width_coverage * 100 +
        avg_prob * 50 -
        horizontal_ratio * 30
    )

    return score_dict, is_valid


def _suppress_around_path(used_mask: np.ndarray, path: np.ndarray, radius: int):
    """
    在used_mask中，沿path周围radius像素内标记为True（已使用）。
    用于下一轮搜索时避开此区域。
    """
    path_int = path.astype(int)
    H, W = used_mask.shape

    for x, y in path_int:
        x = int(np.clip(x, 0, W - 1))
        y = int(np.clip(y, 0, H - 1))
        y_min, y_max = max(0, y - radius), min(H, y + radius + 1)
        x_min, x_max = max(0, x - radius), min(W, x + radius + 1)
        used_mask[y_min:y_max, x_min:x_max] = True


def _min_distance_between_paths(path1: np.ndarray, path2: np.ndarray) -> float:
    """
    计算两条路径之间的最小点对距离。
    """
    min_dist = float('inf')
    # 采样以提高效率
    sample_step = max(1, len(path1) // 20)
    for i in range(0, len(path1), sample_step):
        p1 = path1[i]
        for j in range(0, len(path2), sample_step):
            p2 = path2[j]
            dist = np.linalg.norm(p1 - p2)
            min_dist = min(min_dist, dist)
    return min_dist


def trace_from_prob_map_ridge(prob_map: np.ndarray, num_curves: int = 3,
                               min_prob: float = 0.2, min_distance: int = 15) -> Tuple[List[np.ndarray], Dict]:
    """
    从概率热图直接提取多条脊线路径，通过逐条提取+带状抑制避免强干扰线压制弱曲线。

    Args:
        prob_map: 概率图 [H, W], float32 [0,1]
        num_curves: 期望提取的曲线数量
        min_prob: 最小概率阈值
        min_distance: 曲线间最小距离

    Returns:
        (extracted_paths, debug_info)
        - extracted_paths: 路径列表
        - debug_info: 调试信息字典
    """
    H, W = prob_map.shape
    logger.info(f"\n[fallback_trace] 开始Ridge脊线提取...")
    logger.info(f"[fallback_trace] 概率图尺寸: {W}x{H}")

    # 第1步：水平参考线抑制
    suppressed_prob_map, suppression_mask = suppress_horizontal_reference_lines(prob_map)

    # 第2步：初始化
    extracted_paths = []
    used_mask = np.zeros_like(prob_map, dtype=bool)

    # 调试日志初始化
    debug_log = {
        'suppression': {
            'rows_suppressed': np.where(suppression_mask.any(axis=1))[0].tolist(),
            'mean_coverage_before': float(prob_map.mean()),
            'mean_coverage_after': float(suppressed_prob_map.mean())
        },
        'rounds': []
    }

    # 第3步：逐条提取路径
    for round_idx in range(num_curves):
        logger.info(f"\n[fallback_trace] === Ridge extraction round {round_idx + 1} ===")

        # 3.1 在左侧15%宽度范围内寻找所有候选起点
        left_width = max(int(W * 0.15), 20)
        candidates = _find_all_start_seeds(
            prob_map=suppressed_prob_map,
            used_mask=used_mask,
            left_width=left_width,
            min_prob=min_prob,
            min_distance=min_distance
        )

        if not candidates or candidates[0][2] < min_prob:
            # 没有足够好的候选，停止
            logger.info(f"[fallback_trace] 无足够候选起点，停止提取")
            break

        # 3.2 选择最强候选
        best_col, best_y, best_prob = candidates[0]

        round_log = {
            'round': round_idx + 1,
            'start_col': int(best_col),
            'start_y': int(best_y),
            'start_prob': float(best_prob),
            'candidates_count': len(candidates)
        }

        logger.info(f"[fallback_trace] 起点: col={best_col}, y={best_y}, prob={best_prob:.3f}")
        logger.info(f"[fallback_trace] 候选数: {len(candidates)}")

        # 3.3 从起点出发，DP追踪单条路径
        path = _dp_trace_single_path(
            prob_map=suppressed_prob_map,
            used_mask=used_mask,
            start_col=best_col,
            start_y=best_y,
            penalties={
                'low_prob': 1.0,      # 低概率代价
                'jump': 0.05,         # 上下跳跃代价（降低以允许KM台阶）
                'horizontal': 0.5,    # 水平倾向代价（降低）
                'gap': 0.5            # 缺失像素代价
            },
            max_y_jump=30,
            width_expand=W
        )

        if path is None or len(path) < W * 0.35:
            # 路径太短或追踪失败，跳过
            logger.info(f"[fallback_trace] 路径追踪失败或太短")
            round_log['status'] = 'failed_too_short'
            round_log['path_length'] = 0 if path is None else len(path)
            debug_log['rounds'].append(round_log)
            continue

        # 3.4 对路径进行质量评分
        score_dict, is_valid = _score_path(path, suppressed_prob_map, W)
        round_log.update({
            'path_length': len(path),
            'width_coverage': float(score_dict['width_coverage']),
            'avg_prob': float(score_dict['avg_prob']),
            'y_range': float(score_dict['y_range']),
            'y_std': float(score_dict['y_std']),
            'horizontal_ratio': float(score_dict['horizontal_ratio']),
            'is_valid': is_valid,
            'score': float(score_dict.get('final_score', 0))
        })

        logger.info(f"[fallback_trace] 路径长度: {len(path)} 像素")
        logger.info(f"[fallback_trace] 宽度覆盖: {score_dict['width_coverage']:.1%}")
        logger.info(f"[fallback_trace] 平均概率: {score_dict['avg_prob']:.3f}")
        logger.info(f"[fallback_trace] Y范围: {score_dict['y_range']:.1f} 像素")
        logger.info(f"[fallback_trace] 水平比例: {score_dict['horizontal_ratio']:.1%}")

        if not is_valid:
            logger.info(f"[fallback_trace] 质量检查: FAIL")
            round_log['status'] = 'failed_quality_check'
            debug_log['rounds'].append(round_log)
            continue

        logger.info(f"[fallback_trace] 质量检查: PASS")

        # 3.5 检查与已提路径的最小间距
        min_dist_to_others = float('inf')
        for prev_path in extracted_paths:
            dist = _min_distance_between_paths(path, prev_path)
            min_dist_to_others = min(min_dist_to_others, dist)

        if extracted_paths and min_dist_to_others < min_distance:
            logger.info(f"[fallback_trace] 与已有路径距离: {min_dist_to_others:.1f} < {min_distance}, 太近，丢弃")
            round_log['status'] = 'failed_too_close'
            round_log['min_dist_to_others'] = float(min_dist_to_others)
            debug_log['rounds'].append(round_log)
            continue

        # 3.6 通过所有检查，加入结果
        extracted_paths.append(path)
        round_log['status'] = 'accepted'
        debug_log['rounds'].append(round_log)
        logger.info(f"[fallback_trace] 状态: ACCEPTED ✓")

        # 3.7 在used_mask中做带状抑制，为下一轮腾出空间
        _suppress_around_path(used_mask, path, radius=15)

    # 调试输出
    logger.info(f"\n[fallback_trace] ✓ Ridge提取完成: {len(extracted_paths)} 条路径")
    for log in debug_log['rounds']:
        logger.info(f"  Round {log['round']}: {log['status']}, "
                   f"start=({log['start_col']},{log['start_y']}), "
                   f"coverage={log.get('width_coverage', 'N/A'):.2%}, "
                   f"avg_prob={log.get('avg_prob', 'N/A'):.3f}")

    return extracted_paths, {
        'suppressed_prob_map': suppressed_prob_map,
        'suppression_mask': suppression_mask,
        'used_mask': used_mask,
        'debug_log': debug_log
    }
