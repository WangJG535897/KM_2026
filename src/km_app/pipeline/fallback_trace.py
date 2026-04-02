"""Ridge路径追踪 - 从概率热图直接提取脊线（主力方案）"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import logging

logger = logging.getLogger(__name__)


def suppress_horizontal_reference_lines(
    prob_map: np.ndarray,
    min_horizontal_coverage: float = 0.7,  # 从0.5提高到0.7，更严格
    y_std_threshold: float = 3.0,  # 从5.0降到3.0，更严格
    prob_threshold: float = 0.20,  # 从0.15提高到0.20
    decay_factor: float = 0.5  # 从0.2提高到0.5，不要压太狠
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
    print(f"[suppress_horizontal] 开始检测水平参考线...")

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
        # 只过滤底部5%（X轴区域），保留顶部和中间
        margin_bottom = int(H * 0.95)
        for peak_idx in peaks:
            if peak_idx < margin_bottom:  # 只过滤底部5%
                candidates.append((col, peak_idx, col_prob[peak_idx]))

    # 按概率倒序排列
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def _extract_column_candidates(prob_map: np.ndarray, used_mask: np.ndarray,
                               col: int, top_k: int = 5, min_prob: float = 0.05) -> List[Tuple[int, float]]:
    """
    从某列提取 top-K 候选点（y坐标，概率）

    Args:
        prob_map: 概率图
        used_mask: 已使用mask
        col: 列索引
        top_k: 保留前K个候选
        min_prob: 最小概率阈值

    Returns:
        [(y1, prob1), (y2, prob2), ...] 按概率降序
    """
    H = prob_map.shape[0]
    col_prob = prob_map[:, col] * (1 - used_mask[:, col].astype(float))

    # 找局部峰值
    col_smooth = gaussian_filter1d(col_prob, sigma=1.5)
    peaks, _ = find_peaks(col_smooth, height=min_prob, prominence=0.02, distance=10)  # 降低prominence

    if len(peaks) == 0:
        # 如果没有峰值，取最大值点
        max_y = np.argmax(col_prob)
        if col_prob[max_y] >= min_prob:
            return [(max_y, col_prob[max_y])]
        return []

    # 按概率排序，取top-K
    candidates = [(y, col_prob[y]) for y in peaks]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


def _dp_trace_single_path(prob_map: np.ndarray, used_mask: np.ndarray,
                          start_col: int, start_y: int,
                          penalties: Dict[str, float],
                          max_y_jump: int, width_expand: int,
                          timeout_seconds: float = 2.0) -> Optional[np.ndarray]:
    """
    从(start_col, start_y)出发，向右DP追踪单条路径（候选点DP优化版）

    优化：只在每列的 top-K 候选点之间做转移，而非全像素

    Returns:
        路径 (N, 2) array [[col1, y1], [col2, y2], ...] 或 None
    """
    import time
    start_time = time.time()

    H, W = prob_map.shape
    INF = float('inf')

    # 候选点DP：dp[(col, y)] = (cost, prev_y)
    dp = {}
    dp[(start_col, start_y)] = (0.0, None)

    # 逐列推进（候选点版本）
    max_cols = min(W - start_col, width_expand, 800)  # 限制最大宽度

    successful_cols = 0  # 成功推进的列数

    for offset in range(max_cols):
        col = start_col + offset

        # 超时检查
        if time.time() - start_time > timeout_seconds:
            logger.warning(f"[DP] 超时 {timeout_seconds}s，提前终止")
            break

        if col + 1 >= W:
            break

        # 获取当前列的所有活跃状态
        current_states = [(k, v) for k, v in dp.items() if k[0] == col]
        if not current_states:
            # 如果当前列没有活跃状态，说明DP断了
            if offset > 10:  # 至少推进了10列才算有效
                logger.debug(f"[DP] 在第{offset}列断开，无活跃状态")
            break

        # 获取下一列的候选点
        next_candidates = _extract_column_candidates(prob_map, used_mask, col + 1, top_k=5, min_prob=0.05)
        if not next_candidates:
            # 如果下一列没有候选，从当前状态延续
            for (c, y), (cost, _) in current_states:
                key = (col + 1, y)
                gap_cost = penalties.get('gap', 0.5)
                new_cost = cost + gap_cost
                if key not in dp or new_cost < dp[key][0]:
                    dp[key] = (new_cost, y)
            successful_cols += 1
            continue

        # 对每个当前状态，尝试转移到下一列的候选点
        transitions_made = 0
        for (c, y_curr), (cost_curr, _) in current_states:
            for y_next, prob_next in next_candidates:
                # 检查跳变是否在允许范围内
                dy = y_next - y_curr
                if abs(dy) > max_y_jump:
                    continue

                # 计算代价
                prob_val = max(prob_next, 0.01)
                prob_cost = -np.log(prob_val) * penalties['low_prob']
                jump_cost = abs(dy) * penalties['jump'] if dy != 0 else 0
                horiz_cost = penalties['horizontal'] if dy == 0 else 0

                # 避开used区域
                if used_mask[y_next, col + 1]:
                    continue

                total_cost = cost_curr + prob_cost + jump_cost + horiz_cost

                key = (col + 1, y_next)
                if key not in dp or total_cost < dp[key][0]:
                    dp[key] = (total_cost, y_curr)
                    transitions_made += 1

        if transitions_made > 0:
            successful_cols += 1

    logger.debug(f"[DP] 成功推进 {successful_cols}/{max_cols} 列")
    print(f"  [DP] 成功推进 {successful_cols}/{max_cols} 列, DP表大小={len(dp)}")

    # 回溯找最优终点
    if not dp:
        logger.debug(f"[DP] DP表为空，追踪失败")
        print(f"  [DP] DP表为空，追踪失败")
        return None

    last_col = max([k[0] for k in dp.keys()])
    end_states = [(k, v) for k, v in dp.items() if k[0] == last_col]

    if not end_states:
        logger.debug(f"[DP] 无终点状态，DP失败 (last_col={last_col}, dp_size={len(dp)})")
        print(f"  [DP] 无终点状态 (last_col={last_col}, dp_size={len(dp)})")
        return None

    logger.debug(f"[DP] 找到终点: last_col={last_col}, 候选数={len(end_states)}")
    print(f"  [DP] 找到终点: last_col={last_col}, 候选数={len(end_states)}")

    # 选择成本最低的终点
    best_end = min(end_states, key=lambda x: x[1][0])
    end_col, end_y = best_end[0]

    # 回溯路径
    path = []
    col, y = end_col, end_y

    while col >= start_col:
        path.append([col, y])
        if (col, y) not in dp:
            break
        _, prev_y = dp[(col, y)]
        if prev_y is None:
            break
        col -= 1
        y = prev_y

    path.reverse()

    logger.debug(f"[DP] 回溯路径长度: {len(path)}, 需要>{max_cols * 0.3:.0f}")
    print(f"  [DP] 回溯路径长度: {len(path)}, 需要>{max_cols * 0.3:.0f}")

    # 检查路径长度
    if len(path) < max_cols * 0.3:  # 至少覆盖30%
        logger.debug(f"[DP] 路径太短: {len(path)} < {max_cols * 0.3:.0f}")
        print(f"  [DP] 路径太短")
        return None

    return np.array(path, dtype=np.float32)


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

    优化版：候选点DP + 曲线数限制 + 超时保护
    """
    H, W = prob_map.shape
    print(f"\n[fallback_trace] 开始Ridge脊线提取（优化版）...")
    print(f"[fallback_trace] 概率图尺寸: {W}x{H}")
    logger.info(f"\n[fallback_trace] 开始Ridge脊线提取（优化版）...")
    logger.info(f"[fallback_trace] 概率图尺寸: {W}x{H}")

    # 强制限制曲线数（防止过度提取）
    num_curves = max(1, min(num_curves, 3))
    print(f"[fallback_trace] 限制最大曲线数: {num_curves}")
    logger.info(f"[fallback_trace] 限制最大曲线数: {num_curves}")

    # 第1步：水平参考线抑制
    print(f"[fallback_trace] 开始水平线抑制...")
    suppressed_prob_map, suppression_mask = suppress_horizontal_reference_lines(prob_map)
    print(f"[fallback_trace] 水平线抑制完成")

    # 第2步：初始化
    extracted_paths = []
    used_mask = np.zeros_like(prob_map, dtype=bool)

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
        logger.info(f"\n[fallback_trace] === Ridge extraction round {round_idx + 1}/{num_curves} ===")

        # 3.1 在左侧10%宽度范围内寻找候选起点（缩小搜索范围）
        left_width = max(int(W * 0.10), 15)
        candidates = _find_all_start_seeds(
            prob_map=suppressed_prob_map,
            used_mask=used_mask,
            left_width=left_width,
            min_prob=min_prob,
            min_distance=min_distance
        )

        if not candidates or candidates[0][2] < min_prob:
            logger.info(f"[fallback_trace] 无足够候选起点，停止提取")
            print(f"[fallback_trace] Round {round_idx+1}: 无足够候选 (candidates={len(candidates) if candidates else 0}, best_prob={candidates[0][2] if candidates else 0:.3f}, min_prob={min_prob})")
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
        print(f"[fallback_trace] Round {round_idx+1}: 起点 col={best_col}, y={best_y}, prob={best_prob:.3f}, 候选数={len(candidates)}")

        # 3.3 DP追踪（优化版：候选点DP + 超时保护）
        path = _dp_trace_single_path(
            prob_map=suppressed_prob_map,
            used_mask=used_mask,
            start_col=best_col,
            start_y=best_y,
            penalties={
                'low_prob': 1.0,
                'jump': 0.05,
                'horizontal': 0.5,
                'gap': 0.5
            },
            max_y_jump=15,  # 从30降到15
            width_expand=W,
            timeout_seconds=2.0  # 单条路径最多2秒
        )

        if path is None or len(path) < W * 0.30:  # 降低到30%
            logger.info(f"[fallback_trace] 路径追踪失败或太短")
            print(f"[fallback_trace] Round {round_idx+1}: 路径失败 (path={'None' if path is None else len(path)}, 需要>{W*0.30:.0f})")

            # 打印DP调试信息（从logger获取）
            import logging
            for handler in logger.handlers:
                if hasattr(handler, 'buffer'):
                    for record in handler.buffer[-5:]:  # 最后5条日志
                        if '[DP]' in record.getMessage():
                            print(f"  DEBUG: {record.getMessage()}")

            round_log['status'] = 'failed_too_short'
            round_log['path_length'] = 0 if path is None else len(path)
            debug_log['rounds'].append(round_log)

            # 抑制失败的起点，避免重复尝试
            y_min = max(0, best_y - 20)
            y_max = min(H, best_y + 20)
            x_min = max(0, best_col - 10)
            x_max = min(W, best_col + 10)
            used_mask[y_min:y_max, x_min:x_max] = True
            print(f"[fallback_trace] Round {round_idx+1}: 抑制失败起点 ({best_col}, {best_y})")

            continue

        # 3.4 质量评分
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

        logger.info(f"[fallback_trace] 路径: 长度={len(path)}, 覆盖={score_dict['width_coverage']:.1%}, "
                   f"概率={score_dict['avg_prob']:.3f}, 质量={'PASS' if is_valid else 'FAIL'}")
        print(f"[fallback_trace] Round {round_idx+1}: 长度={len(path)}, 覆盖={score_dict['width_coverage']:.1%}, "
              f"概率={score_dict['avg_prob']:.3f}, y_range={score_dict['y_range']:.0f}, "
              f"horiz={score_dict['horizontal_ratio']:.1%}, 质量={'PASS' if is_valid else 'FAIL'}")

        if not is_valid:
            round_log['status'] = 'failed_quality_check'
            debug_log['rounds'].append(round_log)
            print(f"[fallback_trace] Round {round_idx+1}: 质量检查失败")
            continue

        # 3.5 间距检查
        min_dist_to_others = float('inf')
        for prev_path in extracted_paths:
            dist = _min_distance_between_paths(path, prev_path)
            min_dist_to_others = min(min_dist_to_others, dist)

        if extracted_paths and min_dist_to_others < min_distance:
            logger.info(f"[fallback_trace] 距离={min_dist_to_others:.1f} < {min_distance}, 太近")
            print(f"[fallback_trace] Round {round_idx+1}: 距离太近 ({min_dist_to_others:.1f} < {min_distance})")
            round_log['status'] = 'failed_too_close'
            round_log['min_dist_to_others'] = float(min_dist_to_others)
            debug_log['rounds'].append(round_log)
            continue

        # 3.6 接受路径
        extracted_paths.append(path)
        round_log['status'] = 'accepted'
        debug_log['rounds'].append(round_log)
        logger.info(f"[fallback_trace] 状态: ACCEPTED ✓")
        print(f"[fallback_trace] Round {round_idx+1}: ACCEPTED ✓")

        # 3.7 带状抑制
        _suppress_around_path(used_mask, path, radius=15)

    logger.info(f"\n[fallback_trace] ✓ Ridge提取完成: {len(extracted_paths)} 条路径")
    print(f"[fallback_trace] ✓ 最终提取: {len(extracted_paths)} 条路径")

    return extracted_paths, {
        'suppressed_prob_map': suppressed_prob_map,
        'suppression_mask': suppression_mask,
        'used_mask': used_mask,
        'debug_log': debug_log
    }
