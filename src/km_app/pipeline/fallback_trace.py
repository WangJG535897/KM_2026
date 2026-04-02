"""Fallback路径追踪 - 当常规流程失败时使用"""
import numpy as np
from typing import List, Tuple
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def trace_from_prob_map_ridge(prob_map: np.ndarray, num_curves: int = 3,
                               min_prob: float = 0.2, min_distance: int = 15) -> List[np.ndarray]:
    """直接从概率图提取ridge路径 - 增强版fallback

    Args:
        prob_map: 概率图 [H, W]
        num_curves: 期望提取的曲线数量
        min_prob: 最小概率阈值
        min_distance: 曲线间最小距离

    Returns:
        路径列表
    """
    h, w = prob_map.shape
    print(f"\n[Fallback] 从概率图提取ridge路径...")
    print(f"[Fallback] 概率图尺寸: {w}x{h}")

    # 1. 找起点搜索区域（前15%宽度）
    start_search_width = max(int(w * 0.15), 10)
    print(f"[Fallback] 起点搜索区域: 前{start_search_width}列")

    # 2. 对每列提取候选峰值（允许plateau）
    column_candidates = []
    for x in range(w):
        col = prob_map[:, x]

        # 轻量平滑以处理plateau
        col_smooth = gaussian_filter1d(col, sigma=1.5)

        # 使用find_peaks检测峰值
        peaks, properties = find_peaks(col_smooth,
                                      height=min_prob,
                                      prominence=0.05,
                                      distance=min_distance)

        # 按概率排序
        if len(peaks) > 0:
            peak_probs = col_smooth[peaks]
            sorted_indices = np.argsort(peak_probs)[::-1]
            top_peaks = [(peaks[i], col_smooth[peaks[i]]) for i in sorted_indices[:num_curves]]
            column_candidates.append(top_peaks)
        else:
            column_candidates.append([])

    # 统计候选数
    avg_candidates = np.mean([len(c) for c in column_candidates if len(c) > 0])
    print(f"[Fallback] 平均每列候选峰数: {avg_candidates:.2f}")

    # 3. 找起点列（前15%内第一个有足够候选的列）
    start_x = 0
    for x in range(start_search_width):
        if len(column_candidates[x]) >= 1:
            start_x = x
            break

    if start_x == 0 and len(column_candidates[0]) == 0:
        # 前15%都没有，扩展到前25%
        start_search_width = max(int(w * 0.25), 20)
        print(f"[Fallback] 扩展起点搜索到前{start_search_width}列")
        for x in range(start_search_width):
            if len(column_candidates[x]) >= 1:
                start_x = x
                break

    print(f"[Fallback] 起点列: {start_x}")

    # 4. 动态规划追踪多条路径
    paths = []
    used_mask = np.zeros_like(prob_map, dtype=bool)  # 标记已使用区域

    for curve_idx in range(num_curves):
        if len(column_candidates[start_x]) <= curve_idx:
            break

        # 初始化DP
        # dp[x][y] = (score, prev_y)
        dp = {}

        # 起点
        start_y = column_candidates[start_x][curve_idx][0]
        dp[(start_x, start_y)] = (column_candidates[start_x][curve_idx][1], None)

        # 动态规划追踪
        for x in range(start_x + 1, w):
            if len(column_candidates[x]) == 0:
                # 当前列无候选，从上一列所有状态延续
                if x - 1 in [k[0] for k in dp.keys()]:
                    prev_states = [(k, v) for k, v in dp.items() if k[0] == x - 1]
                    for (prev_x, prev_y), (prev_score, _) in prev_states:
                        # gap penalty
                        new_score = prev_score - 0.02
                        if (x, prev_y) not in dp or new_score > dp[(x, prev_y)][0]:
                            dp[(x, prev_y)] = (new_score, prev_y)
                continue

            # 当前列有候选
            for cand_y, cand_prob in column_candidates[x]:
                # 跳过已使用区域
                if used_mask[cand_y, x]:
                    continue

                best_score = -float('inf')
                best_prev_y = None

                # 从上一列所有状态转移
                if x - 1 in [k[0] for k in dp.keys()]:
                    prev_states = [(k, v) for k, v in dp.items() if k[0] == x - 1]

                    for (prev_x, prev_y), (prev_score, _) in prev_states:
                        # 计算转移代价
                        dy = cand_y - prev_y

                        # 概率奖励
                        prob_reward = cand_prob * 2.0

                        # 连续性惩罚
                        continuity_penalty = abs(dy) * 0.05

                        # 向上跳惩罚（图像坐标y减小=向上=生存率上升，不应明显发生）
                        upward_penalty = max(0, -dy) * 0.15

                        # 总分
                        score = prev_score + prob_reward - continuity_penalty - upward_penalty

                        if score > best_score:
                            best_score = score
                            best_prev_y = prev_y

                if best_prev_y is not None:
                    dp[(x, cand_y)] = (best_score, best_prev_y)

        # 5. 回溯最优路径
        if not dp:
            break

        # 找终点（最后一列的最高分状态）
        last_x = max([k[0] for k in dp.keys()])
        end_states = [(k, v) for k, v in dp.items() if k[0] == last_x]

        if not end_states:
            break

        best_end = max(end_states, key=lambda x: x[1][0])
        end_x, end_y = best_end[0]

        # 回溯路径
        path_points = []
        current_x, current_y = end_x, end_y

        while current_x >= start_x:
            path_points.append([current_x, current_y])

            if (current_x, current_y) in dp:
                _, prev_y = dp[(current_x, current_y)]
                if prev_y is None:
                    break
                current_x -= 1
                current_y = prev_y
            else:
                break

        path_points.reverse()
        path = np.array(path_points)

        # 检查路径长度
        coverage = len(path) / w
        print(f"[Fallback] 路径{curve_idx+1}: {len(path)}点, 覆盖{coverage:.1%}")

        if coverage >= 0.35:  # 至少覆盖35%宽度
            paths.append(path)

            # 标记已使用区域（带状抑制）
            for px, py in path:
                y_min = max(0, py - min_distance // 2)
                y_max = min(h, py + min_distance // 2)
                used_mask[y_min:y_max, px] = True
        else:
            print(f"[Fallback] 路径{curve_idx+1}太短，丢弃")

    # 6. 过滤太接近的路径
    if len(paths) > 1:
        filtered_paths = [paths[0]]
        for i in range(1, len(paths)):
            too_close = False
            for existing_path in filtered_paths:
                # 计算重叠区域平均距离
                x_min = max(paths[i][:, 0].min(), existing_path[:, 0].min())
                x_max = min(paths[i][:, 0].max(), existing_path[:, 0].max())

                if x_max > x_min:
                    distances = []
                    for x in range(x_min, x_max + 1):
                        y1_candidates = paths[i][paths[i][:, 0] == x, 1]
                        y2_candidates = existing_path[existing_path[:, 0] == x, 1]
                        if len(y1_candidates) > 0 and len(y2_candidates) > 0:
                            distances.append(abs(y1_candidates[0] - y2_candidates[0]))

                    if len(distances) > 0 and np.mean(distances) < min_distance:
                        too_close = True
                        break

            if not too_close:
                filtered_paths.append(paths[i])

        paths = filtered_paths

    print(f"[Fallback] ✓ 最终提取: {len(paths)} 条ridge路径\n")

    return paths
