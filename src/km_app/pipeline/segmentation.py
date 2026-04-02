"""分割处理 - 支持二分类和多类"""
import numpy as np
import cv2
from typing import List, Tuple, Dict
from skimage.morphology import skeletonize


def _analyze_components(mask: np.ndarray) -> Dict:
    """分析连通域特征"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return {
            'num_components': 0,
            'elongated_count': 0,
            'horizontal_coverage': 0.0,
            'max_area_ratio': 0.0
        }

    h, w = mask.shape
    total_pixels = h * w

    elongated_count = 0
    covered_cols = set()
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)

        # 统计横向覆盖
        for cx in range(x, x + cw):
            covered_cols.add(cx)

        # 统计细长组件
        if cw > 0 and ch > 0:
            aspect_ratio = cw / ch
            if aspect_ratio > 2.0 and cw > 50:
                elongated_count += 1

        max_area = max(max_area, area)

    return {
        'num_components': len(contours),
        'elongated_count': elongated_count,
        'horizontal_coverage': len(covered_cols) / w if w > 0 else 0.0,
        'max_area_ratio': max_area / total_pixels if total_pixels > 0 else 0.0
    }


def _score_binary_candidate(mask: np.ndarray, prob_map: np.ndarray) -> float:
    """评分二分类候选

    评分策略：
    - 奖励：横向覆盖长、有1-3个细长组件、skeleton覆盖长
    - 惩罚：fg_ratio过大、超大单块、组件太多、组件太方
    """
    h, w = mask.shape
    total_pixels = h * w

    # 基础统计
    fg_pixels = np.sum(mask > 0)
    fg_ratio = fg_pixels / total_pixels

    # 组件分析
    comp_info = _analyze_components(mask)

    # Skeleton分析
    if fg_pixels > 0:
        skeleton = skeletonize(mask > 0).astype(np.uint8) * 255
        skeleton_pixels = np.sum(skeleton > 0)

        # Skeleton横向覆盖
        skeleton_cols = set()
        for x in range(w):
            if np.sum(skeleton[:, x]) > 0:
                skeleton_cols.add(x)
        skeleton_coverage = len(skeleton_cols) / w
    else:
        skeleton_pixels = 0
        skeleton_coverage = 0.0

    # 评分
    score = 0.0

    # 奖励：横向覆盖长
    score += comp_info['horizontal_coverage'] * 30.0

    # 奖励：skeleton覆盖长
    score += skeleton_coverage * 25.0

    # 奖励：有1-3个细长组件
    if 1 <= comp_info['elongated_count'] <= 3:
        score += 20.0
    elif comp_info['elongated_count'] > 3:
        score += 10.0  # 太多组件，部分奖励

    # 惩罚：fg_ratio过大（大白板）
    if fg_ratio > 0.20:
        score -= (fg_ratio - 0.20) * 50.0

    # 惩罚：超大单块
    if comp_info['max_area_ratio'] > 0.15:
        score -= (comp_info['max_area_ratio'] - 0.15) * 40.0

    # 惩罚：组件太多（噪声）
    if comp_info['num_components'] > 5:
        score -= (comp_info['num_components'] - 5) * 3.0

    # 惩罚：没有细长组件
    if comp_info['elongated_count'] == 0:
        score -= 15.0

    return score


def process_binary_segmentation(pred_result: dict, image: np.ndarray,
                                thresholds=[0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]) -> dict:
    """处理二分类分割输出 - 多指标评分选阈值"""
    prob_map = pred_result['prob_map']  # [H, W]

    print(f"\n[Segmentation] 评估{len(thresholds)}个阈值候选...")

    # 多阈值候选评分
    candidates = []
    for thresh in thresholds:
        binary_mask = (prob_map > thresh).astype(np.uint8) * 255
        fg_pixels = np.sum(binary_mask > 0)
        fg_ratio = fg_pixels / (prob_map.shape[0] * prob_map.shape[1])

        # 组件分析
        comp_info = _analyze_components(binary_mask)

        # 评分
        score = _score_binary_candidate(binary_mask, prob_map)

        candidates.append({
            'threshold': thresh,
            'mask': binary_mask,
            'fg_pixels': fg_pixels,
            'fg_ratio': fg_ratio,
            'score': score,
            'num_components': comp_info['num_components'],
            'elongated_count': comp_info['elongated_count'],
            'horizontal_coverage': comp_info['horizontal_coverage']
        })

        print(f"  t={thresh:.2f}: fg={fg_ratio:.3f}, score={score:.1f}, "
              f"comp={comp_info['num_components']}, elong={comp_info['elongated_count']}, "
              f"h_cov={comp_info['horizontal_coverage']:.2f}")

    # 选择最高分候选
    best_candidate = max(candidates, key=lambda x: x['score'])

    print(f"[Segmentation] ✓ 选择阈值{best_candidate['threshold']:.2f} "
          f"(score={best_candidate['score']:.1f})\n")

    return {
        'mode': 'binary',
        'prob_map': prob_map,
        'binary_mask': best_candidate['mask'],
        'best_threshold': best_candidate['threshold'],
        'best_score': best_candidate['score'],
        'candidates': candidates,
        'fg_ratio': best_candidate['fg_ratio']
    }


def process_multiclass_segmentation(pred_result: dict, image: np.ndarray) -> dict:
    """处理多类分割输出"""
    probs = pred_result['probs']  # [num_classes, H, W]
    class_mask = pred_result['class_mask']  # [H, W]
    num_classes = pred_result['num_classes']

    # 分离每个类别的mask
    class_masks = []
    for i in range(1, num_classes):  # 跳过背景类0
        mask = (class_mask == i).astype(np.uint8) * 255
        class_masks.append(mask)

    return {
        'mode': 'multiclass',
        'class_masks': class_masks,
        'combined_mask': (class_mask > 0).astype(np.uint8) * 255,
        'num_curves': num_classes - 1
    }


def filter_components_by_shape(mask: np.ndarray, min_area=100, min_width=50,
                               prob_map: np.ndarray = None) -> List[np.ndarray]:
    """根据形状特征过滤连通域 - 增强版

    Args:
        mask: 二值mask
        min_area: 最小面积
        min_width: 最小宽度
        prob_map: 概率图（可选，用于额外验证）

    Returns:
        有效组件mask列表
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    valid_masks = []

    print(f"[ComponentFilter] 原始组件数: {len(contours)}")

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 计算bbox
        x, y, cw, ch = cv2.boundingRect(cnt)

        # 基础特征
        aspect_ratio = cw / (ch + 1e-6)

        if cw < min_width:
            continue

        # KM曲线特征：横向延展长、细
        if aspect_ratio < 2.0:
            continue

        # Skeleton长度
        component_mask = np.zeros_like(mask)
        cv2.drawContours(component_mask, [cnt], -1, 255, -1)

        skeleton = skeletonize(component_mask > 0).astype(np.uint8) * 255
        skeleton_len = np.sum(skeleton > 0)

        # 横向覆盖
        covered_cols = set()
        for cx in range(x, x + cw):
            if np.sum(component_mask[:, cx]) > 0:
                covered_cols.add(cx)
        h_coverage = len(covered_cols) / w

        # 概率图验证（如果提供）
        avg_prob = 0.0
        if prob_map is not None:
            # 计算组件区域的平均概率
            component_probs = prob_map[component_mask > 0]
            if len(component_probs) > 0:
                avg_prob = np.mean(component_probs)

        # 综合判断
        is_valid = (
            aspect_ratio >= 2.0 and
            cw >= min_width and
            skeleton_len > 50 and
            h_coverage > 0.1
        )

        # 如果有概率图，额外要求平均概率>0.15
        if prob_map is not None and avg_prob < 0.15:
            is_valid = False

        if is_valid:
            valid_masks.append(component_mask)
            print(f"  组件{i+1}: area={area}, bbox={cw}x{ch}, "
                  f"aspect={aspect_ratio:.1f}, skel_len={skeleton_len}, h_cov={h_coverage:.2f}, "
                  f"avg_prob={avg_prob:.3f} ✓")
        else:
            print(f"  组件{i+1}: 不满足条件，丢弃 (aspect={aspect_ratio:.1f}, skel={skeleton_len}, "
                  f"h_cov={h_coverage:.2f}, prob={avg_prob:.3f})")

    print(f"[ComponentFilter] ✓ 保留组件数: {len(valid_masks)}\n")

    return valid_masks


def extract_skeleton_from_mask(mask: np.ndarray) -> np.ndarray:
    """从mask提取skeleton"""
    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    return skeleton


def refine_masks(masks: List[np.ndarray], min_area: int = 100) -> List[np.ndarray]:
    """精修masks - 轻量形态学"""
    refined = []

    for mask in masks:
        # 轻量opening去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 去除小区域
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(mask)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                cv2.drawContours(filtered, [cnt], -1, 255, -1)

        if np.sum(filtered) > 0:
            refined.append(filtered)

    return refined
