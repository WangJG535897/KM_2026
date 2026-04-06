"""Color-first曲线提取器 - 强制纠偏版：从图片提取带颜色的线"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans


def detect_plot_region(roi_image: np.ndarray) -> Tuple[int, int, int, int]:
    """在ROI内检测plot子区域，排除说明表和at-risk表"""
    h, w = roi_image.shape[:2]
    x1 = int(w * 0.05)
    x2 = int(w * 0.75)
    y1 = int(h * 0.05)
    y2 = int(h * 0.75)
    print(f"[plot区域检测] plot_bbox=({x1},{y1},{x2},{y2}), 尺寸={(x2-x1)}x{(y2-y1)}")
    print(f"[plot区域检测] 排除了下方25%区域（at-risk表）")
    return (x1, y1, x2, y2)


def is_fake_curve(path: np.ndarray, roi_w: int, roi_h: int, plot_y2: int) -> Tuple[bool, str]:
    """伪曲线拒绝器"""
    if len(path) < 30:
        return True, "路径太短"
    y_std = np.std(path[:, 1])
    if y_std < 5:
        return True, f"y标准差过小({y_std:.1f}<5)，基本水平线"
    unique_y = len(np.unique(path[:, 1]))
    if unique_y < 5:
        return True, f"unique_y过少({unique_y}<5)，基本水平线"
    bbox_h = path[:, 1].max() - path[:, 1].min()
    if bbox_h < 10:
        return True, f"bbox高度过小({bbox_h}<10)，太扁"
    x_diffs = np.diff(path[:, 0])
    max_x_gap = np.max(x_diffs) if len(x_diffs) > 0 else 0
    if max_x_gap > 50:
        return True, f"存在超大x跳跃({max_x_gap}>50)"
    avg_y = np.mean(path[:, 1])
    if avg_y > plot_y2 * 0.9:
        return True, f"位于plot区下边界附近(avg_y={avg_y:.0f})"
    return False, "通过"


def score_candidate(path: np.ndarray, roi_w: int, roi_h: int) -> float:
    """候选路径综合评分"""
    score = 0.0
    x_span = path[:, 0].max() - path[:, 0].min()
    coverage = x_span / roi_w
    score += coverage * 40
    y_std = np.std(path[:, 1])
    score += min(y_std / 5, 1.0) * 30
    x_diffs = np.diff(path[:, 0])
    max_x_gap = np.max(x_diffs) if len(x_diffs) > 0 else 0
    continuity = max(0, 1 - max_x_gap / 50)
    score += continuity * 20
    bbox_h = path[:, 1].max() - path[:, 1].min()
    score += min(bbox_h / 50, 1.0) * 10
    return score


def trace_from_mask_robust(mask: np.ndarray) -> List[np.ndarray]:
    """从mask追踪曲线 - 修复gap逻辑"""
    h, w = mask.shape
    paths = []
    first_col = None
    for x in range(w):
        if np.sum(mask[:, x] > 0) > 0:
            first_col = x
            break
    if first_col is None:
        return []
    col = mask[:, first_col]
    start_ys = np.where(col > 0)[0]
    if len(start_ys) == 0:
        return []
    for start_y in start_ys:
        path = [[first_col, start_y]]
        current_y = start_y
        gap_count = 0
        max_gap = 8
        for x in range(first_col + 1, w):
            col = mask[:, x]
            ys = np.where(col > 0)[0]
            if len(ys) == 0:
                gap_count += 1
                if gap_count > max_gap:
                    break
                path.append([x, current_y])
                continue
            distances = np.abs(ys - current_y)
            best_idx = np.argmin(distances)
            next_y = ys[best_idx]
            if abs(next_y - current_y) > 25:
                gap_count += 1
                if gap_count > max_gap:
                    break
                path.append([x, current_y])
                continue
            gap_count = 0
            path.append([x, next_y])
            current_y = next_y
        if len(path) > 30:
            paths.append(np.array(path))
    return paths


def extract_colored_curves(image: np.ndarray, roi: Tuple[int,int,int,int] = None, n_colors: int = 5) -> Dict:
    """从图片提取彩色曲线 - 强制纠偏版"""
    H, W = image.shape[:2]
    print(f"\n{'='*60}")
    print(f"[ColorExtract] 开始")
    print(f"[ColorExtract] 图像: {W}x{H}")
    if roi is None:
        roi = (int(W*0.1), int(H*0.1), int(W*0.9), int(H*0.9))
    x1, y1, x2, y2 = roi
    roi_image = image[y1:y2, x1:x2].copy()
    roi_h, roi_w = roi_image.shape[:2]
    print(f"[ColorExtract] ROI: {roi_w}x{roi_h}")
    plot_x1, plot_y1, plot_x2, plot_y2 = detect_plot_region(roi_image)
    plot_image = roi_image[plot_y1:plot_y2, plot_x1:plot_x2].copy()
    plot_h, plot_w = plot_image.shape[:2]
    print(f"[ColorExtract] plot子区域: {plot_w}x{plot_h}")
    gray = cv2.cvtColor(plot_image, cv2.COLOR_BGR2GRAY)
    foreground = (gray < 240).astype(np.uint8) * 255
    fg_pixels = np.sum(foreground > 0)
    print(f"[ColorExtract] plot内深色像素: {fg_pixels} ({fg_pixels/(plot_h*plot_w)*100:.1f}%)")
    if fg_pixels < 50:
        print(f"[ColorExtract] X 像素太少")
        return _empty_result(roi, roi_image, image, plot_image)
    pixels = plot_image[foreground > 0]
    pixels_lab = cv2.cvtColor(pixels.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3)
    n_clusters = min(n_colors, max(2, len(pixels)//1000))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels_lab)
    print(f"[ColorExtract] 聚类: {n_clusters} 个颜色")
    plot_lab = cv2.cvtColor(plot_image, cv2.COLOR_BGR2LAB)
    all_candidates = []
    for i, center in enumerate(kmeans.cluster_centers_):
        diff = plot_lab.astype(np.float32) - center.reshape(1,1,3)
        distance = np.sqrt(np.sum(diff**2, axis=2))
        color_mask = (distance < 50).astype(np.uint8) * 255
        color_mask = cv2.bitwise_and(color_mask, foreground)
        pixel_count = np.sum(color_mask > 0)
        if pixel_count < 300:
            continue
        print(f"  颜色{i+1}: {pixel_count}像素", end=" ")
        paths = trace_from_mask_robust(color_mask)
        for path in paths:
            x_span = path[:, 0].max() - path[:, 0].min()
            coverage = x_span / plot_w
            y_std = np.std(path[:, 1])
            x_diffs = np.diff(path[:, 0])
            max_x_gap = np.max(x_diffs) if len(x_diffs) > 0 else 0
            is_fake, reason = is_fake_curve(path, plot_w, plot_h, plot_y2)
            if is_fake:
                print(f"-> {len(path)}点, 覆盖{coverage:.1%}, y_std={y_std:.1f}, max_gap={max_x_gap} X ({reason})")
                continue
            score = score_candidate(path, plot_w, plot_h)
            all_candidates.append((path, score, coverage, y_std, max_x_gap))
            print(f"-> {len(path)}点, 覆盖{coverage:.1%}, y_std={y_std:.1f}, score={score:.1f} OK")
    print(f"[ColorExtract] 有效候选: {len(all_candidates)} 条")
    if len(all_candidates) == 0:
        print(f"[ColorExtract] X 无有效候选")
        return _empty_result(roi, roi_image, image, plot_image)
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    final_paths = []
    for path, score, coverage, y_std, max_x_gap in all_candidates[:3]:
        if score > 30:
            final_paths.append(path)
            print(f"[ColorExtract] 保留: {len(path)}点, score={score:.1f}, 覆盖={coverage:.1%}, y_std={y_std:.1f}")
        else:
            print(f"[ColorExtract] 拒绝: score={score:.1f}<30")
    print(f"[ColorExtract] 最终: {len(final_paths)} 条曲线")
    paths_roi = []
    paths_global = []
    for path in final_paths:
        path_roi = path.copy()
        path_roi[:, 0] += plot_x1
        path_roi[:, 1] += plot_y1
        paths_roi.append(path_roi)
        path_global = path_roi.copy()
        path_global[:, 0] += x1
        path_global[:, 1] += y1
        paths_global.append(path_global)
    print(f"{'='*60}\n")
    color_masks = []
    separated_masks = []
    for i, center in enumerate(kmeans.cluster_centers_):
        diff = plot_lab.astype(np.float32) - center.reshape(1,1,3)
        distance = np.sqrt(np.sum(diff**2, axis=2))
        color_mask = (distance < 50).astype(np.uint8) * 255
        color_mask = cv2.bitwise_and(color_mask, foreground)
        if np.sum(color_mask > 0) >= 300:
            color_masks.append(color_mask)
    for mask in color_masks:
        num_labels, labels = cv2.connectedComponents(mask)
        for label_id in range(1, num_labels):
            component = (labels == label_id).astype(np.uint8) * 255
            if np.sum(component > 0) >= 50:
                separated_masks.append(component)
    return {
        'roi': roi, 'roi_image': roi_image, 'original_image': image,
        'plot_image': plot_image, 'plot_bbox': (plot_x1, plot_y1, plot_x2, plot_y2),
        'pixel_paths_roi': paths_roi, 'pixel_paths_global': paths_global, 'pixel_paths': paths_roi,
        'color_masks': color_masks, 'separated_masks': separated_masks, 'colors': [],
        'num_curves': len(final_paths),
        'stats': {'n_colors': n_clusters, 'n_components': len(separated_masks), 'n_curves': len(final_paths), 'foreground_pixels': fg_pixels}
    }


def _empty_result(roi, roi_image, image, plot_image=None):
    return {
        'roi': roi, 'roi_image': roi_image, 'original_image': image,
        'plot_image': plot_image if plot_image is not None else roi_image,
        'plot_bbox': (0, 0, roi_image.shape[1], roi_image.shape[0]),
        'pixel_paths_roi': [], 'pixel_paths_global': [], 'pixel_paths': [],
        'color_masks': [], 'separated_masks': [], 'colors': [], 'num_curves': 0,
        'stats': {'n_colors': 0, 'n_components': 0, 'n_curves': 0, 'foreground_pixels': 0}
    }
