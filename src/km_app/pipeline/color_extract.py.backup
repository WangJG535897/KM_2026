"""Color-first曲线提取器 - 稳定版：从图片提取带颜色的线"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans


def trace_from_mask_robust(mask: np.ndarray) -> List[np.ndarray]:
    """
    从mask追踪曲线 - 稳定版

    改进：
    1. 允许短gap桥接
    2. 多起点尝试
    3. 保留top-k路径
    """
    h, w = mask.shape
    paths = []

    # 找第一列
    first_col = None
    for x in range(w):
        if np.sum(mask[:, x] > 0) > 0:
            first_col = x
            break

    if first_col is None:
        return []

    # 找起点（第一列的所有非零点）
    col = mask[:, first_col]
    start_ys = np.where(col > 0)[0]

    if len(start_ys) == 0:
        return []

    # 对每个起点追踪
    for start_y in start_ys:
        path = [[first_col, start_y]]
        current_y = start_y
        gap_count = 0
        max_gap = 10  # 允许10列gap

        for x in range(first_col + 1, w):
            col = mask[:, x]
            ys = np.where(col > 0)[0]

            if len(ys) == 0:
                # 当前列没有点，尝试桥接
                gap_count += 1
                if gap_count > max_gap:
                    break
                # 保持当前y继续
                path.append([x, current_y])
                continue

            # 重置gap
            gap_count = 0

            # 找最近的y
            distances = np.abs(ys - current_y)
            best_idx = np.argmin(distances)
            next_y = ys[best_idx]

            # 限制跳变
            if abs(next_y - current_y) > 30:
                continue

            path.append([x, next_y])
            current_y = next_y

        if len(path) > 30:  # 从50降到30
            paths.append(np.array(path))

    return paths


def extract_colored_curves(image: np.ndarray,
                          roi: Tuple[int,int,int,int] = None,
                          n_colors: int = 5) -> Dict:
    """
    从图片提取彩色曲线 - 稳定版

    改进：
    1. 放宽阈值
    2. top-k保留策略
    3. 相对覆盖率判定
    """
    H, W = image.shape[:2]

    print(f"\n{'='*60}")
    print(f"[ColorExtract] 开始")
    print(f"[ColorExtract] 图像: {W}x{H}")

    # ROI
    if roi is None:
        roi = (int(W*0.1), int(H*0.1), int(W*0.9), int(H*0.9))

    x1, y1, x2, y2 = roi
    roi_image = image[y1:y2, x1:x2].copy()
    roi_h, roi_w = roi_image.shape[:2]
    print(f"[ColorExtract] ROI: {roi_w}x{roi_h}")

    # 提取深色像素
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    foreground = (gray < 240).astype(np.uint8) * 255

    fg_pixels = np.sum(foreground > 0)
    print(f"[ColorExtract] 深色像素: {fg_pixels} ({fg_pixels/(roi_h*roi_w)*100:.1f}%)")

    if fg_pixels < 50:  # 从100降到50
        print(f"[ColorExtract] ✗ 像素太少")
        return _empty_result(roi, roi_image, image)

    # 颜色聚类
    pixels = roi_image[foreground > 0]
    pixels_lab = cv2.cvtColor(pixels.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3)
    n_clusters = min(n_colors, max(2, len(pixels)//1000))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels_lab)

    print(f"[ColorExtract] 聚类: {n_clusters} 个颜色")

    # 为每个颜色生成mask并追踪
    roi_lab = cv2.cvtColor(roi_image, cv2.COLOR_BGR2LAB)
    all_candidates = []  # (path, coverage, pixel_count)

    for i, center in enumerate(kmeans.cluster_centers_):
        diff = roi_lab.astype(np.float32) - center.reshape(1,1,3)
        distance = np.sqrt(np.sum(diff**2, axis=2))
        color_mask = (distance < 50).astype(np.uint8) * 255
        color_mask = cv2.bitwise_and(color_mask, foreground)

        pixel_count = np.sum(color_mask > 0)
        if pixel_count < 300:  # 从500降到300
            continue

        print(f"  颜色{i+1}: {pixel_count}像素", end=" ")

        # 追踪
        paths = trace_from_mask_robust(color_mask)

        for path in paths:
            x_span = path[:, 0].max() - path[:, 0].min()
            coverage = x_span / roi_w

            all_candidates.append((path, coverage, pixel_count))
            print(f"-> {len(path)}点, 覆盖{coverage:.1%}")

    print(f"[ColorExtract] 候选: {len(all_candidates)} 条")

    # Top-k保留策略
    if len(all_candidates) == 0:
        print(f"[ColorExtract] ✗ 无候选")
        return _empty_result(roi, roi_image, image)

    # 按覆盖率排序
    all_candidates.sort(key=lambda x: x[1], reverse=True)

    # 保留策略：
    # 1. 覆盖率>=10%的全部保留
    # 2. 如果没有，保留top-2
    final_paths = []
    for path, coverage, _ in all_candidates:
        if coverage >= 0.10:
            final_paths.append(path)

    if len(final_paths) == 0 and len(all_candidates) > 0:
        print(f"[ColorExtract] 保底策略: 保留top-2")
        for path, coverage, _ in all_candidates[:2]:
            final_paths.append(path)
            print(f"  保底路径: {len(path)}点, 覆盖{coverage:.1%}")

    print(f"[ColorExtract] 最终: {len(final_paths)} 条曲线")

    # 转全图坐标
    paths_global = []
    for path in final_paths:
        path_g = path.copy()
        path_g[:, 0] += x1
        path_g[:, 1] += y1
        paths_global.append(path_g)

    print(f"{'='*60}\n")

    # 收集color_masks和separated_masks用于输出
    color_masks = []
    separated_masks = []
    for i, center in enumerate(kmeans.cluster_centers_):
        diff = roi_lab.astype(np.float32) - center.reshape(1,1,3)
        distance = np.sqrt(np.sum(diff**2, axis=2))
        color_mask = (distance < 50).astype(np.uint8) * 255
        color_mask = cv2.bitwise_and(color_mask, foreground)
        if np.sum(color_mask > 0) >= 300:
            color_masks.append(color_mask)

    # separated_masks用连通域分离
    for mask in color_masks:
        num_labels, labels = cv2.connectedComponents(mask)
        for label_id in range(1, num_labels):
            component = (labels == label_id).astype(np.uint8) * 255
            if np.sum(component > 0) >= 50:
                separated_masks.append(component)

    return {
        'roi': roi,
        'roi_image': roi_image,
        'original_image': image,
        'pixel_paths_roi': final_paths,
        'pixel_paths_global': paths_global,
        'pixel_paths': final_paths,
        'color_masks': color_masks,
        'separated_masks': separated_masks,
        'colors': [],
        'num_curves': len(final_paths),
        'stats': {
            'n_colors': n_clusters,
            'n_components': len(separated_masks),
            'n_curves': len(final_paths),
            'foreground_pixels': fg_pixels
        }
    }


def _empty_result(roi, roi_image, image):
    return {
        'roi': roi,
        'roi_image': roi_image,
        'original_image': image,
        'pixel_paths_roi': [],
        'pixel_paths_global': [],
        'pixel_paths': [],
        'color_masks': [],
        'separated_masks': [],
        'colors': [],
        'num_curves': 0,
        'stats': {
            'n_colors': 0,
            'n_components': 0,
            'n_curves': 0,
            'foreground_pixels': 0
        }
    }
