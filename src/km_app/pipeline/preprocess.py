"""图像预处理"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict


def auto_detect_roi(image: np.ndarray, margin: int = 10) -> Tuple[int, int, int, int]:
    """自动检测plot区域ROI - 改进版

    Returns:
        (x1, y1, x2, y2) ROI坐标
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. 边缘检测找初始候选
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"[ROI] 未找到轮廓，使用全图")
        return (0, 0, w, h)

    # 找最大矩形区域作为初始候选
    max_area = 0
    best_rect = None

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area > max_area and cw > 100 and ch > 100:
            max_area = area
            best_rect = (x, y, cw, ch)

    if best_rect is None:
        print(f"[ROI] 未找到有效矩形，使用全图")
        return (0, 0, w, h)

    x, y, cw, ch = best_rect
    print(f"[ROI] 初始候选框: x={x}, y={y}, w={cw}, h={ch}")

    # 2. 收紧策略：检测轴线和主内容区域
    # 在候选框内寻找更精确的plot边界
    roi_gray = gray[y:y+ch, x:x+cw]

    # 检测水平线（x轴）
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(roi_gray, cv2.MORPH_OPEN, horizontal_kernel)

    # 检测垂直线（y轴）
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(roi_gray, cv2.MORPH_OPEN, vertical_kernel)

    # 找x轴位置（底部横线）
    h_sum = np.sum(horizontal_lines, axis=1)
    if np.max(h_sum) > 0:
        # 找最下方的强横线
        strong_h_lines = np.where(h_sum > np.max(h_sum) * 0.3)[0]
        if len(strong_h_lines) > 0:
            bottom_line = strong_h_lines[-1]
            # 收紧底部，去掉at-risk table
            ch = min(ch, bottom_line + 20)

    # 找y轴位置（左侧竖线）
    v_sum = np.sum(vertical_lines, axis=0)
    if np.max(v_sum) > 0:
        # 找最左侧的强竖线
        strong_v_lines = np.where(v_sum > np.max(v_sum) * 0.3)[0]
        if len(strong_v_lines) > 0:
            left_line = strong_v_lines[0]
            # 收紧左侧
            x = x + max(0, left_line - 10)
            cw = cw - max(0, left_line - 10)

    # 3. 去除顶部和右侧过多空白
    # 检测内容密度
    roi_gray_tight = gray[y:y+ch, x:x+cw]

    # 从上往下找第一个有内容的行
    row_density = np.sum(roi_gray_tight < 250, axis=1)
    content_rows = np.where(row_density > cw * 0.1)[0]
    if len(content_rows) > 0:
        top_content = content_rows[0]
        y = y + max(0, top_content - 20)
        ch = ch - max(0, top_content - 20)

    # 从右往左找最后有内容的列
    col_density = np.sum(roi_gray_tight < 250, axis=0)
    content_cols = np.where(col_density > ch * 0.1)[0]
    if len(content_cols) > 0:
        right_content = content_cols[-1]
        cw = min(cw, right_content + 30)

    # 4. 添加小边距并转换为(x1,y1,x2,y2)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + cw + margin)
    y2 = min(h, y + ch + margin)

    roi_w = x2 - x1
    roi_h = y2 - y1

    print(f"[ROI] 收紧后ROI: ({x1}, {y1}, {x2}, {y2}), 尺寸: {roi_w}x{roi_h}")

    return (x1, y1, x2, y2)


def crop_roi(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """裁剪ROI区域

    Args:
        image: 输入图像
        roi: (x1, y1, x2, y2) ROI坐标

    Returns:
        裁剪后的图像
    """
    x1, y1, x2, y2 = roi

    # 边界检查
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI: ({x1},{y1},{x2},{y2})")

    return image[y1:y2, x1:x2].copy()


def normalize_image(image: np.ndarray) -> np.ndarray:
    """归一化图像"""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    return image
