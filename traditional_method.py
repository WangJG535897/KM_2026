"""传统图像处理方法提取曲线 - 不依赖深度学习模型"""
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from km_app.io import load_image, save_image, export_all_curves
from km_app.pipeline import CoordinateMapper, apply_km_constraints_batch
from km_app.utils import draw_curves_on_image


def extract_curves_traditional(image_path, x_max=42.0, output_dir="outputs/traditional"):
    """使用传统方法提取曲线"""
    print("=" * 60)
    print("传统图像处理方法")
    print("=" * 60)

    # 1. 加载图像
    print("\n[1] 加载图像...")
    image = load_image(image_path)
    print(f"  图像尺寸: {image.shape}")

    # 2. 转换为灰度图
    print("\n[2] 预处理...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 反转（如果曲线是黑色）
    if gray.mean() > 127:
        gray = 255 - gray

    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 3. 检测曲线
    print("\n[3] 检测曲线...")

    # 边缘检测
    edges = cv2.Canny(cleaned, 50, 150)

    # 霍夫线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                           minLineLength=30, maxLineGap=10)

    print(f"  检测到 {len(lines) if lines is not None else 0} 条线段")

    # 4. 保存调试结果
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    save_image(image, f"{output_dir}/01_original.png")
    save_image(gray, f"{output_dir}/02_gray.png")
    save_image(binary, f"{output_dir}/03_binary.png")
    save_image(cleaned, f"{output_dir}/04_cleaned.png")
    save_image(edges, f"{output_dir}/05_edges.png")

    # 绘制检测到的线段
    if lines is not None:
        line_img = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        save_image(line_img, f"{output_dir}/06_detected_lines.png")

    print(f"\n结果已保存到: {output_dir}")
    print("\n⚠ 注意: 这是传统方法的临时方案")
    print("深度学习模型的问题是权重不匹配，需要原始训练代码才能完全修复")


if __name__ == "__main__":
    image_path = r"C:/Users/32665/Desktop/生存曲线数据集/image_108.png"
    extract_curves_traditional(image_path)
