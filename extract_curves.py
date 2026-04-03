"""简化CLI - Color-first曲线提取（不依赖模型）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from km_app.pipeline.color_extract import extract_colored_curves
from km_app.io import load_image, save_image


def main():
    import argparse
    parser = argparse.ArgumentParser(description="彩色曲线提取工具")
    parser.add_argument("--image", "-i", required=True, help="输入图像路径")
    parser.add_argument("--outdir", "-o", default="outputs/color_first", help="输出目录")
    parser.add_argument("--n-colors", type=int, default=5, help="颜色聚类数")
    args = parser.parse_args()

    # 1. 加载图像
    print(f"加载图像: {args.image}")
    image = load_image(args.image)

    # 2. 提取曲线
    result = extract_colored_curves(image, n_colors=args.n_colors)

    # 3. 保存结果
    output_path = Path(args.outdir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n保存结果到: {output_path}")

    # 保存color masks
    for i, mask in enumerate(result['color_masks']):
        save_image(mask, str(output_path / f"mask_color_{i+1}.png"))

    # 保存component masks
    for i, mask in enumerate(result['separated_masks']):
        save_image(mask, str(output_path / f"mask_component_{i+1}.png"))

    # 保存overlay
    if len(result['pixel_paths_roi']) > 0:
        overlay = result['roi_image'].copy()
        colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255)]
        for i, path in enumerate(result['pixel_paths_roi']):
            color = colors[i % len(colors)]
            for j in range(len(path)-1):
                cv2.line(overlay, tuple(path[j]), tuple(path[j+1]), color, 2)
        save_image(overlay, str(output_path / "result.png"))

    # 保存全图结果
    if len(result['pixel_paths_global']) > 0:
        result_global = result['original_image'].copy()
        for i, path in enumerate(result['pixel_paths_global']):
            color = colors[i % len(colors)]
            for j in range(len(path)-1):
                cv2.line(result_global, tuple(path[j]), tuple(path[j+1]), color, 2)
        save_image(result_global, str(output_path / "result_global.png"))

    # 保存像素坐标CSV
    for i, path in enumerate(result['pixel_paths_global']):
        csv_path = output_path / f"curve_pixels_{i+1}.csv"
        np.savetxt(csv_path, path, delimiter=',', header='x,y', comments='', fmt='%d')
        print(f"  保存: curve_pixels_{i+1}.csv ({len(path)} 个点)")

    # 保存统计信息
    with open(output_path / "process.log", 'w', encoding='utf-8') as f:
        f.write(f"颜色数: {result['stats']['n_colors']}\n")
        f.write(f"连通域数: {result['stats']['n_components']}\n")
        f.write(f"最终曲线数: {result['num_curves']}\n")
        f.write(f"前景像素: {result['stats'].get('foreground_pixels', 'N/A')}\n")

    print(f"\n✓ 成功提取 {result['num_curves']} 条曲线")
    print(f"✓ 结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
