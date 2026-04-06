"""CLI命令行入口 - Color-first默认"""
import sys
import argparse
from pathlib import Path
import numpy as np

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from km_app.io import load_image, save_image, setup_logger
from km_app.config import DEFAULT_OUTPUT_DIR


def main():
    """CLI主函数 - Color-first默认"""
    parser = argparse.ArgumentParser(description="曲线提取工具 - 从图片提取带颜色的线")

    parser.add_argument("--image", "-i", required=True, help="输入图像路径")
    parser.add_argument("--outdir", "-o", default=str(DEFAULT_OUTPUT_DIR),
                       help="输出目录")
    parser.add_argument("--n-colors", type=int, default=5,
                       help="颜色聚类数（默认5）")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    args = parser.parse_args()

    # 设置日志
    output_path = Path(args.outdir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = output_path / "process.log"
    logger = setup_logger("CLI", log_file)

    try:
        # 1. 加载图像
        logger.info(f"加载图像: {args.image}")
        print(f"\n加载图像: {args.image}")
        image = load_image(args.image)
        h, w = image.shape[:2]
        logger.info(f"图像尺寸: {w}x{h}")
        print(f"图像尺寸: {w}x{h}")

        # 2. Color-first提取
        logger.info("开始提取彩色曲线...")
        print("开始提取彩色曲线...")

        from km_app.pipeline.color_extract import extract_colored_curves
        result = extract_colored_curves(image, roi=None, n_colors=args.n_colors)

        logger.info(f"提取完成: {result['num_curves']} 条曲线")
        print(f"\n提取完成: {result['num_curves']} 条曲线")

        # 3. 保存结果
        logger.info(f"保存结果到: {output_path}")
        print(f"保存结果到: {output_path}")

        # 保存ROI裁剪
        save_image(result['roi_image'], str(output_path / "roi_crop.png"))

        # 保存color masks
        for i, mask in enumerate(result['color_masks']):
            save_image(mask, str(output_path / f"mask_color_{i+1}.png"))

        # 保存separated masks
        for i, mask in enumerate(result['separated_masks']):
            save_image(mask, str(output_path / f"mask_component_{i+1}.png"))

        # 保存ROI局部结果
        if len(result['pixel_paths_roi']) > 0:
            from km_app.utils import draw_curves_on_image
            result_roi = draw_curves_on_image(result['roi_image'], result['pixel_paths_roi'])
            save_image(result_roi, str(output_path / "result_roi.png"))

        # 保存全图结果
        if len(result['pixel_paths_global']) > 0:
            from km_app.utils import draw_curves_on_image
            result_global = draw_curves_on_image(result['original_image'], result['pixel_paths_global'])
            save_image(result_global, str(output_path / "result_global.png"))

            # 保存像素坐标CSV
            for i, path in enumerate(result['pixel_paths_global']):
                csv_path = output_path / f"curve_pixels_{i+1}.csv"
                np.savetxt(csv_path, path, delimiter=',', header='x,y', comments='', fmt='%d')
                logger.info(f"保存曲线{i+1}: {len(path)}个点")

        # 保存debug overlay
        if len(result['pixel_paths_roi']) > 0:
            import cv2
            debug_overlay = result['roi_image'].copy()
            colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255)]
            for i, path in enumerate(result['pixel_paths_roi']):
                color = colors[i % len(colors)]
                for j in range(len(path)-1):
                    cv2.line(debug_overlay, tuple(path[j]), tuple(path[j+1]), color, 2)
            save_image(debug_overlay, str(output_path / "debug_overlay.png"))

        # 保存处理信息到log
        info_lines = [
            f"图像: {args.image}",
            f"尺寸: {w}x{h}",
            f"ROI: {result['roi']}",
            f"颜色数: {result['stats']['n_colors']}",
            f"连通域数: {result['stats']['n_components']}",
            f"最终曲线数: {result['num_curves']}",
            f"前景像素: {result['stats'].get('foreground_pixels', 'N/A')}"
        ]
        for line in info_lines:
            logger.info(line)

        with open(output_path / "process.log", 'a', encoding='utf-8') as f:
            f.write('\n' + '='*60 + '\n')
            f.write('\n'.join(info_lines))
            f.write('\n' + '='*60 + '\n')

        logger.info("=" * 60)
        logger.info("处理完成!")
        logger.info(f"输出目录: {output_path}")
        logger.info("=" * 60)

        print(f"\n[成功] 提取 {result['num_curves']} 条曲线")
        print(f"[成功] 结果已保存到: {output_path}")

    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
