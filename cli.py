"""CLI命令行入口"""
import sys
import argparse
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from km_app.model import ModelInference
from km_app.pipeline import KMPipeline
from km_app.io import load_image, save_image, export_all_curves, setup_logger
from km_app.utils import draw_curves_on_image, create_mask_visualization
from km_app.config import DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_X_MAX


def main():
    """CLI主函数"""
    parser = argparse.ArgumentParser(description="KM生存曲线提取工具 - 命令行版本")

    parser.add_argument("--image", "-i", required=True, help="输入图像路径")
    parser.add_argument("--checkpoint", "-c", default=str(DEFAULT_MODEL_PATH),
                       help="模型权重路径")
    parser.add_argument("--outdir", "-o", default=str(DEFAULT_OUTPUT_DIR),
                       help="输出目录")
    parser.add_argument("--x-max", type=float, default=DEFAULT_X_MAX,
                       help="X轴最大时间")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    args = parser.parse_args()

    # 设置日志
    log_file = Path(args.outdir) / "process.log" if args.debug else None
    logger = setup_logger("KM_CLI", log_file)

    try:
        # 1. 加载图像
        logger.info(f"加载图像: {args.image}")
        image = load_image(args.image)
        logger.info(f"图像尺寸: {image.shape}")

        # 2. 加载模型
        logger.info(f"加载模型: {args.checkpoint}")
        inference = ModelInference(args.checkpoint)
        inference.load_model()

        # 3. 推理
        logger.info("开始推理...")
        pred_result = inference.predict(image)
        logger.info(f"推理完成，输出类别数: {pred_result['num_classes']}")

        # 4. 处理
        logger.info("处理曲线...")
        pipeline = KMPipeline(x_max=args.x_max)
        result = pipeline.process(image, pred_result)
        logger.info(f"提取到 {result['num_curves']} 条曲线")

        # 5. 保存结果
        output_path = Path(args.outdir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"保存结果到: {output_path}")

        # 保存原图
        save_image(image, str(output_path / "original.png"))

        # 保存ROI
        save_image(result['roi_image'], str(output_path / "roi.png"))

        # 保存结果图
        result_image = draw_curves_on_image(result['roi_image'], result['pixel_paths'])
        save_image(result_image, str(output_path / "result.png"))

        # 保存masks
        if args.debug:
            for i, mask in enumerate(result['refined_masks']):
                save_image(mask, str(output_path / f"mask_{i+1}.png"))

            # 保存mask可视化
            mask_vis = create_mask_visualization(result['refined_masks'],
                                                result['roi_image'].shape)
            save_image(mask_vis, str(output_path / "masks_visualization.png"))

        # 导出CSV
        export_all_curves(result['chart_coords'], str(output_path), "curve")

        logger.info("=" * 60)
        logger.info("处理完成!")
        logger.info(f"提取曲线数: {result['num_curves']}")
        logger.info(f"输出目录: {output_path}")
        logger.info("=" * 60)

        print(f"\n✓ 成功提取 {result['num_curves']} 条曲线")
        print(f"✓ 结果已保存到: {output_path}")

    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        print(f"\n✗ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
