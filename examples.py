"""使用示例 - 展示完整的处理流程"""
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))

from km_app.model import ModelInference
from km_app.pipeline import KMPipeline
from km_app.io import load_image, save_image, export_all_curves
from km_app.utils import draw_curves_on_image, create_mask_visualization


def example_basic():
    """基础示例：处理单张图像"""
    print("=" * 60)
    print("示例1: 基础处理流程")
    print("=" * 60)

    # 配置
    image_path = "path/to/your/km_curve.png"  # 替换为实际路径
    checkpoint_path = "models/best_model.pth"
    output_dir = "outputs/example1"
    x_max = 48.0

    # 1. 加载图像
    print("1. 加载图像...")
    image = load_image(image_path)
    print(f"   图像尺寸: {image.shape}")

    # 2. 加载模型并推理
    print("2. 模型推理...")
    inference = ModelInference(checkpoint_path)
    inference.load_model()
    pred_result = inference.predict(image)
    print(f"   输出类别数: {pred_result['num_classes']}")

    # 3. 处理曲线
    print("3. 处理曲线...")
    pipeline = KMPipeline(x_max=x_max)
    result = pipeline.process(image, pred_result)
    print(f"   提取到 {result['num_curves']} 条曲线")

    # 4. 保存结果
    print("4. 保存结果...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 保存可视化
    result_image = draw_curves_on_image(result['roi_image'], result['pixel_paths'])
    save_image(result_image, f"{output_dir}/result.png")

    # 导出CSV
    export_all_curves(result['chart_coords'], output_dir, "curve")

    print(f"✓ 完成! 结果保存到: {output_dir}")
    print()


def example_with_custom_roi():
    """示例2: 使用自定义ROI"""
    print("=" * 60)
    print("示例2: 自定义ROI")
    print("=" * 60)

    image_path = "path/to/your/km_curve.png"
    checkpoint_path = "models/best_model.pth"
    output_dir = "outputs/example2"

    # 加载图像
    image = load_image(image_path)

    # 自定义ROI (x, y, w, h)
    custom_roi = (50, 50, 700, 500)

    # 推理
    inference = ModelInference(checkpoint_path)
    inference.load_model()
    pred_result = inference.predict(image)

    # 使用自定义ROI处理
    pipeline = KMPipeline(x_max=60.0)
    result = pipeline.process(image, pred_result, roi=custom_roi)

    print(f"✓ 使用自定义ROI提取到 {result['num_curves']} 条曲线")
    print()


def example_batch_processing():
    """示例3: 批量处理多张图像"""
    print("=" * 60)
    print("示例3: 批量处理")
    print("=" * 60)

    image_dir = Path("path/to/image/folder")
    checkpoint_path = "models/best_model.pth"
    output_base = "outputs/batch"

    # 加载模型（只加载一次）
    inference = ModelInference(checkpoint_path)
    inference.load_model()

    # 处理所有图像
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

    for i, image_path in enumerate(image_files):
        print(f"\n处理 {i+1}/{len(image_files)}: {image_path.name}")

        try:
            # 加载图像
            image = load_image(str(image_path))

            # 推理
            pred_result = inference.predict(image)

            # 处理
            pipeline = KMPipeline(x_max=48.0)
            result = pipeline.process(image, pred_result)

            # 保存
            output_dir = Path(output_base) / image_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)

            result_image = draw_curves_on_image(result['roi_image'], result['pixel_paths'])
            save_image(result_image, str(output_dir / "result.png"))
            export_all_curves(result['chart_coords'], str(output_dir), "curve")

            print(f"  ✓ 提取到 {result['num_curves']} 条曲线")

        except Exception as e:
            print(f"  ✗ 失败: {e}")

    print(f"\n✓ 批量处理完成!")


def example_debug_visualization():
    """示例4: 调试可视化"""
    print("=" * 60)
    print("示例4: 调试可视化")
    print("=" * 60)

    image_path = "path/to/your/km_curve.png"
    checkpoint_path = "models/best_model.pth"
    output_dir = "outputs/debug"

    # 加载和处理
    image = load_image(image_path)
    inference = ModelInference(checkpoint_path)
    inference.load_model()
    pred_result = inference.predict(image)

    pipeline = KMPipeline(x_max=48.0)
    result = pipeline.process(image, pred_result)

    # 保存所有中间结果
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. 原图
    save_image(image, f"{output_dir}/01_original.png")

    # 2. ROI
    save_image(result['roi_image'], f"{output_dir}/02_roi.png")

    # 3. 各类别mask
    for i, mask in enumerate(result['class_masks']):
        save_image(mask, f"{output_dir}/03_class_mask_{i+1}.png")

    # 4. 精修后的mask
    for i, mask in enumerate(result['refined_masks']):
        save_image(mask, f"{output_dir}/04_refined_mask_{i+1}.png")

    # 5. Mask可视化
    mask_vis = create_mask_visualization(result['refined_masks'], result['roi_image'].shape)
    save_image(mask_vis, f"{output_dir}/05_masks_visualization.png")

    # 6. 最终结果
    result_image = draw_curves_on_image(result['roi_image'], result['pixel_paths'])
    save_image(result_image, f"{output_dir}/06_final_result.png")

    print(f"✓ 调试可视化保存到: {output_dir}")
    print()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("KM曲线提取工具 - 使用示例")
    print("=" * 60 + "\n")

    print("注意: 请先修改示例中的图像路径为实际路径\n")

    # 取消注释以运行相应示例
    # example_basic()
    # example_with_custom_roi()
    # example_batch_processing()
    # example_debug_visualization()

    print("提示: 取消注释相应函数以运行示例")


if __name__ == "__main__":
    main()
