"""完整流程测试 - 使用实际图像"""
import sys
import io
from pathlib import Path

# 设置UTF-8输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent / "src"))

from km_app.model import ModelInference
from km_app.pipeline import KMPipeline
from km_app.io import load_image, save_image, export_all_curves
from km_app.utils import draw_curves_on_image


def test_full_pipeline():
    """测试完整流程"""
    print("=" * 60)
    print("完整流程测试")
    print("=" * 60)

    # 配置
    image_path = r"C:/Users/32665/Desktop/生存曲线数据集/image_108.png"
    checkpoint_path = "models/best_model_new.pth"  # 使用新训练的模型
    output_dir = "outputs/test_full"
    x_max = 40.0

    try:
        # 1. 加载图像
        print("\n[1/5] 加载图像...")
        image = load_image(image_path)
        print(f"  ✓ 图像尺寸: {image.shape}")

        # 2. 加载模型
        print("\n[2/5] 加载模型...")
        inference = ModelInference(checkpoint_path)
        inference.load_model()
        print(f"  ✓ 模型已加载")

        # 3. 推理
        print("\n[3/5] 模型推理...")
        pred_result = inference.predict(image)
        print(f"  ✓ 推理完成")
        print(f"  - 输出类别数: {pred_result['num_classes']}")
        print(f"  - 输出形状: {pred_result['logits'].shape}")

        # 调试：检查每个类别的像素数
        import numpy as np
        class_mask = pred_result['class_mask']
        print(f"  - Class mask shape: {class_mask.shape}")
        for i in range(pred_result['num_classes']):
            count = np.sum(class_mask == i)
            print(f"    类别 {i}: {count} 像素 ({count/class_mask.size*100:.2f}%)")

        # 保存class_mask用于调试
        Path("outputs/debug").mkdir(parents=True, exist_ok=True)
        import cv2
        cv2.imwrite("outputs/debug/class_mask.png", class_mask * 85)  # 乘以85让类别可见

        # 4. 处理曲线
        print("\n[4/5] 处理曲线...")
        pipeline = KMPipeline(x_max=x_max)
        result = pipeline.process(image, pred_result)
        print(f"  ✓ 提取到 {result['num_curves']} 条曲线")
        print(f"  - ROI: {result['roi']}")

        # 5. 保存结果
        print("\n[5/5] 保存结果...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 保存原图
        save_image(image, f"{output_dir}/01_original.png")

        # 保存ROI
        save_image(result['roi_image'], f"{output_dir}/02_roi.png")

        # 保存结果图
        if result['pixel_paths']:
            result_image = draw_curves_on_image(result['roi_image'], result['pixel_paths'])
            save_image(result_image, f"{output_dir}/03_result.png")

        # 保存masks
        for i, mask in enumerate(result['refined_masks']):
            save_image(mask, f"{output_dir}/04_mask_{i+1}.png")

        # 导出CSV
        if result['chart_coords']:
            export_all_curves(result['chart_coords'], output_dir, "curve")

        print(f"  ✓ 结果已保存到: {output_dir}")

        # 总结
        print("\n" + "=" * 60)
        print("测试完成!")
        print("=" * 60)
        print(f"提取曲线数: {result['num_curves']}")
        print(f"输出目录: {output_dir}")
        print("\n请检查输出目录中的结果文件:")
        print("  - 01_original.png: 原始图像")
        print("  - 02_roi.png: ROI区域")
        print("  - 03_result.png: 曲线叠加结果")
        print("  - 04_mask_*.png: 各类别mask")
        print("  - curve_*.csv: 曲线坐标数据")
        print("  - all_curves.csv: 所有曲线合并")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_full_pipeline()
