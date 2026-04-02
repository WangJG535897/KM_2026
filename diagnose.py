"""诊断脚本 - 查看中间处理结果"""
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent / "src"))

from km_app.model import ModelInference
from km_app.pipeline import KMPipeline
from km_app.io import load_image, save_image
from km_app.utils import create_mask_visualization
import numpy as np
import cv2


def diagnose():
    """诊断处理流程"""
    print("=" * 60)
    print("诊断脚本")
    print("=" * 60)

    image_path = r"C:/Users/32665/Desktop/生存曲线数据集/image_108.png"
    checkpoint_path = "models/best_model.pth"
    output_dir = "outputs/diagnosis"
    x_max = 42.0

    try:
        # 1. 加载图像
        print("\n[1] 加载图像...")
        image = load_image(image_path)
        print(f"  图像尺寸: {image.shape}")

        # 2. 加载模型并推理
        print("\n[2] 模型推理...")
        inference = ModelInference(checkpoint_path)
        inference.load_model()
        pred_result = inference.predict(image)

        # 保存模型原始输出
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 可视化每个类别的概率图
        probs = pred_result['probs']
        for i in range(pred_result['num_classes']):
            prob_map = (probs[i] * 255).astype(np.uint8)
            prob_colored = cv2.applyColorMap(prob_map, cv2.COLORMAP_JET)
            save_image(prob_colored, f"{output_dir}/prob_class_{i}.png")
            print(f"  保存类别{i}概率图: prob_class_{i}.png")

        # 保存argmax结果
        class_mask = pred_result['class_mask']
        class_mask_colored = (class_mask * 40).astype(np.uint8)
        class_mask_colored = cv2.applyColorMap(class_mask_colored, cv2.COLORMAP_JET)
        save_image(class_mask_colored, f"{output_dir}/class_mask.png")
        print(f"  保存分类mask: class_mask.png")

        # 统计每个类别的像素数
        print("\n[3] 类别统计:")
        for i in range(pred_result['num_classes']):
            count = np.sum(class_mask == i)
            percentage = count / class_mask.size * 100
            print(f"  类别{i}: {count:8d} 像素 ({percentage:5.2f}%)")

        # 3. 处理曲线
        print("\n[4] 处理曲线...")
        pipeline = KMPipeline(x_max=x_max)
        result = pipeline.process(image, pred_result)

        print(f"  ROI: {result['roi']}")
        print(f"  提取曲线数: {result['num_curves']}")

        # 保存ROI
        save_image(result['roi_image'], f"{output_dir}/roi.png")

        # 保存每个mask
        for i, mask in enumerate(result['refined_masks']):
            save_image(mask, f"{output_dir}/refined_mask_{i+1}.png")
            print(f"  保存精修mask {i+1}: 像素数={np.sum(mask > 0)}")

        # 保存路径追踪结果
        if result['pixel_paths']:
            for i, path in enumerate(result['pixel_paths']):
                print(f"\n  曲线{i+1}路径信息:")
                print(f"    点数: {len(path)}")
                if len(path) > 0:
                    print(f"    X范围: {path[:, 0].min()} - {path[:, 0].max()}")
                    print(f"    Y范围: {path[:, 1].min()} - {path[:, 1].max()}")

                    # 在ROI上绘制路径
                    debug_img = result['roi_image'].copy()
                    for j in range(len(path) - 1):
                        pt1 = tuple(path[j].astype(int))
                        pt2 = tuple(path[j + 1].astype(int))
                        cv2.line(debug_img, pt1, pt2, (0, 255, 0), 2)
                    save_image(debug_img, f"{output_dir}/path_{i+1}.png")

        # 保存图表坐标
        if result['chart_coords']:
            for i, coords in enumerate(result['chart_coords']):
                print(f"\n  曲线{i+1}图表坐标:")
                if len(coords) > 0:
                    print(f"    时间范围: {coords[:, 0].min():.2f} - {coords[:, 0].max():.2f}")
                    print(f"    生存率范围: {coords[:, 1].min():.2f} - {coords[:, 1].max():.2f}")

        print(f"\n诊断结果已保存到: {output_dir}")
        print("\n请检查:")
        print("  1. prob_class_*.png - 每个类别的概率图")
        print("  2. class_mask.png - 最终分类结果")
        print("  3. refined_mask_*.png - 精修后的mask")
        print("  4. path_*.png - 追踪的路径")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    diagnose()
