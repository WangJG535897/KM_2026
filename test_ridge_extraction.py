"""测试Ridge脊线提取功能"""
import sys
import numpy as np
import cv2
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from km_app.model.inference import ModelInference
from km_app.pipeline.postprocess import KMPipeline
from km_app.io.logger import setup_logger

# 设置日志
logger = setup_logger("test_ridge", log_file="test_ridge.log")

def test_ridge_extraction():
    """测试Ridge提取流程"""
    print("=" * 60)
    print("测试Ridge脊线提取功能")
    print("=" * 60)

    # 1. 加载模型
    model_path = "models/best_model_binary.pth"
    if not Path(model_path).exists():
        print(f"错误：模型文件不存在: {model_path}")
        return

    print(f"\n[1/4] 加载模型: {model_path}")
    inferencer = ModelInference(model_path, device='cpu')
    inferencer.load_model()

    # 2. 查找测试图像
    test_images = list(Path("training_data/images").glob("*.png"))
    if not test_images:
        print("错误：未找到测试图像")
        return

    test_image_path = test_images[0]
    print(f"\n[2/4] 加载测试图像: {test_image_path}")
    image = cv2.imread(str(test_image_path))
    if image is None:
        print(f"错误：无法加载图像")
        return

    print(f"  图像尺寸: {image.shape}")

    # 3. 推理
    print(f"\n[3/4] 模型推理...")
    pred_result = inferencer.predict(image)
    print(f"  模式: {pred_result['mode']}")
    print(f"  概率图尺寸: {pred_result['prob_map'].shape}")
    print(f"  概率图统计: min={pred_result['prob_map'].min():.3f}, "
          f"max={pred_result['prob_map'].max():.3f}, "
          f"mean={pred_result['prob_map'].mean():.3f}")

    # 4. 后处理（Ridge提取）
    print(f"\n[4/4] 后处理（Ridge主链提取）...")
    pipeline = KMPipeline(x_max=48.0, y_range=(0.0, 100.0))
    result = pipeline.process(image, pred_result)

    # 5. 输出结果
    print(f"\n" + "=" * 60)
    print("提取结果")
    print("=" * 60)
    print(f"  选择方法: {result['selected_method']}")
    print(f"  Ridge路径数: {result.get('ridge_paths_count', 'N/A')}")
    print(f"  Regular路径数: {result.get('regular_paths_count', 'N/A')}")
    print(f"  最终曲线数: {result['num_curves']}")
    print(f"  估计曲线数: {result['estimated_curve_count']}")

    if result['num_curves'] > 0:
        print(f"\n  ✓ 成功提取 {result['num_curves']} 条曲线")
        for i, coords in enumerate(result['chart_coords']):
            print(f"    曲线{i+1}: {len(coords)} 个点")
    else:
        print(f"\n  ⚠️ 未提取到曲线")

    # 6. 检查调试文件
    output_dir = Path("outputs")
    debug_files = [
        "prob_map_suppressed.png",
        "horizontal_suppression_mask.png",
        "ridge_paths.png"
    ]

    print(f"\n调试文件:")
    for fname in debug_files:
        fpath = output_dir / fname
        if fpath.exists():
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname} (未生成)")

    print(f"\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print(f"\n详细日志请查看: test_ridge.log")
    print(f"调试图像请查看: outputs/")

if __name__ == "__main__":
    try:
        test_ridge_extraction()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
