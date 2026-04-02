"""快速测试脚本 - 验证模型加载和基本功能"""
import sys
import io
from pathlib import Path

# 设置UTF-8输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent / "src"))

from km_app.model import CheckpointInspector, ModelInference
from km_app.config import DEFAULT_MODEL_PATH


def test_checkpoint():
    """测试checkpoint检查"""
    print("=" * 60)
    print("测试1: Checkpoint检查")
    print("=" * 60)

    inspector = CheckpointInspector(DEFAULT_MODEL_PATH)
    state_dict = inspector.load()
    inspector.print_summary()

    print("\n✓ Checkpoint检查通过\n")


def test_model_loading():
    """测试模型加载"""
    print("=" * 60)
    print("测试2: 模型加载")
    print("=" * 60)

    inference = ModelInference(DEFAULT_MODEL_PATH)
    inference.load_model()

    print(f"✓ 模型加载成功")
    print(f"  - 输入通道: {inference.input_channels}")
    print(f"  - 输出类别: {inference.num_classes}")
    print(f"  - 设备: {inference.device}")
    print()


def test_inference():
    """测试推理"""
    print("=" * 60)
    print("测试3: 推理测试（使用随机图像）")
    print("=" * 60)

    import numpy as np

    # 创建随机测试图像
    test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

    inference = ModelInference(DEFAULT_MODEL_PATH)
    inference.load_model()

    result = inference.predict(test_image)

    print(f"✓ 推理成功")
    print(f"  - 输出形状: {result['logits'].shape}")
    print(f"  - 类别数: {result['num_classes']}")
    print(f"  - Class mask形状: {result['class_mask'].shape}")
    print()


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("KM曲线提取工具 - 快速测试")
    print("=" * 60 + "\n")

    try:
        # 检查模型文件是否存在
        if not Path(DEFAULT_MODEL_PATH).exists():
            print(f"✗ 错误: 模型文件不存在: {DEFAULT_MODEL_PATH}")
            return

        # 运行测试
        test_checkpoint()
        test_model_loading()
        test_inference()

        print("=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)
        print("\n可以开始使用:")
        print("  GUI模式: python app.py")
        print("  CLI模式: python cli.py --image <图像路径>")
        print()

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
