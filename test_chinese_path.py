"""测试中文路径图像加载"""
import sys
import io
from pathlib import Path

# 设置UTF-8输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent / "src"))

from km_app.io import load_image, save_image
import numpy as np

def test_chinese_path():
    """测试中文路径"""
    # 测试路径
    test_path = r"C:/Users/32665/Desktop/生存曲线数据集/image_108.png"

    print("=" * 60)
    print("测试中文路径图像加载")
    print("=" * 60)
    print(f"测试路径: {test_path}")
    print()

    try:
        # 检查文件是否存在
        if not Path(test_path).exists():
            print(f"✗ 文件不存在: {test_path}")
            return

        print("✓ 文件存在")

        # 加载图像
        print("正在加载图像...")
        image = load_image(test_path)

        print(f"✓ 图像加载成功!")
        print(f"  - 尺寸: {image.shape}")
        print(f"  - 类型: {image.dtype}")
        print()

        # 测试保存
        output_path = "outputs/test_chinese_path.png"
        print(f"测试保存到: {output_path}")
        save_image(image, output_path)
        print("✓ 保存成功!")

    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_chinese_path()
