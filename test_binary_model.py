"""测试二分类模型"""
import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'training')
from model import UNet

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    print("加载模型...")
    model = UNet(in_channels=3).to(device)
    checkpoint = torch.load('models/best_model_binary.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ 模型加载成功")

    # 加载测试图像
    img_path = r'C:/Users/32665/Desktop/生存曲线数据集/image_108.png'
    print(f"\n加载图像: {img_path}")

    with open(img_path, 'rb') as f:
        img_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"图像尺寸: {image.shape}")

    # 预处理
    image_resized = cv2.resize(image, (512, 512))
    tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0).to(device)

    # 推理
    print("\n推理中...")
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output)

    # 统计
    prob_np = prob[0, 0].cpu().numpy()
    print(f"\n概率统计:")
    print(f"  最大值: {prob_np.max():.4f}")
    print(f"  最小值: {prob_np.min():.4f}")
    print(f"  平均值: {prob_np.mean():.4f}")
    print(f"  >0.3的像素: {(prob_np > 0.3).sum()} ({(prob_np > 0.3).sum() / prob_np.size * 100:.2f}%)")
    print(f"  >0.5的像素: {(prob_np > 0.5).sum()} ({(prob_np > 0.5).sum() / prob_np.size * 100:.2f}%)")

    # 保存结果
    output_dir = Path('outputs/test_binary')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 概率热图
    heatmap = (prob_np * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / 'prob_heatmap.png'), heatmap_color)

    # 不同阈值的二值图
    for thresh in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        binary = (prob_np > thresh).astype(np.uint8) * 255
        cv2.imwrite(str(output_dir / f'binary_thresh_{thresh:.2f}.png'), binary)
        fg_pixels = (prob_np > thresh).sum()
        print(f"  阈值{thresh}: {fg_pixels}像素 ({fg_pixels / prob_np.size * 100:.2f}%)")

    print(f"\n结果已保存到: {output_dir}")
    print("\n请检查:")
    print("  1. prob_heatmap.png - 概率热图")
    print("  2. binary_thresh_*.png - 不同阈值的二值图")

if __name__ == '__main__':
    test_model()
