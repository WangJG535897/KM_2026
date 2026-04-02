"""模型推理"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import cv2


# 内嵌二分类UNet定义
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class BinaryUNet(nn.Module):
    """UNet - 二分类：输出1通道前景概率"""

    def __init__(self, in_channels=3):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # 输出：1通道logits
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # 输出logits（不做sigmoid）
        return self.out(d1)


class ModelInference:
    """模型推理器 - 支持二分类和多类分割"""

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.input_channels = 3
        self.mode = None  # 'binary' or 'multiclass'
        self.num_classes = None

    def load_model(self):
        """加载模型 - 自动识别二分类或多类"""
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)

        # 检查模型类型
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 检查输出层通道数
        out_key = 'out.weight'
        if out_key in state_dict:
            out_channels = state_dict[out_key].shape[0]
            if out_channels == 1:
                self.mode = 'binary'
                self.num_classes = 1
                print(f"检测到二分类模型 (1通道输出)")
            else:
                self.mode = 'multiclass'
                self.num_classes = out_channels
                print(f"检测到多类分割模型 ({out_channels}类)")
        else:
            raise ValueError("无法识别模型类型")

        # 加载对应模型
        if self.mode == 'binary':
            self.model = BinaryUNet(in_channels=3)
        else:
            from .unet import UNet as MulticlassUNet
            self.model = MulticlassUNet(in_channels=3, num_classes=self.num_classes)

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ 模型加载成功 (模式: {self.mode}, 设备: {self.device})")

    def preprocess(self, image: np.ndarray, roi: Tuple[int, int, int, int] = None,
                   target_size: Tuple[int, int] = (512, 512)) -> Tuple[torch.Tensor, Dict]:
        """预处理图像 - ROI优先"""
        # 如果指定ROI，先裁剪
        if roi is not None:
            x1, y1, x2, y2 = roi
            image = image[y1:y2, x1:x2]

        original_h, original_w = image.shape[:2]

        # Resize
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        # 转换为tensor并归一化（与训练一致）
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(self.device)

        transform_info = {
            'original_size': (original_h, original_w),
            'target_size': target_size,
            'roi': roi
        }

        return tensor, transform_info

    def postprocess(self, output: torch.Tensor, transform_info: Dict) -> Dict:
        """后处理 - 区分二分类和多类"""
        original_h, original_w = transform_info['original_size']

        if self.mode == 'binary':
            # 二分类：output shape [1, 1, H, W]
            logits = output[0, 0].cpu().numpy()  # [H, W]
            prob_map = torch.sigmoid(output[0, 0]).cpu().numpy()  # [H, W]

            # Resize回原图尺寸
            prob_map_resized = cv2.resize(prob_map, (original_w, original_h),
                                         interpolation=cv2.INTER_LINEAR)
            logits_resized = cv2.resize(logits, (original_w, original_h),
                                       interpolation=cv2.INTER_LINEAR)

            return {
                'mode': 'binary',
                'prob_map': prob_map_resized,  # [H, W]
                'raw_logits': logits_resized,  # [H, W]
                'transform_info': transform_info
            }
        else:
            # 多类：output shape [1, num_classes, H, W]
            logits = output[0].cpu().numpy()  # [num_classes, H, W]

            # Resize回原图尺寸
            masks = []
            for i in range(logits.shape[0]):
                mask = cv2.resize(logits[i], (original_w, original_h),
                                interpolation=cv2.INTER_LINEAR)
                masks.append(mask)
            masks = np.stack(masks, axis=0)  # [num_classes, H, W]

            # 生成类别mask和概率
            class_mask = masks.argmax(axis=0).astype(np.uint8)
            probs = torch.softmax(torch.from_numpy(masks), dim=0).numpy()

            return {
                'mode': 'multiclass',
                'logits': masks,
                'probs': probs,
                'class_mask': class_mask,
                'num_classes': self.num_classes,
                'transform_info': transform_info
            }

    @torch.no_grad()
    def predict(self, image: np.ndarray, roi: Tuple[int, int, int, int] = None,
                target_size: Tuple[int, int] = (512, 512)) -> Dict:
        """推理 - ROI优先"""
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用load_model()")

        # 预处理（包含ROI裁剪）
        tensor, transform_info = self.preprocess(image, roi, target_size)

        # 推理
        output = self.model(tensor)

        # 后处理
        result = self.postprocess(output, transform_info)

        return result
