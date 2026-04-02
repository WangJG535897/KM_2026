"""模型适配器 - 基于实际checkpoint结构重建"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """Bottleneck残差块 (1x1 -> 3x3 -> 1x1)"""
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class DoubleConv(nn.Module):
    """双卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class KMSegmentationModel(nn.Module):
    """KM曲线分割模型 - 匹配实际checkpoint"""

    def __init__(self, in_channels=3, num_classes=6):
        super().__init__()

        # Encoder1: 初始卷积
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Encoder2: MaxPool + Bottleneck
        downsample2 = nn.Sequential(
            nn.Conv2d(64, 256, 1, 1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            Bottleneck(64, 64, 256, 1, downsample2),
            Bottleneck(256, 64, 256)
        )

        # Encoder3
        downsample3 = nn.Sequential(
            nn.Conv2d(256, 512, 1, 2, bias=False),
            nn.BatchNorm2d(512)
        )
        self.encoder3 = nn.Sequential(
            Bottleneck(256, 128, 512, 2, downsample3),
            Bottleneck(512, 128, 512)
        )

        # Encoder4
        downsample4 = nn.Sequential(
            nn.Conv2d(512, 1024, 1, 2, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.encoder4 = nn.Sequential(
            Bottleneck(512, 256, 1024, 2, downsample4),
            Bottleneck(1024, 256, 1024)
        )

        # Encoder5
        downsample5 = nn.Sequential(
            nn.Conv2d(1024, 2048, 1, 2, bias=False),
            nn.BatchNorm2d(2048)
        )
        self.encoder5 = nn.Sequential(
            Bottleneck(1024, 512, 2048, 2, downsample5),
            Bottleneck(2048, 512, 2048)
        )

        # Decoder (直接上采样，不使用skip connection)
        self.decoder4 = DoubleConv(2048, 1024)
        self.decoder3 = DoubleConv(1024, 512)
        self.decoder2 = DoubleConv(512, 256)
        self.decoder1 = DoubleConv(128, 64)
        self.decoder0 = DoubleConv(32, 32)

        # 通道调整层
        self.channel_adjust_256_to_128 = nn.Conv2d(256, 128, 1)
        self.channel_adjust_64_to_32 = nn.Conv2d(64, 32, 1)

        # Final
        self.final_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)      # /2, 64
        e2 = self.encoder2(e1)     # /4, 256
        e3 = self.encoder3(e2)     # /8, 512
        e4 = self.encoder4(e3)     # /16, 1024
        e5 = self.encoder5(e4)     # /32, 2048

        # Decoder (逐步上采样)
        d4 = self.decoder4(e5)     # 1024
        d4 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)

        d3 = self.decoder3(d4)     # 512
        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)

        d2 = self.decoder2(d3)     # 256
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)

        # 256 -> 128
        d2_adjusted = self.channel_adjust_256_to_128(d2)  # 128
        d1 = self.decoder1(d2_adjusted)  # 64
        d1 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)

        # 64 -> 32
        d1_adjusted = self.channel_adjust_64_to_32(d1)  # 32
        d0 = self.decoder0(d1_adjusted)  # 32
        d0 = F.interpolate(d0, size=x.shape[2:], mode='bilinear', align_corners=False)

        out = self.final_conv(d0)
        return out
