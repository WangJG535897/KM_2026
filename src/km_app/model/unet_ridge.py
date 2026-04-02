"""UNetRidge模型 - 原始定义"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class UpBlock(nn.Module):
    """上采样块，带skip connection"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 使用upconv命名以匹配checkpoint
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upconv(x)
        # 确保尺寸匹配
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv(x)
        return x


class UNetRidge(nn.Module):
    """
    U-Net with ResNet encoder for ridge heatmap regression

    输出：(B, K, H, W) logits，K=曲线数量
    """

    def __init__(self, num_classes=2, pretrained=True, encoder='resnet50'):
        super().__init__()
        self.num_classes = num_classes

        # Encoder (ResNet)
        if encoder == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")

        # 提取 ResNet 各层 - 使用encoder命名以匹配checkpoint
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        self.encoder2 = nn.Sequential(
            resnet.maxpool,
            resnet.layer1
        )

        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        # Decoder - 使用double_conv命名以匹配checkpoint
        self.upconv4 = nn.ConvTranspose2d(encoder_channels[4], 1024, kernel_size=2, stride=2)
        self.decoder4 = nn.Module()
        self.decoder4.double_conv = nn.Sequential(
            nn.Conv2d(1024 + encoder_channels[3], 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder3 = nn.Module()
        self.decoder3.double_conv = nn.Sequential(
            nn.Conv2d(512 + encoder_channels[2], 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = nn.Module()
        self.decoder2.double_conv = nn.Sequential(
            nn.Conv2d(256 + encoder_channels[1], 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Module()
        self.decoder1.double_conv = nn.Sequential(
            nn.Conv2d(64 + encoder_channels[0], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 最终上采样到原始分辨率
        self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder0 = nn.Module()
        self.decoder0.double_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 输出层
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)   # /2, 64
        e2 = self.encoder2(e1)  # /4, 64 or 256
        e3 = self.encoder3(e2)  # /8, 128 or 512
        e4 = self.encoder4(e3)  # /16, 256 or 1024
        e5 = self.encoder5(e4)  # /32, 512 or 2048

        # Decoder with skip connections
        d4 = self.upconv4(e5)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4.double_conv(d4)

        d3 = self.upconv3(d4)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3.double_conv(d3)

        d2 = self.upconv2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2.double_conv(d2)

        d1 = self.upconv1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1.double_conv(d1)

        d0 = self.upconv0(d1)
        d0 = self.decoder0.double_conv(d0)

        # 输出 logits
        out = self.final_conv(d0)

        return out
