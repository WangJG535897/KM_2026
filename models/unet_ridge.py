"""
U-Net for Ridge Heatmap Regression
高分辨率 U-Net，专用于细线 centerline heatmap 预测
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class UNetRidge(nn.Module):
    """
    U-Net with ResNet encoder for ridge heatmap regression
    
    输出：(B, K, H, W) logits，K=曲线数量
    """
    
    def __init__(self, num_classes=2, pretrained=True, encoder='resnet34'):
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
        
        # 提取 ResNet 各层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Decoder
        self.up4 = UpBlock(encoder_channels[4], encoder_channels[3])
        self.up3 = UpBlock(encoder_channels[3], encoder_channels[2])
        self.up2 = UpBlock(encoder_channels[2], encoder_channels[1])
        self.up1 = UpBlock(encoder_channels[1], encoder_channels[0])
        
        # 最终上采样到原始分辨率
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[0], 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 输出层
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x0 = self.conv1(x)  # /2
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        
        x1 = self.maxpool(x0)  # /4
        x1 = self.layer1(x1)
        
        x2 = self.layer2(x1)  # /8
        x3 = self.layer3(x2)  # /16
        x4 = self.layer4(x3)  # /32
        
        # Decoder with skip connections
        d4 = self.up4(x4, x3)  # /16
        d3 = self.up3(d4, x2)  # /8
        d2 = self.up2(d3, x1)  # /4
        d1 = self.up1(d2, x0)  # /2
        
        d0 = self.up0(d1)  # /1 (原始分辨率)
        
        # 输出 logits
        out = self.out_conv(d0)
        
        return out


class UpBlock(nn.Module):
    """上采样块 with skip connection"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 
                                      kernel_size=2, stride=2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # 处理尺寸不匹配
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        return x


class CombinedLoss(nn.Module):
    """
    组合损失：BCE + Soft Dice
    """
    
    def __init__(self, bce_weight=1.0, dice_weight=0.5, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(self, logits, targets):
        """
        logits: (B, K, H, W)
        targets: (B, K, H, W) float [0, 1]
        """
        # Focal BCE (对稀疏细线更稳定)
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = self.focal_alpha * (1 - pt) ** self.focal_gamma
        
        # 重要：weight 参数不能有梯度
        focal_bce = F.binary_cross_entropy_with_logits(
            logits, targets, weight=focal_weight.detach(), reduction='mean'
        )
        
        # Soft Dice Loss
        dice = self.soft_dice_loss(probs, targets)
        
        # 总损失
        total_loss = self.bce_weight * focal_bce + self.dice_weight * dice
        
        return total_loss, focal_bce, dice
    
    def soft_dice_loss(self, probs, targets, smooth=1.0):
        """
        Soft Dice Loss（每个通道独立计算）
        """
        # probs, targets: (B, K, H, W)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


def compute_metrics(logits, targets, threshold=0.5):
    """
    计算评估指标
    
    返回：
    - dice_mean: 平均 Dice
    - dice_per_class: 每个类别的 Dice
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    # 每个类别的 Dice
    dice_per_class = []
    
    for k in range(logits.shape[1]):
        pred_k = preds[:, k]
        target_k = (targets[:, k] > 0.5).float()
        
        intersection = (pred_k * target_k).sum()
        union = pred_k.sum() + target_k.sum()
        
        if union > 0:
            dice = (2.0 * intersection / union).item()
        else:
            dice = 1.0  # 都为空则认为完美
        
        dice_per_class.append(dice)
    
    metrics = {
        'dice_mean': np.mean(dice_per_class),
        'dice_per_class': dice_per_class
    }
    
    return metrics


if __name__ == '__main__':
    import numpy as np
    
    # 测试模型
    model = UNetRidge(num_classes=2, pretrained=False, encoder='resnet34')
    
    x = torch.randn(2, 3, 896, 896)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # 测试损失
    targets = torch.rand(2, 2, 896, 896)
    criterion = CombinedLoss()
    loss, bce, dice = criterion(y, targets)
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"BCE: {bce.item():.4f}")
    print(f"Dice: {dice.item():.4f}")
    
    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
