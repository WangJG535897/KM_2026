"""训练脚本 - 二分类版本"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import sys
import io
from tqdm import tqdm
import numpy as np
import cv2
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent))

from model import UNet
from dataset import KMCurveDataset, get_train_transform, get_val_transform


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

        return 1.0 - dice


class CombinedLoss(nn.Module):
    """BCE + Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=50.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        # pos_weight作为buffer，自动跟随模型设备
        self.register_buffer('pos_weight_tensor', torch.tensor([pos_weight]))
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        # 动态创建BCE loss，使用当前设备的pos_weight
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)
        bce = bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


def compute_metrics(pred, target):
    """计算评估指标"""
    pred_binary = (torch.sigmoid(pred) > 0.3).float()
    target_binary = (target > 0.5).float()

    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)

    return {'precision': precision, 'recall': recall, 'dice': dice}


def save_debug_images(images, masks, preds, epoch, output_dir, phase='train', num_samples=4):
    """保存调试图像"""
    debug_dir = output_dir / 'debug' / f'epoch_{epoch:03d}' / phase
    debug_dir.mkdir(parents=True, exist_ok=True)

    num_samples = min(num_samples, images.shape[0])

    for i in range(num_samples):
        # 原图
        img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)

        # GT mask
        gt_mask = (masks[i, 0].detach().cpu().numpy() * 255).astype(np.uint8)

        # 预测概率图
        pred_prob = torch.sigmoid(preds[i, 0]).detach().cpu().numpy()
        pred_heatmap = (pred_prob * 255).astype(np.uint8)
        pred_heatmap_color = cv2.applyColorMap(pred_heatmap, cv2.COLORMAP_JET)

        # 多阈值二值化
        for thresh in [0.2, 0.3, 0.4]:
            pred_binary = (pred_prob > thresh).astype(np.uint8) * 255
            cv2.imwrite(str(debug_dir / f'sample_{i}_binary_t{thresh:.1f}.png'), pred_binary)

        # 保存
        cv2.imwrite(str(debug_dir / f'sample_{i}_image.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(debug_dir / f'sample_{i}_gt.png'), gt_mask)
        cv2.imwrite(str(debug_dir / f'sample_{i}_prob.png'), pred_heatmap_color)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, output_dir):
    model.train()
    total_loss = 0
    fg_probs = []
    all_metrics = {'precision': [], 'recall': [], 'dice': []}

    pbar = tqdm(dataloader, desc="训练")
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 统计
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            max_prob = probs.max().item()
            mean_fg_prob = probs[masks > 0.5].mean().item() if (masks > 0.5).any() else 0.0
            fg_probs.append(mean_fg_prob)

            metrics = compute_metrics(outputs, masks)
            for k, v in metrics.items():
                all_metrics[k].append(v)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'fg_p': f'{mean_fg_prob:.3f}',
            'dice': f'{metrics["dice"]:.3f}'
        })

        if batch_idx == 0:
            save_debug_images(images, masks, outputs, epoch, output_dir, 'train')

    avg_loss = total_loss / len(dataloader)
    avg_fg_prob = np.mean(fg_probs) if fg_probs else 0.0
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}

    return avg_loss, avg_fg_prob, avg_metrics


def validate(model, dataloader, criterion, device, epoch, output_dir):
    model.eval()
    total_loss = 0
    fg_probs = []
    all_metrics = {'precision': [], 'recall': [], 'dice': []}

    pbar = tqdm(dataloader, desc="验证")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            max_prob = probs.max().item()
            mean_fg_prob = probs[masks > 0.5].mean().item() if (masks > 0.5).any() else 0.0
            fg_probs.append(mean_fg_prob)

            metrics = compute_metrics(outputs, masks)
            for k, v in metrics.items():
                all_metrics[k].append(v)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'fg_p': f'{mean_fg_prob:.3f}',
                'dice': f'{metrics["dice"]:.3f}'
            })

            if batch_idx == 0:
                save_debug_images(images, masks, outputs, epoch, output_dir, 'val')

    avg_loss = total_loss / len(dataloader)
    avg_fg_prob = np.mean(fg_probs) if fg_probs else 0.0
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}

    return avg_loss, avg_fg_prob, avg_metrics


def train(config):
    set_seed(config.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("KM曲线分割模型训练 - 二分类版本")
    if config.get('smoke_test', False):
        print("【冒烟测试模式】")
    print("=" * 60)
    print(f"设备: {device}")

    # 数据集
    print("\n加载数据集...")
    train_dataset = KMCurveDataset(
        config['images_dir'],
        config['masks_dir'],
        transform=get_train_transform(config['image_size'])
    )

    val_dataset = KMCurveDataset(
        config['images_dir'],
        config['masks_dir'],
        transform=get_val_transform(config['image_size'])
    )

    # 稳定的拆分逻辑
    total_size = len(train_dataset)
    indices = list(range(total_size))
    random.shuffle(indices)

    train_size = int(0.8 * total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    print(f"训练集: {len(train_subset)} 张")
    print(f"验证集: {len(val_subset)} 张")

    train_loader = DataLoader(
        train_subset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # 模型
    print("\n创建模型...")
    model = UNet(in_channels=3).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 损失和优化器
    print(f"\n使用 BCE(pos_weight={config['pos_weight']}) + Dice Loss")
    criterion = CombinedLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=config['pos_weight']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # 训练
    print("\n开始训练...")
    print("=" * 60)

    best_val_loss = float('inf')
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 60)

        train_loss, train_fg_prob, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, output_dir
        )
        val_loss, val_fg_prob, val_metrics = validate(
            model, val_loader, criterion, device, epoch, output_dir
        )

        scheduler.step()

        print(f"训练 - Loss: {train_loss:.4f}, FG_Prob: {train_fg_prob:.3f}, Dice: {train_metrics['dice']:.3f}")
        print(f"验证 - Loss: {val_loss:.4f}, FG_Prob: {val_fg_prob:.3f}, Dice: {val_metrics['dice']:.3f}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_metrics['dice'],
            }, output_dir / 'best_model_binary.pth')
            print(f"✓ 保存最佳模型 (val_loss: {val_loss:.4f}, val_dice: {val_metrics['dice']:.3f})")

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存在: {output_dir / 'best_model_binary.pth'}")
    print(f"调试图像保存在: {output_dir / 'debug'}")
    print("=" * 60)


if __name__ == "__main__":
    # 基线配置
    config = {
        'images_dir': r'C:\Users\32665\Desktop\KM_2026.3.31\training_data\images',
        'masks_dir': r'C:\Users\32665\Desktop\KM_2026.3.31\training_data\masks',
        'image_size': 512,
        'batch_size': 2,
        'epochs': 100,  # 正式训练
        'lr': 1e-4,
        'num_workers': 0,
        'pos_weight': 50.0,
        'output_dir': r'C:\Users\32665\Desktop\KM_2026.3.31\models',
        'seed': 42,
        'smoke_test': False  # 改为True进行冒烟测试
    }

    # 冒烟测试配置
    if config['smoke_test']:
        config['epochs'] = 10

    train(config)
