"""训练数据集 - 二分类版本"""
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class KMCurveDataset(Dataset):
    """KM曲线数据集 - 二分类：背景 vs 曲线前景"""

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        self.image_files = sorted(list(self.images_dir.glob("*.png")))
        print(f"数据集: {len(self.image_files)} 张图像")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.masks_dir / image_path.name

        # 读取图像
        with open(str(image_path), 'rb') as f:
            image_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取mask
        with open(str(mask_path), 'rb') as f:
            mask_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        mask = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)

        # 合并所有前景类为1（numpy阶段）
        binary_mask = (mask > 0).astype(np.float32)

        # 数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']  # tensor (3, H, W)
            binary_mask = augmented['mask']  # tensor (H, W)

        # 稳定的二值化：兼容tensor
        if isinstance(binary_mask, torch.Tensor):
            binary_mask = (binary_mask > 0.5).float()
        else:
            binary_mask = torch.from_numpy((binary_mask > 0.5).astype(np.float32))

        # 确保shape正确
        if binary_mask.dim() == 2:
            binary_mask = binary_mask.unsqueeze(0)  # (1, H, W)

        return image, binary_mask


def get_train_transform(image_size=512):
    """训练数据增强"""
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transform(image_size=512):
    """验证数据变换"""
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
