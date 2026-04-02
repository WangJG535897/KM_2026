"""图像IO"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional


def load_image(image_path: str) -> Optional[np.ndarray]:
    """加载图像 - 支持中文路径"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 使用numpy读取以支持中文路径
    try:
        # 方法1: 使用numpy fromfile
        with open(str(path), 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"无法解码图像: {image_path}")

        return image
    except Exception as e:
        raise ValueError(f"无法读取图像: {image_path}, 错误: {e}")


def save_image(image: np.ndarray, output_path: str):
    """保存图像 - 支持中文路径"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 使用imencode支持中文路径
    ext = path.suffix.lower()
    if not ext:
        ext = '.png'

    success, encoded_image = cv2.imencode(ext, image)
    if success:
        with open(str(path), 'wb') as f:
            f.write(encoded_image.tobytes())
    else:
        raise ValueError(f"无法编码图像: {output_path}")
