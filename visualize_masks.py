"""批量生成mask可视化版本"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def visualize_mask(mask_path, output_path):
    """将mask转换为可视化版本（白色曲线）"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # 将所有非0值变为255（白色）
    vis = np.zeros_like(mask)
    vis[mask > 0] = 255

    cv2.imwrite(str(output_path), vis)

def main():
    masks_dir = Path('training_data/masks')
    output_dir = Path('training_data/masks_vis')
    output_dir.mkdir(exist_ok=True)

    mask_files = list(masks_dir.glob('*.png'))
    # 排除已经是可视化版本的文件
    mask_files = [f for f in mask_files if '_vis' not in f.name]

    print(f'找到 {len(mask_files)} 个mask文件')
    print(f'输出目录: {output_dir}')

    for mask_file in tqdm(mask_files, desc='生成可视化版本'):
        output_file = output_dir / mask_file.name
        visualize_mask(mask_file, output_file)

    print(f'\n完成! 可视化文件保存在: {output_dir}')

if __name__ == '__main__':
    main()
