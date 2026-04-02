"""数据集处理 - KM曲线专用mask生成"""
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def create_km_curve_mask(image_shape, shapes, line_thickness=6, debug=False):
    """
    从linestrip标注创建KM曲线mask

    核心原则：
    1. 保留原始几何，不做平滑
    2. 使用细线条，避免粘连
    3. 保留step-like结构
    4. 每条曲线独立渲染

    Args:
        image_shape: (H, W)
        shapes: JSON中的shapes列表
        line_thickness: 曲线粗细（像素），建议6-8
        debug: 是否输出调试信息

    Returns:
        mask: (H, W) 0=背景, 1-N=各条曲线
        stats: 统计信息字典
    """
    h, w = image_shape

    # 每条曲线独立渲染
    curve_masks = {}
    stats = {
        'num_curves': 0,
        'per_curve_pixels': {},
        'overlap_pixels': 0,
        'total_fg_pixels': 0
    }

    for shape in shapes:
        if shape['shape_type'] != 'linestrip':
            continue

        try:
            curve_id = int(shape['label'])
        except:
            continue

        # 获取点序列
        points = np.array(shape['points'], dtype=np.float32)

        if len(points) < 2:
            continue

        # 过滤越界点
        points[:, 0] = np.clip(points[:, 0], 0, w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, h - 1)

        # 创建该曲线的独立mask
        curve_mask = np.zeros((h, w), dtype=np.uint8)

        # 转换为整数坐标
        points_int = np.round(points).astype(np.int32)

        # 逐段绘制折线，保留原始几何
        # 使用cv2.polylines而不是逐段cv2.line，避免端点重复膨胀
        cv2.polylines(
            curve_mask,
            [points_int],
            isClosed=False,
            color=1,
            thickness=line_thickness,
            lineType=cv2.LINE_8  # 使用8连通，避免断裂
        )

        curve_masks[curve_id] = curve_mask
        pixel_count = np.sum(curve_mask)
        stats['per_curve_pixels'][curve_id] = pixel_count
        stats['num_curves'] += 1

    # 检测overlap
    if len(curve_masks) > 1:
        all_masks = np.stack(list(curve_masks.values()), axis=0)
        overlap_map = (all_masks.sum(axis=0) > 1).astype(np.uint8)
        stats['overlap_pixels'] = np.sum(overlap_map)

    # 生成最终mask
    # 策略：后绘制的曲线优先（通常是前景曲线）
    final_mask = np.zeros((h, w), dtype=np.uint8)
    for curve_id in sorted(curve_masks.keys()):
        mask = curve_masks[curve_id]
        final_mask[mask > 0] = curve_id

    stats['total_fg_pixels'] = np.sum(final_mask > 0)
    stats['fg_ratio'] = stats['total_fg_pixels'] / (h * w)

    if debug:
        print(f"  曲线数: {stats['num_curves']}")
        for cid, pixels in stats['per_curve_pixels'].items():
            print(f"    曲线{cid}: {pixels}像素 ({pixels/(h*w)*100:.2f}%)")
        if stats['overlap_pixels'] > 0:
            print(f"  重叠像素: {stats['overlap_pixels']} ({stats['overlap_pixels']/(h*w)*100:.2f}%)")
        print(f"  总前景: {stats['total_fg_pixels']}像素 ({stats['fg_ratio']*100:.2f}%)")

    return final_mask, stats


def process_dataset(data_dir, output_dir, line_thickness=6, debug=False, save_debug_vis=False):
    """
    处理整个数据集

    Args:
        data_dir: 包含img_*.png和img_*.json的目录
        output_dir: 输出目录
        line_thickness: 曲线粗细（像素）
        debug: 是否输出统计信息
        save_debug_vis: 是否保存调试可视化
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # 创建输出目录
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    if save_debug_vis:
        debug_dir = output_dir / "debug_vis"
        debug_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有JSON文件
    json_files = sorted(data_dir.glob("*.json"))

    print(f'找到 {len(json_files)} 个标注文件')
    print(f'输出目录: {output_dir}')
    print(f'曲线粗细: {line_thickness} 像素')
    if debug:
        print('调试模式: 开启')
    print()

    # 统计
    global_stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'curves_per_image': [],
        'fg_ratios': [],
        'overlap_count': 0
    }

    for json_path in tqdm(json_files, desc='处理数据集'):
        try:
            # 读取JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 读取图像
            image_path = json_path.with_suffix('.png')
            with open(str(image_path), 'rb') as f:
                image_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")

            h, w = image.shape[:2]

            # 生成mask
            mask, stats = create_km_curve_mask(
                (h, w),
                data['shapes'],
                line_thickness=line_thickness,
                debug=debug and global_stats['success'] < 3  # 只打印前3个
            )

            # 统计
            global_stats['curves_per_image'].append(stats['num_curves'])
            global_stats['fg_ratios'].append(stats['fg_ratio'])
            if stats['overlap_pixels'] > 0:
                global_stats['overlap_count'] += 1

            # 保存
            image_name = image_path.name
            cv2.imwrite(str(images_dir / image_name), image)
            cv2.imwrite(str(masks_dir / image_name), mask)

            # 保存调试可视化
            if save_debug_vis and global_stats['success'] < 10:
                # 原图
                debug_img = image.copy()
                # mask叠加
                mask_vis = np.zeros_like(image)
                mask_vis[mask > 0] = [0, 255, 0]  # 绿色
                overlay = cv2.addWeighted(debug_img, 0.7, mask_vis, 0.3, 0)

                cv2.imwrite(str(debug_dir / f'{image_name[:-4]}_overlay.png'), overlay)
                cv2.imwrite(str(debug_dir / f'{image_name[:-4]}_mask.png'), mask * 85)

            global_stats['success'] += 1
            global_stats['total'] += 1

        except Exception as e:
            if debug:
                print(f"错误处理 {json_path}: {e}")
            global_stats['failed'] += 1
            global_stats['total'] += 1

    # 打印统计
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"总数: {global_stats['total']}")
    print(f"成功: {global_stats['success']}")
    print(f"失败: {global_stats['failed']}")

    if global_stats['curves_per_image']:
        print(f"\n每张图像的曲线数:")
        print(f"  平均: {np.mean(global_stats['curves_per_image']):.1f}")
        print(f"  最小: {np.min(global_stats['curves_per_image'])}")
        print(f"  最大: {np.max(global_stats['curves_per_image'])}")

    if global_stats['fg_ratios']:
        print(f"\n前景占比:")
        print(f"  平均: {np.mean(global_stats['fg_ratios'])*100:.2f}%")
        print(f"  最小: {np.min(global_stats['fg_ratios'])*100:.2f}%")
        print(f"  最大: {np.max(global_stats['fg_ratios'])*100:.2f}%")

    if global_stats['overlap_count'] > 0:
        print(f"\n有重叠的图像: {global_stats['overlap_count']} 张")

    print(f"\n输出:")
    print(f"  图像: {images_dir}")
    print(f"  Masks: {masks_dir}")
    if save_debug_vis:
        print(f"  调试可视化: {debug_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # 配置
    DATA_DIR = r"C:\Users\32665\Desktop\dataset_round1"
    OUTPUT_DIR = r"C:\Users\32665\Desktop\KM_2026.3.31\training_data"

    # 核心参数
    LINE_THICKNESS = 6  # 细线条，避免粘连
    DEBUG = True        # 输出统计信息
    SAVE_DEBUG_VIS = True  # 保存前10张的调试可视化

    process_dataset(
        DATA_DIR,
        OUTPUT_DIR,
        line_thickness=LINE_THICKNESS,
        debug=DEBUG,
        save_debug_vis=SAVE_DEBUG_VIS
    )
