# 曲线提取工具

从图片中提取带颜色的线，输出像素坐标轨迹。

## 功能

- 从图片提取彩色曲线
- 输出像素坐标CSV
- 支持多条曲线
- GUI和CLI双模式

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### CLI模式（推荐）

```bash
python cli.py --image path/to/image.png
```

参数：
- `--image, -i`: 输入图像路径（必需）
- `--outdir, -o`: 输出目录（默认：outputs）
- `--n-colors`: 颜色聚类数（默认：5）
- `--debug`: 启用调试模式

### GUI模式

```bash
python app.py
```

操作步骤：
1. 选择图像文件
2. 选择ROI模式（自动/手动/全图）
3. 点击"运行提取"
4. 查看结果

## 输出结果

输出目录包含：

- `roi_crop.png`: ROI区域
- `mask_color_*.png`: 各颜色mask
- `mask_component_*.png`: 各连通域mask
- `result_roi.png`: ROI局部结果
- `result_global.png`: 全图结果
- `curve_pixels_*.csv`: 各条曲线像素坐标
- `debug_overlay.png`: 调试叠加图
- `process.log`: 处理日志

## 技术路线

1. ROI检测
2. 深色像素提取
3. 颜色聚类（LAB空间 + KMeans）
4. 逐列追踪
5. 像素坐标输出

## 项目结构

```
KM_2026.3.31/
├── cli.py                  # CLI入口
├── app.py                  # GUI入口
├── requirements.txt
├── outputs/               # 输出目录
└── src/
    └── km_app/
        ├── config.py
        ├── pipeline/      # 处理管线
        │   ├── color_extract.py  # 主提取器
        │   ├── trace.py
        │   └── ...
        ├── gui/           # GUI界面
        └── io/            # 输入输出
```

## 注意事项

- 默认使用传统图像处理方法，不依赖深度学习模型
- 输入：RGB图像
- 输出：像素坐标（x, y）
- 适用于有明显颜色区分的曲线图

## 故障排除

### 提取不到曲线
- 检查图像是否包含明显的深色线条
- 尝试调整 `--n-colors` 参数
- 使用GUI手动框选ROI

### 曲线不完整
- 检查图像质量和分辨率
- 确保曲线在ROI范围内
- 查看 `process.log` 了解详细信息
