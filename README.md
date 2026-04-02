# KM生存曲线提取工具

从Kaplan-Meier生存曲线图像中自动提取曲线数据的完整工具。

## 功能特性

- 基于深度学习的曲线分割
- 自动ROI检测
- 多曲线追踪和分离
- KM先验约束（单调性、台阶形状等）
- 像素坐标到图表坐标映射
- CSV数据导出
- 图形界面和命令行双模式

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 注意：opencv-contrib-python包含ximgproc模块
pip install opencv-contrib-python
```

## 使用方法

### GUI模式

```bash
python app.py
```

操作步骤：
1. 选择KM曲线图像
2. 选择模型权重文件（默认：models/best_model.pth）
3. 设置X轴最大时间
4. 点击"运行提取"
5. 查看结果并导出

### CLI模式

```bash
python cli.py --image path/to/image.png --x-max 48
```

参数说明：
- `--image, -i`: 输入图像路径（必需）
- `--checkpoint, -c`: 模型权重路径（默认：models/best_model.pth）
- `--outdir, -o`: 输出目录（默认：outputs）
- `--x-max`: X轴最大时间（默认：48）
- `--debug`: 启用调试模式，保存中间结果

## 输出结果

处理完成后，输出目录包含：

- `original.png`: 原始图像
- `roi.png`: ROI区域
- `result.png`: 曲线叠加结果
- `curve_1.csv`, `curve_2.csv`, ...: 各条曲线坐标
- `all_curves.csv`: 所有曲线合并
- `mask_*.png`: 各类别mask（调试模式）
- `process.log`: 处理日志（调试模式）

## 项目结构

```
KM_2026.3.31/
├── app.py                  # GUI启动入口
├── cli.py                  # CLI启动入口
├── requirements.txt        # 依赖列表
├── models/
│   └── best_model.pth     # 训练好的模型
├── outputs/               # 输出目录
└── src/
    └── km_app/
        ├── config.py      # 配置
        ├── model/         # 模型相关
        ├── pipeline/      # 处理管线
        ├── gui/           # GUI界面
        ├── io/            # 输入输出
        └── utils/         # 工具函数
```

## 技术路线

1. **图像预处理**: 自动ROI检测、归一化
2. **模型推理**: ResNet风格encoder-decoder分割网络
3. **后处理**:
   - 颜色精修和曲线分离
   - 骨架提取和路径追踪
   - KM先验约束（单调性、台阶形状）
4. **坐标映射**: 像素坐标→图表坐标
5. **结果导出**: CSV格式

## 注意事项

- 模型输入：3通道RGB图像
- 模型输出：6类（背景+5条曲线）
- 推理时会resize到512x512，但最终结果映射回原图
- KM约束确保曲线符合生存分析规律

## 故障排除

### 模型加载失败
- 检查模型文件路径是否正确
- 确认模型文件完整未损坏

### 曲线提取不准确
- 调整X轴最大时间参数
- 检查图像质量和分辨率
- 确保图像包含完整的KM曲线

### GUI无法启动
- 确认已安装PySide6
- 检查Python版本（建议3.8+）
