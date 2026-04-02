# 安装和运行指南

## 环境要求

- Python 3.8+
- Windows 10/11
- 建议使用虚拟环境

## 安装步骤

### 1. 创建虚拟环境（推荐）

```bash
cd C:\Users\32665\Desktop\KM_2026.3.31
python -m venv venv
venv\Scripts\activate
```

### 2. 安装依赖

```bash
# 基础依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 其他依赖
pip install opencv-contrib-python numpy Pillow scikit-image scikit-learn scipy pandas matplotlib

# GUI依赖
pip install PySide6
```

或者直接：

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python test_quick.py
```

如果看到"所有测试通过"，说明安装成功。

## 运行

### GUI模式

```bash
python app.py
```

### CLI模式

```bash
# 基本用法
python cli.py --image path/to/km_curve.png

# 指定参数
python cli.py --image path/to/km_curve.png --x-max 60 --outdir results

# 调试模式
python cli.py --image path/to/km_curve.png --debug
```

## 常见问题

### Q1: 提示缺少opencv-contrib-python

**解决方案**:
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python
```

### Q2: PySide6安装失败

**解决方案**:
```bash
pip install PySide6 --upgrade
```

如果仍然失败，可以只使用CLI模式（不需要PySide6）。

### Q3: CUDA不可用

**解决方案**:
程序会自动使用CPU，无需担心。如果需要GPU加速：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q4: 模型加载失败

**检查**:
- 确认 `models/best_model.pth` 文件存在
- 运行 `python test_quick.py` 查看详细错误

## 目录说明

```
KM_2026.3.31/
├── app.py              # GUI启动（需要PySide6）
├── cli.py              # CLI启动（推荐先用这个测试）
├── test_quick.py       # 快速测试脚本
├── requirements.txt    # 依赖列表
├── README.md          # 使用说明
├── INSTALL.md         # 本文件
├── models/
│   └── best_model.pth # 你的模型权重
├── outputs/           # 输出目录（自动创建）
└── src/
    └── km_app/        # 核心代码
```

## 下一步

1. 运行 `python test_quick.py` 验证环境
2. 准备一张KM曲线图像
3. 运行 `python cli.py --image your_image.png`
4. 查看 `outputs/` 目录中的结果
