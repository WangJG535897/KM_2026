"""全局配置"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 默认路径 - 使用最新best checkpoint
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model_binary.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 模型配置
MODEL_INPUT_SIZE = (512, 512)  # 推理时resize大小
MODEL_INPUT_CHANNELS = 3

# 处理配置
DEFAULT_X_MAX = 48.0  # 默认X轴最大时间
DEFAULT_Y_RANGE = (0.0, 100.0)  # Y轴范围（生存率百分比）

# 调试配置
DEBUG_MODE = True
SAVE_INTERMEDIATE = True
