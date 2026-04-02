"""模型检查点检查器"""
import torch
from pathlib import Path
from typing import Dict, Any, Optional


class CheckpointInspector:
    """检查和分析模型checkpoint"""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = None
        self.state_dict = None
        self.metadata = {}

    def load(self) -> Dict[str, Any]:
        """加载checkpoint"""
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        # 提取state_dict
        if isinstance(self.checkpoint, dict):
            if 'model_state_dict' in self.checkpoint:
                self.state_dict = self.checkpoint['model_state_dict']
                self.metadata = {k: v for k, v in self.checkpoint.items()
                               if k != 'model_state_dict'}
            elif 'state_dict' in self.checkpoint:
                self.state_dict = self.checkpoint['state_dict']
                self.metadata = {k: v for k, v in self.checkpoint.items()
                               if k != 'state_dict'}
            else:
                self.state_dict = self.checkpoint
        else:
            self.state_dict = self.checkpoint

        return self.state_dict

    def get_input_shape(self) -> Optional[tuple]:
        """推断输入形状"""
        # 查找第一个卷积层
        for key, tensor in self.state_dict.items():
            if 'encoder' in key.lower() and 'weight' in key and len(tensor.shape) == 4:
                # Conv2d weight shape: [out_channels, in_channels, kernel_h, kernel_w]
                return (tensor.shape[1],)  # 返回输入通道数
        return None

    def get_output_shape(self) -> Optional[tuple]:
        """推断输出形状"""
        # 查找最后的卷积层
        for key in ['final_conv.weight', 'outc.weight', 'classifier.weight']:
            if key in self.state_dict:
                tensor = self.state_dict[key]
                if len(tensor.shape) == 4:
                    return (tensor.shape[0],)  # 返回输出通道数
        return None

    def print_summary(self):
        """打印摘要信息"""
        print("=" * 60)
        print("Checkpoint Summary")
        print("=" * 60)
        print(f"Path: {self.checkpoint_path}")
        print(f"Metadata: {self.metadata}")

        input_shape = self.get_input_shape()
        output_shape = self.get_output_shape()

        if input_shape:
            print(f"Input channels: {input_shape[0]}")
        if output_shape:
            print(f"Output classes: {output_shape[0]}")

        print(f"\nTotal parameters: {len(self.state_dict)}")
        print("\nFirst 5 layers:")
        for i, (k, v) in enumerate(list(self.state_dict.items())[:5]):
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")

        print("\nLast 5 layers:")
        for k, v in list(self.state_dict.items())[-5:]:
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")
        print("=" * 60)
