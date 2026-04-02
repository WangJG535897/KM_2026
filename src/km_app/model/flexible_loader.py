"""简化的模型包装器 - 直接加载权重"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlexibleModel(nn.Module):
    """灵活的模型包装器 - 直接从checkpoint加载"""

    def __init__(self, state_dict):
        super().__init__()
        # 直接使用state_dict创建参数
        for name, param in state_dict.items():
            # 将点号替换为模块层次
            parts = name.split('.')
            self._create_param_path(parts, param)

    def _create_param_path(self, parts, param):
        """递归创建参数路径"""
        if len(parts) == 1:
            setattr(self, parts[0], nn.Parameter(param))
        else:
            if not hasattr(self, parts[0]):
                setattr(self, parts[0], nn.Module())
            module = getattr(self, parts[0])
            self._add_to_module(module, parts[1:], param)

    def _add_to_module(self, module, parts, param):
        """添加参数到模块"""
        if len(parts) == 1:
            module.register_parameter(parts[0], nn.Parameter(param))
        else:
            if not hasattr(module, parts[0]):
                setattr(module, parts[0], nn.Module())
            next_module = getattr(module, parts[0])
            self._add_to_module(next_module, parts[1:], param)


def load_model_flexible(checkpoint_path: str, device='cpu'):
    """灵活加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 创建一个简单的forward包装
    class SimpleWrapper(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            self.state_dict_cache = state_dict
            # 尝试从state_dict重建模型
            from .model_adapter import KMSegmentationModel
            self.model = KMSegmentationModel(in_channels=3, num_classes=6)
            self.model.load_state_dict(state_dict, strict=False)

        def forward(self, x):
            return self.model(x)

    model = SimpleWrapper(state_dict)
    model.to(device)
    model.eval()

    return model
