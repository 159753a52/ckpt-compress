"""参数名解析注册表：统一的 param_name → layer_type 映射。

新增模型只需调用 register_type_rule() 注册一个解析函数即可。
"""

import torch
from typing import Dict, Callable, List

# 注册表：model_family -> parse_fn(name, tensor) -> layer_type
_TYPE_RULES: Dict[str, Callable[[str, torch.Tensor], str]] = {}


def register_type_rule(family: str, fn: Callable[[str, torch.Tensor], str]):
    """注册模型族的参数类型解析规则。"""
    _TYPE_RULES[family] = fn


def infer_layer_type(name: str, tensor: torch.Tensor, model_family: str) -> str:
    """根据参数名和张量推断 layer type。

    返回值:
        'attn' | 'mlp' | 'conv' | 'fc' | 'others_linear' | 'skip'
        'skip' 表示不可剪枝（embedding / norm / bias / 1D）。
    """
    if model_family not in _TYPE_RULES:
        raise ValueError(
            f"No type rules for model family: {model_family}. "
            f"Register with register_type_rule(). Available: {list(_TYPE_RULES.keys())}"
        )
    return _TYPE_RULES[model_family](name, tensor)


def build_type_map(
    named_params: Dict[str, torch.Tensor], model_family: str
) -> Dict[str, str]:
    """为所有参数构建 {name: layer_type} 映射。"""
    return {
        name: infer_layer_type(name, tensor, model_family)
        for name, tensor in named_params.items()
    }


def get_prunable_types(model_family: str) -> List[str]:
    """返回该模型族中可剪枝的 type 列表。"""
    mapping = {
        'gpt2': ['attn', 'mlp', 'others_linear'],
        'bert': ['attn', 'mlp', 'others_linear'],
        'resnet': ['conv', 'fc'],
    }
    return mapping.get(model_family, ['attn', 'mlp', 'others_linear'])


def list_registered_families() -> List[str]:
    return list(_TYPE_RULES.keys())


# ============================================================
# 内置规则
# ============================================================

def _gpt2_type_rule(name: str, tensor: torch.Tensor) -> str:
    if tensor.dim() < 2:
        return 'skip'
    skip_patterns = ['ln_', 'bias', 'wte', 'wpe']
    if any(p in name for p in skip_patterns):
        return 'skip'
    if 'attn' in name:
        return 'attn'
    if 'mlp' in name:
        return 'mlp'
    return 'others_linear'


def _bert_type_rule(name: str, tensor: torch.Tensor) -> str:
    if tensor.dim() < 2:
        return 'skip'
    skip_patterns = ['LayerNorm', 'layernorm', 'bias', 'embeddings', 'pooler', 'classifier']
    if any(p in name for p in skip_patterns):
        return 'skip'
    if 'attention' in name:
        return 'attn'
    if 'intermediate' in name or 'output.dense' in name:
        return 'mlp'
    return 'others_linear'


def _resnet_type_rule(name: str, tensor: torch.Tensor) -> str:
    if tensor.dim() < 2:
        return 'skip'
    skip_patterns = ['bn', 'bias', 'downsample.1']
    if any(p in name for p in skip_patterns):
        return 'skip'
    if 'conv' in name or 'downsample.0' in name:
        return 'conv'
    if 'fc' in name:
        return 'fc'
    return 'skip'


register_type_rule('gpt2', _gpt2_type_rule)
register_type_rule('bert', _bert_type_rule)
register_type_rule('resnet', _resnet_type_rule)
