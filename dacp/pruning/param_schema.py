"""参数名解析注册表：统一的 param_name → layer_type 映射。

新增模型只需调用 register_type_rule() 注册解析函数和可剪枝类型即可。
"""

import torch
from typing import Callable, Dict, List, NamedTuple, Sequence, Tuple


TypeRule = Callable[[str, torch.Tensor], str]


class _TypeSchema(NamedTuple):
    rule: TypeRule
    prunable_types: Tuple[str, ...]


_TYPE_SCHEMAS: Dict[str, _TypeSchema] = {}


def _get_schema(model_family: str) -> _TypeSchema:
    try:
        return _TYPE_SCHEMAS[model_family]
    except KeyError as exc:
        raise ValueError(
            f"No type schema for model family: {model_family}. "
            "Register with register_type_rule(). "
            f"Available: {list(_TYPE_SCHEMAS)}"
        ) from exc


def register_type_rule(
    family: str,
    fn: TypeRule,
    prunable_types: Sequence[str],
) -> None:
    """Register one model family's classifier and searchable layer types."""
    if not isinstance(family, str) or not family.strip():
        raise ValueError("family must be a non-empty string")
    if not callable(fn):
        raise TypeError("fn must be callable")
    normalized_types = tuple(prunable_types)
    if not normalized_types or any(
        not isinstance(layer_type, str) or not layer_type
        for layer_type in normalized_types
    ):
        raise ValueError("prunable_types must contain non-empty strings")
    if "skip" in normalized_types:
        raise ValueError("prunable_types must not contain the reserved 'skip' type")
    if len(set(normalized_types)) != len(normalized_types):
        raise ValueError("prunable_types must not contain duplicates")
    _TYPE_SCHEMAS[family] = _TypeSchema(fn, normalized_types)


def infer_layer_type(name: str, tensor: torch.Tensor, model_family: str) -> str:
    """根据参数名和张量推断 layer type。

    返回值:
        'attn' | 'mlp' | 'conv' | 'fc' | 'others_linear' | 'skip'
        'skip' 表示不可剪枝（embedding / norm / bias / 1D）。
    """
    schema = _get_schema(model_family)
    layer_type = schema.rule(name, tensor)
    if layer_type != 'skip' and layer_type not in schema.prunable_types:
        raise ValueError(
            f"Type rule for {model_family!r} returned unregistered type "
            f"{layer_type!r} for parameter {name!r}"
        )
    return layer_type


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
    return list(_get_schema(model_family).prunable_types)


def list_registered_families() -> List[str]:
    return list(_TYPE_SCHEMAS)


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


def _pythia_type_rule(name: str, tensor: torch.Tensor) -> str:
    if tensor.dim() < 2:
        return 'skip'
    skip_patterns = ['layernorm', 'layer_norm', 'bias', 'embed_in', 'embed_out']
    if any(p in name.lower() for p in skip_patterns):
        return 'skip'
    if 'attention' in name.lower():
        return 'attn'
    if 'mlp' in name.lower() or 'dense' in name.lower():
        return 'mlp'
    return 'others_linear'


def _vit_type_rule(name: str, tensor: torch.Tensor) -> str:
    if tensor.dim() < 2:
        return 'skip'
    skip_patterns = ['layernorm', 'layer_norm', 'bias', 'cls_token',
                     'position_embedding', 'patch_embed', 'embeddings']
    if any(p in name.lower() for p in skip_patterns):
        return 'skip'
    if 'attention' in name.lower():
        return 'attn'
    if 'intermediate' in name.lower() or 'output.dense' in name.lower() or 'mlp' in name.lower():
        return 'mlp'
    if 'classifier' in name.lower():
        return 'fc'
    return 'others_linear'


register_type_rule(
    'gpt2',
    _gpt2_type_rule,
    ('attn', 'mlp', 'others_linear'),
)
register_type_rule(
    'bert',
    _bert_type_rule,
    ('attn', 'mlp', 'others_linear'),
)
register_type_rule('resnet', _resnet_type_rule, ('conv', 'fc'))
register_type_rule(
    'pythia',
    _pythia_type_rule,
    ('attn', 'mlp', 'others_linear'),
)
register_type_rule(
    'vit',
    _vit_type_rule,
    ('attn', 'mlp', 'fc', 'others_linear'),
)
