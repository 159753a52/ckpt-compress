"""
检查点压缩的张量操作。

提供状态字典展平和还原的工具函数。
"""

import torch
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass


@dataclass
class TensorSpec:
    """状态字典中张量的规格说明。"""
    key: str
    shape: torch.Size
    dtype: torch.dtype
    numel: int
    offset: int


@dataclass
class FlattenSpec:
    """状态字典展平/还原的规格说明。"""
    tensor_specs: List[TensorSpec]
    non_tensor_items: Dict[str, Any]
    total_numel: int


def flatten_state_dict(
    state_dict: Dict[str, Any]
) -> Tuple[torch.Tensor, FlattenSpec]:
    """
    将状态字典展平为单个一维张量。

    参数:
        state_dict: 参数名称到张量或其他值的映射字典。

    返回:
        一个元组 (flat_vector, spec)，其中:
        - flat_vector: 包含所有张量值拼接后的一维张量
        - spec: 包含还原所需信息的 FlattenSpec
    """
    tensor_specs = []
    non_tensor_items = {}
    tensors_to_concat = []
    offset = 0

    # 按排序顺序处理项目以确保确定性行为
    for key in sorted(state_dict.keys()):
        value = state_dict[key]

        if isinstance(value, torch.Tensor):
            numel = value.numel()
            tensor_specs.append(TensorSpec(
                key=key,
                shape=value.shape,
                dtype=value.dtype,
                numel=numel,
                offset=offset,
            ))
            if numel > 0:
                # 转换为 float32 进行拼接，原始 dtype 存储在 spec 中
                tensors_to_concat.append(value.flatten().float())
            offset += numel
        else:
            # 单独存储非张量项
            non_tensor_items[key] = value

    # 创建展平向量
    if tensors_to_concat:
        flat_vector = torch.cat(tensors_to_concat)
    else:
        flat_vector = torch.tensor([], dtype=torch.float32)

    spec = FlattenSpec(
        tensor_specs=tensor_specs,
        non_tensor_items=non_tensor_items,
        total_numel=offset,
    )

    return flat_vector, spec


def unflatten_state_dict(
    flat_vector: torch.Tensor,
    spec: FlattenSpec
) -> Dict[str, Any]:
    """
    将一维张量还原为状态字典。

    参数:
        flat_vector: 包含所有张量值拼接后的一维张量。
        spec: 包含还原所需信息的 FlattenSpec。

    返回:
        参数名称到张量的映射字典。

    异常:
        ValueError: 如果 flat_vector 长度与 spec 不匹配。
    """
    if flat_vector.numel() != spec.total_numel:
        raise ValueError(
            f"Vector length {flat_vector.numel()} doesn't match spec "
            f"total_numel {spec.total_numel}"
        )

    result = {}

    # 还原张量
    for tensor_spec in spec.tensor_specs:
        if tensor_spec.numel > 0:
            start = tensor_spec.offset
            end = start + tensor_spec.numel
            tensor_data = flat_vector[start:end]
            # 重塑并转换回原始 dtype
            tensor = tensor_data.reshape(tensor_spec.shape).to(tensor_spec.dtype)
        else:
            # 处理空张量
            tensor = torch.tensor([], dtype=tensor_spec.dtype).reshape(tensor_spec.shape)

        result[tensor_spec.key] = tensor

    # 还原非张量项
    result.update(spec.non_tensor_items)

    return result
