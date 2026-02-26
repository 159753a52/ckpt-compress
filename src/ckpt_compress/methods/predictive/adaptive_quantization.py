"""
预测残差压缩 - 自适应量化模块。

实现敏感度感知量化:
- 高敏感度参数获得更多位数（更低误差）
- 低敏感度参数获得更少位数（更高压缩率）

核心思想: 并非所有参数对训练同等重要。
通过基于敏感度分配位数，我们可以在保持训练质量的同时
实现更好的压缩。
"""

import torch
from typing import Dict, Tuple, Any
from dataclasses import dataclass


def compute_sensitivity(
    weight: torch.Tensor,
    grad: torch.Tensor
) -> torch.Tensor:
    """
    计算参数敏感度。

    敏感度 = |grad| * |weight|

    更高的敏感度意味着参数对损失有更大影响。

    参数:
        weight: 权重张量
        grad: 梯度张量

    返回:
        敏感度张量（与输入形状相同）
    """
    return torch.abs(grad) * torch.abs(weight)


def assign_bit_allocation(
    sensitivity: torch.Tensor,
    min_bits: int = 2,
    max_bits: int = 8
) -> torch.Tensor:
    """
    基于敏感度分配位数。

    更高敏感度 -> 更多位数 -> 更低量化误差。

    参数:
        sensitivity: 敏感度张量
        min_bits: 分配的最小位数
        max_bits: 分配的最大位数

    返回:
        位数分配张量（整数）
    """
    # 将敏感度归一化到 [0, 1]
    s_min = sensitivity.min()
    s_max = sensitivity.max()

    if s_max - s_min < 1e-8:
        # 均匀敏感度，使用中间位数
        mid_bits = (min_bits + max_bits) // 2
        return torch.full_like(sensitivity, mid_bits, dtype=torch.int32)

    normalized = (sensitivity - s_min) / (s_max - s_min)

    # 映射到位数范围
    bits = min_bits + normalized * (max_bits - min_bits)
    bits = torch.round(bits).to(torch.int32)

    # 限制到有效范围
    bits = torch.clamp(bits, min_bits, max_bits)

    return bits


def adaptive_quantize(
    values: torch.Tensor,
    n_bits: int
) -> Tuple[torch.Tensor, float, float]:
    """
    将值量化到 n_bits 精度。

    使用对称量化: q = round(v / scale)

    参数:
        values: 要量化的值
        n_bits: 位数

    返回:
        (indices, scale, zero_point) 元组
    """
    n_levels = 2 ** n_bits
    v_min = values.min().item()
    v_max = values.max().item()

    # 对称量化
    v_abs_max = max(abs(v_min), abs(v_max))
    if v_abs_max < 1e-8:
        # 全零
        return torch.zeros_like(values, dtype=torch.int32), 1.0, 0.0

    scale = v_abs_max / (n_levels // 2 - 1)
    zero_point = n_levels // 2

    # 量化
    indices = torch.round(values / scale).to(torch.int32) + zero_point
    indices = torch.clamp(indices, 0, n_levels - 1)

    return indices, scale, zero_point


def adaptive_dequantize(
    indices: torch.Tensor,
    scale: float,
    zero_point: float
) -> torch.Tensor:
    """
    将索引反量化回值。

    参数:
        indices: 量化索引
        scale: 量化缩放因子
        zero_point: 零点

    返回:
        反量化的值
    """
    return (indices.float() - zero_point) * scale


@dataclass
class QuantizationResult:
    """量化结果。"""
    indices: torch.Tensor
    scale: float
    zero_point: float
    bits: int
    shape: Tuple[int, ...]


class AdaptiveQuantizer:
    """
    具有敏感度感知位数分配的自适应量化器。

    为高敏感度参数分配更多位数，
    以更好地保持训练动态。
    """

    def __init__(
        self,
        min_bits: int = 2,
        max_bits: int = 8,
        use_per_tensor_bits: bool = True
    ):
        """
        初始化自适应量化器。

        参数:
            min_bits: 低敏感度参数的最小位数
            max_bits: 高敏感度参数的最大位数
            use_per_tensor_bits: 如果为 True，每个张量使用单一位数分配
                                 如果为 False，使用逐元素位数分配
        """
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.use_per_tensor_bits = use_per_tensor_bits

    def quantize(
        self,
        tensor: torch.Tensor,
        grad: torch.Tensor
    ) -> Dict[str, Any]:
        """
        使用敏感度感知位数分配量化张量。

        参数:
            tensor: 要量化的张量
            grad: 用于敏感度的梯度张量

        返回:
            包含量化数据的字典
        """
        # 计算敏感度
        sensitivity = compute_sensitivity(tensor, grad)

        if self.use_per_tensor_bits:
            # 对整个张量使用平均敏感度
            avg_sensitivity = sensitivity.mean()
            bits = assign_bit_allocation(
                avg_sensitivity.unsqueeze(0),
                self.min_bits,
                self.max_bits
            )[0].item()

            indices, scale, zero_point = adaptive_quantize(tensor.flatten(), bits)

            return {
                "indices": indices,
                "scale": scale,
                "zero_point": zero_point,
                "bits": bits,
                "shape": tensor.shape,
            }
        else:
            # 逐元素位数分配（更复杂，质量更好）
            bits = assign_bit_allocation(sensitivity, self.min_bits, self.max_bits)

            # 按位数分配分组以高效存储
            unique_bits = torch.unique(bits)
            indices_list = []
            scales = {}
            zero_points = {}

            flat_tensor = tensor.flatten()
            flat_bits = bits.flatten()

            all_indices = torch.zeros_like(flat_tensor, dtype=torch.int32)

            for b in unique_bits:
                b_val = b.item()
                mask = (flat_bits == b_val)
                values = flat_tensor[mask]

                if values.numel() > 0:
                    q_indices, scale, zp = adaptive_quantize(values, b_val)
                    all_indices[mask] = q_indices
                    scales[b_val] = scale
                    zero_points[b_val] = zp

            return {
                "indices": all_indices,
                "bits": flat_bits,
                "scales": scales,
                "zero_points": zero_points,
                "shape": tensor.shape,
            }

    def dequantize(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        将数据反量化回张量。

        参数:
            data: 来自 quantize() 的量化数据

        返回:
            重建的张量
        """
        shape = data["shape"]

        if "scale" in data:
            # 逐张量量化
            indices = data["indices"]
            scale = data["scale"]
            zero_point = data["zero_point"]

            values = adaptive_dequantize(indices, scale, zero_point)
            return values.reshape(shape)
        else:
            # 逐元素量化
            indices = data["indices"]
            bits = data["bits"]
            scales = data["scales"]
            zero_points = data["zero_points"]

            values = torch.zeros_like(indices, dtype=torch.float32)
            unique_bits = torch.unique(bits)

            for b in unique_bits:
                b_val = b.item()
                mask = (bits == b_val)
                if b_val in scales:
                    values[mask] = adaptive_dequantize(
                        indices[mask],
                        scales[b_val],
                        zero_points[b_val]
                    )

            return values.reshape(shape)

    def quantize_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        grad_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, Any]]:
        """
        量化状态字典。

        参数:
            state_dict: 要量化的状态字典
            grad_dict: 用于敏感度的梯度字典

        返回:
            每个键的量化数据字典
        """
        result = {}
        for key in state_dict:
            tensor = state_dict[key]
            grad = grad_dict.get(key, torch.zeros_like(tensor))
            result[key] = self.quantize(tensor, grad)
        return result

    def dequantize_state_dict(
        self,
        compressed: Dict[str, Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        反量化压缩的状态字典。

        参数:
            compressed: 压缩的状态字典

        返回:
            重建的状态字典
        """
        result = {}
        for key in compressed:
            result[key] = self.dequantize(compressed[key])
        return result
