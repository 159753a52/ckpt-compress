"""
Inshrinkerator 分区模块。

实现三向参数分区:
- 保护: 高重要性参数以 bfloat16 存储
- 剪枝: 低重要性参数设为零
- 量化: 剩余参数进行非均匀量化
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .metrics import magnitude, sensitivity


def _stable_select(values: torch.Tensor, count: int, *, largest: bool) -> torch.Tensor:
    """Select exactly ``count`` flattened entries with stable tie handling."""
    if not 0 <= count <= values.numel():
        raise ValueError("selection count must fit the tensor")
    selected = torch.zeros(values.numel(), dtype=torch.bool, device=values.device)
    if count:
        order = torch.argsort(values.flatten(), descending=largest, stable=True)
        selected[order[:count]] = True
    return selected


@dataclass
class PartitionConfig:
    """参数分区配置。"""

    protect_fraction: float = 0.001  # 保护比例 (0.1%)
    prune_fraction: float = 0.2  # 剪枝比例 (20%)
    n_bins: int = 8  # 量化桶数
    use_sensitivity: bool = True  # 使用敏感度指标

    def __post_init__(self) -> None:
        for name, value in (
            ("protect_fraction", self.protect_fraction),
            ("prune_fraction", self.prune_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
        if self.protect_fraction + self.prune_fraction > 1.0:
            raise ValueError("protect_fraction + prune_fraction must not exceed 1")
        if not isinstance(self.n_bins, int) or isinstance(self.n_bins, bool) or self.n_bins < 1:
            raise ValueError(f"n_bins must be a positive integer, got {self.n_bins}")
        if not isinstance(self.use_sensitivity, bool):
            raise TypeError("use_sensitivity must be a boolean")


@dataclass
class PartitionResult:
    """参数分区结果。"""

    protect_mask: torch.Tensor  # 1 = 保护, 0 = 不保护
    prune_mask: torch.Tensor  # 1 = 剪枝, 0 = 不剪枝
    quantize_mask: torch.Tensor  # 1 = 量化, 0 = 不量化

    def get_protected_values(self, values: torch.Tensor) -> torch.Tensor:
        """获取 bfloat16 格式的受保护值。"""
        protected = values[self.protect_mask == 1]
        return protected.to(torch.bfloat16)

    def get_pruned_values(self, values: torch.Tensor) -> torch.Tensor:
        """获取剪枝位置设为零的值。"""
        result = values.clone()
        result[self.prune_mask == 1] = 0.0
        return result

    def get_quantize_values(self, values: torch.Tensor) -> torch.Tensor:
        """获取要量化的值。"""
        return values[self.quantize_mask == 1]


def compute_thresholds(
    mag: torch.Tensor, sens: torch.Tensor, protect_fraction: float, prune_fraction: float
) -> tuple:
    """
    计算保护和剪枝阈值。

    参数:
        mag: 幅度值
        sens: 敏感度值
        protect_fraction: 保护比例
        prune_fraction: 剪枝比例

    返回:
        (protect_threshold_mag, protect_threshold_sens, prune_threshold) 元组
    """
    n = mag.numel()

    if n == 0:
        return 0.0, float("inf"), 0.0

    # 保护阈值: 按任一指标取前 protect_fraction
    protect_k = max(1, int(n * protect_fraction))
    if protect_k < n:
        protect_thresh_mag = torch.kthvalue(mag.flatten(), n - protect_k + 1).values.item()
        # 对于敏感度，如果所有值都为零（无敏感度信息），使用 inf
        if sens.max() > 0:
            protect_thresh_sens = torch.kthvalue(sens.flatten(), n - protect_k + 1).values.item()
        else:
            protect_thresh_sens = float("inf")  # 不基于敏感度保护
    else:
        protect_thresh_mag = mag.min().item()
        protect_thresh_sens = float("inf") if sens.max() == 0 else sens.min().item()

    # 剪枝阈值: 按幅度取后 prune_fraction
    prune_k = max(1, int(n * prune_fraction))
    if prune_k < n and prune_fraction > 0:
        prune_thresh = torch.kthvalue(mag.flatten(), prune_k).values.item()
    else:
        prune_thresh = 0.0

    return protect_thresh_mag, protect_thresh_sens, prune_thresh


def partition(values: torch.Tensor, grad: torch.Tensor, config: PartitionConfig) -> PartitionResult:
    """
    将参数分区为保护/剪枝/量化集合。

    参数:
        values: 参数值
        grad: 梯度值（用于敏感度）
        config: 分区配置

    返回:
        包含每个分区掩码的 PartitionResult
    """
    if not isinstance(values, torch.Tensor) or not isinstance(grad, torch.Tensor):
        raise TypeError("values and grad must be torch.Tensor objects")
    n = values.numel()
    if values.shape != grad.shape:
        raise ValueError("values and grad must have the same shape")
    if values.device != grad.device:
        raise ValueError("values and grad must be on the same device")
    if not values.is_floating_point() or not grad.is_floating_point():
        raise TypeError("values and grad must be floating-point tensors")
    if not torch.isfinite(values).all().item() or not torch.isfinite(grad).all().item():
        raise ValueError("values and grad must contain only finite values")

    if n == 0:
        empty = torch.zeros(0, dtype=torch.long, device=values.device)
        return PartitionResult(
            protect_mask=empty,
            prune_mask=empty,
            quantize_mask=empty,
        )

    # 计算重要性指标
    mag = magnitude(values)
    sens = sensitivity(values, grad) if config.use_sensitivity else mag

    # 展平以构造稳定的精确计数分区。
    mag_flat = mag.flatten()
    sens_flat = sens.flatten()

    # 创建掩码
    protect_selected = torch.zeros(n, dtype=torch.bool, device=values.device)
    prune_selected = torch.zeros(n, dtype=torch.bool, device=values.device)

    # 保护: 高幅度或高敏感度
    if config.protect_fraction > 0:
        protect_count = max(1, int(n * config.protect_fraction))
        protect_selected = _stable_select(mag_flat, protect_count, largest=True)
        if config.use_sensitivity and sens_flat.max().item() > 0:
            protect_selected |= _stable_select(sens_flat, protect_count, largest=True)

    # 剪枝: 低幅度（但未被保护）
    if config.prune_fraction > 0:
        prune_count = n if config.prune_fraction == 1.0 else max(1, int(n * config.prune_fraction))
        available = torch.nonzero(~protect_selected, as_tuple=False).flatten()
        if available.numel() < prune_count:
            raise ValueError("Protected parameters leave too little capacity for prune_fraction")
        available_order = torch.argsort(mag_flat[available], stable=True)
        prune_selected[available[available_order[:prune_count]]] = True

    # 量化: 其他所有
    quantize_selected = ~protect_selected & ~prune_selected

    # 将掩码重塑为原始形状
    original_shape = values.shape
    return PartitionResult(
        protect_mask=protect_selected.long().reshape(original_shape),
        prune_mask=prune_selected.long().reshape(original_shape),
        quantize_mask=quantize_selected.long().reshape(original_shape),
    )
