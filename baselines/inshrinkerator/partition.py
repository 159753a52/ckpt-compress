"""
Inshrinkerator 分区模块。

实现三向参数分区:
- 保护: 高重要性参数以 bfloat16 存储
- 剪枝: 低重要性参数设为零
- 量化: 剩余参数进行非均匀量化
"""

import torch
from typing import Optional
from dataclasses import dataclass

from .metrics import magnitude, sensitivity


@dataclass
class PartitionConfig:
    """参数分区配置。"""
    protect_fraction: float = 0.001  # 保护比例 (0.1%)
    prune_fraction: float = 0.2      # 剪枝比例 (20%)
    n_bins: int = 8                  # 量化桶数
    use_sensitivity: bool = True     # 使用敏感度指标


@dataclass
class PartitionResult:
    """参数分区结果。"""
    protect_mask: torch.Tensor   # 1 = 保护, 0 = 不保护
    prune_mask: torch.Tensor     # 1 = 剪枝, 0 = 不剪枝
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
    mag: torch.Tensor,
    sens: torch.Tensor,
    protect_fraction: float,
    prune_fraction: float
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
        return 0.0, float('inf'), 0.0

    # 保护阈值: 按任一指标取前 protect_fraction
    protect_k = max(1, int(n * protect_fraction))
    if protect_k < n:
        protect_thresh_mag = torch.kthvalue(mag.flatten(), n - protect_k + 1).values.item()
        # 对于敏感度，如果所有值都为零（无敏感度信息），使用 inf
        if sens.max() > 0:
            protect_thresh_sens = torch.kthvalue(sens.flatten(), n - protect_k + 1).values.item()
        else:
            protect_thresh_sens = float('inf')  # 不基于敏感度保护
    else:
        protect_thresh_mag = mag.min().item()
        protect_thresh_sens = float('inf') if sens.max() == 0 else sens.min().item()

    # 剪枝阈值: 按幅度取后 prune_fraction
    prune_k = max(1, int(n * prune_fraction))
    if prune_k < n and prune_fraction > 0:
        prune_thresh = torch.kthvalue(mag.flatten(), prune_k).values.item()
    else:
        prune_thresh = 0.0

    return protect_thresh_mag, protect_thresh_sens, prune_thresh


def partition(
    values: torch.Tensor,
    grad: torch.Tensor,
    config: PartitionConfig
) -> PartitionResult:
    """
    将参数分区为保护/剪枝/量化集合。

    参数:
        values: 参数值
        grad: 梯度值（用于敏感度）
        config: 分区配置

    返回:
        包含每个分区掩码的 PartitionResult
    """
    n = values.numel()

    if n == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return PartitionResult(
            protect_mask=empty,
            prune_mask=empty,
            quantize_mask=empty,
        )

    # 计算重要性指标
    mag = magnitude(values)
    sens = sensitivity(values, grad) if config.use_sensitivity else mag

    # 展平以计算阈值
    mag_flat = mag.flatten()
    sens_flat = sens.flatten()

    # 计算阈值
    protect_thresh_mag, protect_thresh_sens, prune_thresh = compute_thresholds(
        mag_flat, sens_flat, config.protect_fraction, config.prune_fraction
    )

    # 创建掩码
    protect_mask = torch.zeros(n, dtype=torch.long)
    prune_mask = torch.zeros(n, dtype=torch.long)

    # 保护: 高幅度或高敏感度
    if config.protect_fraction > 0:
        protect_by_mag = mag_flat >= protect_thresh_mag
        protect_by_sens = sens_flat >= protect_thresh_sens
        protect_mask = (protect_by_mag | protect_by_sens).long()

    # 剪枝: 低幅度（但未被保护）
    if config.prune_fraction > 0:
        prune_candidates = mag_flat <= prune_thresh
        prune_mask = (prune_candidates & (protect_mask == 0)).long()

    # 量化: 其他所有
    quantize_mask = ((protect_mask == 0) & (prune_mask == 0)).long()

    # 将掩码重塑为原始形状
    original_shape = values.shape
    return PartitionResult(
        protect_mask=protect_mask.reshape(original_shape),
        prune_mask=prune_mask.reshape(original_shape),
        quantize_mask=quantize_mask.reshape(original_shape),
    )
