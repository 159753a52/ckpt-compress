"""
Inshrinkerator 分位数草图模块。

实现基于 DDSketch 的分位数估计，用于高效阈值计算。
基于 Inshrinkerator 论文的算法 1。
"""

import math
from collections import defaultdict
from typing import Dict, Optional

import torch


def compute_gamma(alpha: float) -> float:
    """
    计算 DDSketch 的 gamma 参数。

    gamma = (1 + alpha) / (1 - alpha)

    参数:
        alpha: 相对精度参数 (0 < alpha < 1)

    返回:
        Gamma 值
    """
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be finite and in (0, 1), got {alpha}")
    return (1 + alpha) / (1 - alpha)


def compute_bucket_index(values: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    使用对数桶映射计算值的桶索引。

    bucket_id = floor(log_gamma(x)) = floor(log(x) / log(gamma))

    参数:
        values: 正值张量
        gamma: Gamma 参数

    返回:
        桶索引张量
    """
    if not math.isfinite(gamma) or gamma <= 1.0:
        raise ValueError(f"gamma must be finite and greater than 1, got {gamma}")
    if values.is_complex():
        raise TypeError("values must be real")
    working = values if values.is_floating_point() else values.float()
    if not torch.isfinite(working).all().item():
        raise ValueError("values must be finite")

    working = working.clone()
    working[working <= 0] = torch.finfo(working.dtype).tiny

    log_gamma = math.log(gamma)
    indices = torch.floor(torch.log(working) / log_gamma).long()

    return indices


class QuantileSketch:
    """
    基于 DDSketch 的分位数草图，用于高效分位数估计。

    提供具有有界相对误差的近似分位数查询。
    """

    def __init__(self, alpha: float = 0.01):
        """
        初始化分位数草图。

        参数:
            alpha: 相对精度参数（默认 0.01 = 1% 误差）
        """
        self.alpha = alpha
        self.gamma = compute_gamma(alpha)
        self.buckets: Dict[int, int] = defaultdict(int)
        self.count = 0
        self.min_index: Optional[int] = None
        self.max_index: Optional[int] = None

    def add(self, values: torch.Tensor) -> None:
        """
        向草图添加值。

        参数:
            values: 要添加的正值张量
        """
        if values.numel() == 0:
            return

        # 计算桶索引
        indices = compute_bucket_index(values, self.gamma)

        # 更新桶计数
        for raw_index in indices.flatten().tolist():
            idx = int(raw_index)
            self.buckets[idx] += 1
            self.count += 1

            if self.min_index is None or idx < self.min_index:
                self.min_index = idx
            if self.max_index is None or idx > self.max_index:
                self.max_index = idx

    def quantile(self, q: float) -> float:
        """
        估计第 q 分位数。

        参数:
            q: 要估计的分位数 (0 <= q <= 1)

        返回:
            估计的分位数值
        """
        if not math.isfinite(q) or not 0.0 <= q <= 1.0:
            raise ValueError(f"q must be finite and in [0, 1], got {q}")
        if self.count == 0:
            return 0.0

        target_rank = q * self.count

        # 从最小桶到最大桶累积计数
        cumulative = 0
        sorted_indices = sorted(self.buckets.keys())

        for idx in sorted_indices:
            cumulative += self.buckets[idx]
            if cumulative >= target_rank:
                # 返回桶中心值
                # 桶 idx 包含范围 [gamma^idx, gamma^(idx+1)) 内的值
                # 返回几何平均值作为代表
                bucket_low = self.gamma**idx
                bucket_high = self.gamma ** (idx + 1)
                return float(math.sqrt(bucket_low * bucket_high))

        # 如果到达这里，返回最大桶中心
        if self.max_index is not None:
            return float(self.gamma ** (self.max_index + 0.5))

        return 0.0

    def merge(self, other: "QuantileSketch") -> None:
        """
        将另一个草图合并到此草图中。

        参数:
            other: 具有相同 alpha 的另一个 QuantileSketch
        """
        if not math.isclose(self.alpha, other.alpha, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("Cannot merge quantile sketches with different alpha values")
        if other.count == 0:
            return

        for idx, count in other.buckets.items():
            self.buckets[idx] += count

        self.count += other.count

        if other.min_index is not None:
            if self.min_index is None or other.min_index < self.min_index:
                self.min_index = other.min_index

        if other.max_index is not None:
            if self.max_index is None or other.max_index > self.max_index:
                self.max_index = other.max_index

    def get_bucket_histogram(self) -> Dict[int, int]:
        """
        获取桶直方图。

        返回:
            桶索引到计数的映射字典
        """
        return dict(self.buckets)
