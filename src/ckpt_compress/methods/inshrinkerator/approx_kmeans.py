"""
Inshrinkerator 近似 K-means 模块。

实现基于草图的近似 K-means，用于高效非均匀量化。
基于 Inshrinkerator 论文的算法 2/3。
"""

import torch
from typing import Optional

from .sketch import QuantileSketch


def compute_sample_weights(
    freq: torch.Tensor,
    mag: torch.Tensor,
    sigma: float = 0.2
) -> torch.Tensor:
    """
    计算结合频率和幅度的样本权重。

    w^i = sigma * freq + (1 - sigma) * mag

    参数:
        freq: 归一化频率权重
        mag: 归一化幅度权重
        sigma: 混合系数（默认 0.2）

    返回:
        组合权重
    """
    return sigma * freq + (1 - sigma) * mag


def weighted_kmeans_plusplus_init(
    values: torch.Tensor,
    weights: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    加权 K-means++ 初始化。

    参数:
        values: 一维值张量
        weights: 样本权重
        k: 中心数

    返回:
        初始中心张量
    """
    n = values.numel()
    if n == 0:
        return torch.tensor([])

    k = min(k, n)
    centers = torch.zeros(k, dtype=values.dtype, device=values.device)

    # 第一个中心：加权随机选择
    probs = weights / weights.sum()
    idx = torch.multinomial(probs, 1).item()
    centers[0] = values[idx]

    for i in range(1, k):
        # 计算到最近现有中心的距离
        distances = torch.abs(values.unsqueeze(1) - centers[:i].unsqueeze(0))
        min_distances = distances.min(dim=1).values

        # 按距离平方和样本权重加权
        selection_probs = (min_distances ** 2) * weights
        if selection_probs.sum() == 0:
            selection_probs = weights.clone()
        selection_probs = selection_probs / selection_probs.sum()

        # 采样下一个中心
        idx = torch.multinomial(selection_probs, 1).item()
        centers[i] = values[idx]

    return centers


def approx_kmeans(
    values: torch.Tensor,
    k: int,
    alpha: float = 0.01,
    sigma: float = 0.2,
    max_iter: int = 50,
    tol: float = 1e-4
) -> torch.Tensor:
    """
    使用基于草图的直方图进行近似 K-means。

    参数:
        values: 要聚类的一维值张量
        k: 聚类数
        alpha: 草图精度参数
        sigma: 权重混合系数
        max_iter: 最大迭代次数
        tol: 收敛容差

    返回:
        排序后的聚类中心
    """
    values = values.flatten()
    n = values.numel()

    if n == 0:
        return torch.tensor([])

    if n <= k:
        # 唯一值不足
        unique = torch.unique(values)
        return torch.sort(unique).values

    # 构建草图直方图
    sketch = QuantileSketch(alpha=alpha)
    sketch.add(values.abs())

    # 获取桶直方图
    bucket_hist = sketch.get_bucket_histogram()

    if len(bucket_hist) == 0:
        return torch.tensor([values.mean()])

    # 将桶转换为代表值和权重
    bucket_indices = sorted(bucket_hist.keys())
    bucket_values = []
    bucket_freqs = []
    bucket_mags = []

    gamma = sketch.gamma
    for idx in bucket_indices:
        count = bucket_hist[idx]
        # 桶中心值
        center = gamma ** (idx + 0.5)
        bucket_values.append(center)
        bucket_freqs.append(count)
        bucket_mags.append(center * count)  # 幅度贡献

    bucket_values = torch.tensor(bucket_values, dtype=values.dtype)
    bucket_freqs = torch.tensor(bucket_freqs, dtype=torch.float32)
    bucket_mags = torch.tensor(bucket_mags, dtype=torch.float32)

    # 归一化
    bucket_freqs = bucket_freqs / bucket_freqs.sum()
    bucket_mags = bucket_mags / (bucket_mags.sum() + 1e-10)

    # 计算样本权重
    weights = compute_sample_weights(bucket_freqs, bucket_mags, sigma)

    # 使用加权 K-means++ 初始化中心
    actual_k = min(k, len(bucket_values))
    centers = weighted_kmeans_plusplus_init(bucket_values, weights, actual_k)

    # 在桶代表上运行 K-means
    for _ in range(max_iter):
        # 将桶分配到最近的中心
        distances = torch.abs(bucket_values.unsqueeze(1) - centers.unsqueeze(0))
        assignments = torch.argmin(distances, dim=1)

        # 更新中心（按桶频率加权）
        new_centers = torch.zeros_like(centers)
        for i in range(actual_k):
            mask = (assignments == i)
            if mask.sum() > 0:
                weighted_sum = (bucket_values[mask] * bucket_freqs[mask]).sum()
                weight_sum = bucket_freqs[mask].sum()
                new_centers[i] = weighted_sum / (weight_sum + 1e-10)
            else:
                new_centers[i] = centers[i]

        # 检查收敛
        if torch.allclose(new_centers, centers, atol=tol):
            centers = new_centers
            break
        centers = new_centers

    # 排序中心以保持一致编码
    centers = torch.sort(centers).values

    return centers


def quantize_to_centers(
    values: torch.Tensor,
    centers: torch.Tensor
) -> torch.Tensor:
    """
    将值量化到最近的中心。

    参数:
        values: 要量化的值
        centers: 聚类中心

    返回:
        最近中心的索引
    """
    if centers.numel() == 0:
        return torch.zeros(values.shape, dtype=torch.long)

    values_flat = values.flatten()
    distances = torch.abs(values_flat.unsqueeze(1) - centers.unsqueeze(0))
    indices = torch.argmin(distances, dim=1)

    return indices.reshape(values.shape)
