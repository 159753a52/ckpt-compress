"""
Inshrinkerator 近似 K-means 模块。

实现基于草图的近似 K-means，用于高效非均匀量化。
基于 Inshrinkerator 论文的算法 2/3。
"""

import math

import torch

from .sketch import QuantileSketch


def compute_sample_weights(
    freq: torch.Tensor,
    mag: torch.Tensor,
    sigma: float = 0.2,
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
    if freq.shape != mag.shape:
        raise ValueError("freq and mag must have identical shapes")
    if freq.device != mag.device:
        raise ValueError("freq and mag must be on the same device")
    if not math.isfinite(sigma) or not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma must be finite and in [0, 1], got {sigma}")
    if not torch.isfinite(freq).all().item() or not torch.isfinite(mag).all().item():
        raise ValueError("freq and mag must be finite")
    if (freq < 0).any().item() or (mag < 0).any().item():
        raise ValueError("freq and mag must be non-negative")
    return sigma * freq + (1 - sigma) * mag


def weighted_kmeans_plusplus_init(
    values: torch.Tensor,
    weights: torch.Tensor,
    k: int,
    *,
    generator: torch.Generator | None = None,
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
    values = values.flatten()
    weights = weights.flatten()
    n = values.numel()
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if weights.numel() != n:
        raise ValueError("weights must contain one value per sample")
    if values.device != weights.device:
        raise ValueError("values and weights must be on the same device")
    if not torch.isfinite(values).all().item() or not torch.isfinite(weights).all().item():
        raise ValueError("values and weights must be finite")
    if (weights < 0).any().item():
        raise ValueError("weights must be non-negative")
    if n == 0:
        return values.new_empty((0,))

    k = min(k, n)
    centers = torch.zeros(k, dtype=values.dtype, device=values.device)

    # 第一个中心：加权随机选择
    weight_sum = weights.sum()
    probs = weights / weight_sum if weight_sum.item() > 0 else torch.full_like(weights, 1 / n)
    idx = torch.multinomial(probs, 1, generator=generator).item()
    centers[0] = values[idx]

    for i in range(1, k):
        # 计算到最近现有中心的距离
        distances = torch.abs(values.unsqueeze(1) - centers[:i].unsqueeze(0))
        min_distances = distances.min(dim=1).values

        # 按距离平方和样本权重加权
        selection_probs = (min_distances**2) * weights
        if selection_probs.sum().item() <= 0:
            selection_probs = weights.clone()
        selection_sum = selection_probs.sum()
        if selection_sum.item() <= 0:
            selection_probs = torch.full_like(selection_probs, 1 / n)
        else:
            selection_probs = selection_probs / selection_sum

        # 采样下一个中心
        idx = torch.multinomial(selection_probs, 1, generator=generator).item()
        centers[i] = values[idx]

    return centers


def approx_kmeans(
    values: torch.Tensor,
    k: int,
    alpha: float = 0.01,
    sigma: float = 0.2,
    max_iter: int = 50,
    tol: float = 1e-4,
    seed: int = 42,
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

    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError(f"tol must be finite and positive, got {tol}")
    if not values.is_floating_point() or values.is_complex():
        raise TypeError("values must be a real floating-point tensor")
    if not torch.isfinite(values).all().item():
        raise ValueError("values must be finite")
    if (values < 0).any().item():
        raise ValueError("values must contain non-negative magnitudes")
    compute_sample_weights(values.new_empty((0,)), values.new_empty((0,)), sigma)

    if n == 0:
        return values.new_empty((0,))

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
        return values.mean().reshape(1)

    # 将桶转换为代表值和权重
    bucket_indices = sorted(bucket_hist.keys())
    bucket_values_list: list[float] = []
    bucket_freqs_list: list[int] = []
    bucket_mags_list: list[float] = []

    gamma = sketch.gamma
    for idx in bucket_indices:
        count = bucket_hist[idx]
        # 桶中心值
        center = gamma ** (idx + 0.5)
        bucket_values_list.append(center)
        bucket_freqs_list.append(count)
        bucket_mags_list.append(center * count)  # 幅度贡献

    bucket_values = values.new_tensor(bucket_values_list)
    bucket_freqs = torch.tensor(bucket_freqs_list, dtype=torch.float32, device=values.device)
    bucket_mags = torch.tensor(bucket_mags_list, dtype=torch.float32, device=values.device)

    # 归一化
    bucket_freqs = bucket_freqs / bucket_freqs.sum()
    bucket_mags = bucket_mags / (bucket_mags.sum() + 1e-10)

    # 计算样本权重
    weights = compute_sample_weights(bucket_freqs, bucket_mags, sigma)

    # 使用加权 K-means++ 初始化中心
    actual_k = min(k, len(bucket_values))
    generator = torch.Generator(device=values.device).manual_seed(seed)
    centers = weighted_kmeans_plusplus_init(
        bucket_values,
        weights,
        actual_k,
        generator=generator,
    )

    # 在桶代表上运行 K-means
    for _ in range(max_iter):
        # 将桶分配到最近的中心
        distances = torch.abs(bucket_values.unsqueeze(1) - centers.unsqueeze(0))
        assignments = torch.argmin(distances, dim=1)

        # 更新中心（按桶频率加权）
        new_centers = torch.zeros_like(centers)
        for i in range(actual_k):
            mask = assignments == i
            if mask.any().item():
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
    centers: torch.Tensor,
) -> torch.Tensor:
    """
    将值量化到最近的中心。

    参数:
        values: 要量化的值
        centers: 聚类中心

    返回:
        最近中心的索引
    """
    if centers.ndim != 1:
        raise ValueError("centers must be one-dimensional")
    if values.device != centers.device:
        raise ValueError("values and centers must be on the same device")
    if not torch.isfinite(values).all().item() or not torch.isfinite(centers).all().item():
        raise ValueError("values and centers must be finite")
    if centers.numel() == 0:
        return torch.zeros(values.shape, dtype=torch.long, device=values.device)

    values_flat = values.flatten()
    distances = torch.abs(values_flat.unsqueeze(1) - centers.unsqueeze(0))
    indices = torch.argmin(distances, dim=1)

    return indices.reshape(values.shape)
