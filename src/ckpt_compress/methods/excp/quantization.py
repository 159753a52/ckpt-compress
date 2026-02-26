"""
ExCP K-means 量化模块。

使用 K-means 聚类实现非均匀量化:
- 非零值被聚类到 2^n - 1 个中心
- 零值作为特殊级别保留（索引 0）
- 支持 int4 打包以提高存储效率
"""

import torch
from typing import Tuple


def kmeans_quantize_nonzero(
    x: torch.Tensor,
    n_bits: int = 4,
    max_iter: int = 100,
    tol: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 K-means 量化张量，保留零值。

    非零值被聚类到 2^n - 1 个中心。
    零值获得索引 0 和中心 0。

    参数:
        x: 要量化的输入张量
        n_bits: 量化位数（默认 4 -> 16 个级别）
        max_iter: K-means 最大迭代次数
        tol: 收敛容差

    返回:
        (indices, centers) 元组，其中:
        - indices: 量化索引（与 x 形状相同）
        - centers: 聚类中心（长度为 2^n_bits）
    """
    original_shape = x.shape
    x_flat = x.flatten()

    # 量化级别数
    n_levels = 2 ** n_bits
    n_nonzero_centers = n_levels - 1  # 索引 0 保留给零值

    # 处理边界情况
    if x_flat.numel() == 0:
        centers = torch.zeros(n_levels, dtype=x.dtype, device=x.device)
        indices = torch.zeros(0, dtype=torch.long, device=x.device)
        return indices.reshape(original_shape), centers

    # 分离零值和非零值
    zero_mask = (x_flat == 0)
    nonzero_mask = ~zero_mask
    nonzero_values = x_flat[nonzero_mask]

    # 初始化索引（初始全为零）
    indices = torch.zeros(x_flat.numel(), dtype=torch.long, device=x.device)

    # 初始化中心，索引 0 为零
    centers = torch.zeros(n_levels, dtype=x.dtype, device=x.device)

    if nonzero_values.numel() == 0:
        # 全零情况
        return indices.reshape(original_shape), centers

    # 获取唯一的非零值
    unique_nonzero = torch.unique(nonzero_values)
    n_unique = unique_nonzero.numel()

    # 确定实际使用的中心数
    actual_n_centers = min(n_nonzero_centers, n_unique)

    if actual_n_centers == 0:
        return indices.reshape(original_shape), centers

    # 使用 K-means++ 风格初始化中心
    if n_unique <= actual_n_centers:
        # 使用所有唯一值作为中心
        nonzero_centers = unique_nonzero[:actual_n_centers]
    else:
        # K-means++ 初始化
        nonzero_centers = _kmeans_plusplus_init(nonzero_values, actual_n_centers)

    # 运行 K-means
    for _ in range(max_iter):
        # 将每个非零值分配到最近的中心
        distances = torch.abs(nonzero_values.unsqueeze(1) - nonzero_centers.unsqueeze(0))
        assignments = torch.argmin(distances, dim=1)

        # 更新中心
        new_centers = torch.zeros_like(nonzero_centers)
        for i in range(actual_n_centers):
            mask = (assignments == i)
            if mask.sum() > 0:
                new_centers[i] = nonzero_values[mask].mean()
            else:
                new_centers[i] = nonzero_centers[i]

        # 检查收敛
        if torch.allclose(new_centers, nonzero_centers, atol=tol):
            nonzero_centers = new_centers
            break
        nonzero_centers = new_centers

    # 最终分配
    distances = torch.abs(nonzero_values.unsqueeze(1) - nonzero_centers.unsqueeze(0))
    nonzero_indices = torch.argmin(distances, dim=1) + 1  # +1 因为 0 保留给零值

    # 填充非零位置的索引
    indices[nonzero_mask] = nonzero_indices

    # 填充中心（索引 0 已经是 0）
    centers[1:actual_n_centers + 1] = nonzero_centers

    return indices.reshape(original_shape), centers


def _kmeans_plusplus_init(
    values: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    K-means++ 聚类中心初始化。

    参数:
        values: 要聚类的一维张量
        k: 要初始化的中心数

    返回:
        包含 k 个初始中心的张量
    """
    n = values.numel()
    centers = torch.zeros(k, dtype=values.dtype, device=values.device)

    # 第一个中心：随机选择
    idx = torch.randint(0, n, (1,)).item()
    centers[0] = values[idx]

    for i in range(1, k):
        # 计算到最近现有中心的距离
        distances = torch.abs(values.unsqueeze(1) - centers[:i].unsqueeze(0))
        min_distances = distances.min(dim=1).values

        # 距离平方作为概率权重
        probs = min_distances ** 2
        probs = probs / probs.sum()

        # 采样下一个中心
        idx = torch.multinomial(probs, 1).item()
        centers[i] = values[idx]

    return centers


def dequantize(
    indices: torch.Tensor,
    centers: torch.Tensor
) -> torch.Tensor:
    """
    使用中心将索引反量化回值。

    参数:
        indices: 量化索引
        centers: 聚类中心

    返回:
        反量化的张量（与 indices 形状相同）
    """
    return centers[indices]


def pack_int4(indices: torch.Tensor) -> torch.Tensor:
    """
    将 int4 索引打包到 int8（每字节两个索引）。

    参数:
        indices: 范围 [0, 15] 的索引张量

    返回:
        打包后的张量（长度减半，向上取整）
    """
    indices = indices.flatten().to(torch.uint8)
    n = indices.numel()

    # 如果需要，填充到偶数长度
    if n % 2 == 1:
        indices = torch.cat([indices, torch.zeros(1, dtype=torch.uint8, device=indices.device)])

    # 重塑为成对
    indices = indices.reshape(-1, 2)

    # 打包：高半字节 = 第一个，低半字节 = 第二个
    packed = (indices[:, 0] << 4) | indices[:, 1]

    return packed


def unpack_int4(packed: torch.Tensor, original_length: int) -> torch.Tensor:
    """
    将 int8 解包回 int4 索引。

    参数:
        packed: 打包后的张量
        original_length: 原始索引数量

    返回:
        解包后的索引张量
    """
    packed = packed.flatten()

    # 解包高半字节和低半字节
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F

    # 交错排列
    unpacked = torch.stack([high, low], dim=1).flatten()

    # 裁剪到原始长度
    return unpacked[:original_length].to(torch.uint8)
