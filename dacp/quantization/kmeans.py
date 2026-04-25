"""
K-means 量化器

使用 DDSketch 对数直方图加速的近似 K-means 聚类量化。
灵感来自 Inshrinkerator (SoCC'24) 的两阶段方案：
1. DDSketch 对数空间投影：将百万参数降维为几千个直方图桶
2. 在桶上做加权 K-means++ 聚类：时间从 O(n·k) 降至 O(B·k)

核心设计（对齐 Inshrinkerator）：
- 聚类在绝对值上进行，符号单独存储 → k 个 centroid 全部用于幅度分辨率
- 支持 mask-aware 模式：排除剪枝后的零值，避免浪费 centroid
- 加权策略：w_i = σ·频率 + (1-σ)·幅度，平衡分辨率分配
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional


def _dtype_from_str(dtype_str: str) -> torch.dtype:
    """将字符串形式的 dtype 安全映射为 torch.dtype。"""
    mapping = {
        'torch.float16': torch.float16,
        'torch.float32': torch.float32,
        'torch.float64': torch.float64,
        'torch.bfloat16': torch.bfloat16,
        'torch.int8': torch.int8,
        'torch.int16': torch.int16,
        'torch.int32': torch.int32,
        'torch.int64': torch.int64,
        'torch.uint8': torch.uint8,
        'torch.bool': torch.bool,
    }
    return mapping.get(dtype_str, torch.float32)


def _ddsketch_histogram(values: np.ndarray, alpha: float = 0.01):
    """DDSketch 对数空间直方图（仅正值 / 绝对值）。

    将输入数值投影到 log_γ 空间，生成 (桶中心值, 桶频率) 直方图。
    输入必须为非负值（绝对值），零值单独处理。

    Args:
        values: 1D numpy 数组，应为绝对值（非负）
        alpha: 相对误差界（默认 1%）

    Returns:
        bucket_centers: 桶的中心值（绝对值空间，均为正）
        bucket_counts: 每个桶的元素数量
        zero_count: 零值（含极小值）数量
    """
    gamma = (1 + alpha) / (1 - alpha)
    log_gamma = np.log(gamma)

    abs_vals = np.abs(values)  # 确保非负

    # 分离零值和非零值
    eps = 1e-30
    nonzero_mask = abs_vals > eps
    zero_count = int(np.sum(~nonzero_mask))
    abs_nonzero = abs_vals[nonzero_mask]

    if len(abs_nonzero) == 0:
        return np.array([0.0]), np.array([len(values)]), len(values)

    # 对数空间桶索引（仅正值，无需符号区分）
    bucket_indices = np.ceil(np.log(abs_nonzero) / log_gamma).astype(np.int64)

    unique_keys, inverse, counts = np.unique(bucket_indices, return_inverse=True, return_counts=True)

    # 还原桶中心值：取每个桶中所有绝对值的均值
    centers = np.zeros(len(unique_keys))
    for i in range(len(unique_keys)):
        mask = inverse == i
        centers[i] = np.mean(abs_nonzero[mask])

    return centers, counts, zero_count


def _weighted_kmeans_pp_init(data: np.ndarray, weights: np.ndarray, k: int,
                             rng: np.random.RandomState) -> np.ndarray:
    """加权 K-means++ 初始化。

    Args:
        data: 数据点 (B,)
        weights: 样本权重 (B,)
        k: 簇数
        rng: 随机数生成器

    Returns:
        centers: 初始簇中心 (k,)
    """
    n = len(data)
    # 按权重选第一个中心
    prob = weights / weights.sum()
    idx = rng.choice(n, p=prob)
    centers = [data[idx]]

    for _ in range(1, k):
        # 计算到最近中心的距离
        dists = np.min([(data - c) ** 2 for c in centers], axis=0)
        # 加权距离
        weighted_dists = dists * weights
        total = weighted_dists.sum()
        if total <= 0:
            idx = rng.choice(n)
        else:
            prob = weighted_dists / total
            idx = rng.choice(n, p=prob)
        centers.append(data[idx])

    return np.array(centers)


def _weighted_kmeans(data: np.ndarray, weights: np.ndarray, k: int,
                     max_iter: int = 50, seed: int = 42) -> np.ndarray:
    """在直方图桶上做加权 K-means 聚类。

    Args:
        data: 桶中心值 (B,)，应为正值（绝对值空间）
        weights: 桶权重 (B,)
        k: 簇数
        max_iter: 最大迭代（对齐 Inshrinkerator 默认 50）

    Returns:
        centroids: 最终簇中心 (k,)，已排序
    """
    rng = np.random.RandomState(seed)

    # 如果数据点不够 k 个，直接返回
    if len(data) <= k:
        return data.copy()

    # K-means++ 初始化
    centroids = _weighted_kmeans_pp_init(data, weights, k, rng)

    for _ in range(max_iter):
        # 分配：每个数据点到最近中心
        dists = np.abs(data[:, None] - centroids[None, :])  # (B, k)
        assignments = np.argmin(dists, axis=1)  # (B,)

        # 更新：加权均值
        new_centroids = np.zeros(k)
        for j in range(k):
            mask = assignments == j
            if mask.any():
                new_centroids[j] = np.average(data[mask], weights=weights[mask])
            else:
                new_centroids[j] = centroids[j]

        # 收敛检查
        if np.allclose(centroids, new_centroids, atol=1e-7):
            break
        centroids = new_centroids

    return np.sort(centroids)


class KMeansQuantizer:
    """
    DDSketch 加速的近似 K-means 量化器。

    两阶段流程：
    1. DDSketch 对数直方图将参数降维至 ~几千个桶
    2. 在桶上做加权 K-means++ 聚类，得到量化中心

    核心设计（对齐 Inshrinkerator）：
    - 在绝对值上聚类，符号单独存储 → k 个 centroid 全部用于幅度
    - 支持 mask 参数排除剪枝后的零值

    使用示例:
        quantizer = KMeansQuantizer(n_clusters=256)
        quantized, metadata = quantizer.quantize(weight_tensor)
        recovered = quantizer.dequantize(quantized, metadata)

        # mask-aware 模式（排除零值）:
        quantized, metadata = quantizer.quantize(weight_tensor, mask=pruning_mask)
        recovered = quantizer.dequantize(quantized, metadata)
    """

    def __init__(self, n_clusters: int = 256, max_iter: int = 50, seed: int = 42,
                 alpha: float = 0.01, sigma: float = 0.2):
        """
        Args:
            n_clusters: 簇数量（量化级别），全部分配给幅度
            max_iter: K-means 最大迭代（对齐 Inshrinkerator 默认 50）
            seed: 随机种子
            alpha: DDSketch 相对误差界（默认 1%）
            sigma: 权重混合系数，σ·频率 + (1-σ)·幅度（默认 0.2）
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        self.alpha = alpha
        self.sigma = sigma

    def quantize(
        self,
        weight: torch.Tensor,
        shape: Tuple[int, ...] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        对权重进行近似 K-means 量化（绝对值聚类 + 符号分离）。

        Args:
            weight: 输入权重张量
            shape: 原始形状（如果 weight 已被展平）
            mask: 可选的二值掩码（1=保留, 0=剪枝）。
                  若提供，仅对 mask=1 的位置聚类，mask=0 的位置保持 0。

        Returns:
            quantized: 量化后的索引张量（int64），mask=0 位置索引为 -1
            metadata: 解量化所需的元数据
        """
        if shape is None:
            shape = weight.shape

        weight_flat = weight.flatten().cpu().numpy().astype(np.float64)

        # mask-aware: 仅对非零/非剪枝位置聚类
        if mask is not None:
            mask_flat = mask.flatten().cpu().numpy().astype(bool)
        else:
            # 无 mask 时，排除精确零值（可能来自之前的剪枝）
            mask_flat = np.abs(weight_flat) > 1e-30

        active_values = weight_flat[mask_flat]

        if len(active_values) == 0:
            # 全部被剪枝，无需量化
            indices = torch.full(shape, -1, dtype=torch.int64)
            metadata = {
                'centroids': np.array([0.0]),
                'signs': np.zeros(np.prod(shape)),
                'shape': shape,
                'n_clusters': 0,
                'dtype': str(weight.dtype),
                'mask_flat': mask_flat,
            }
            return indices, metadata

        # 若活跃位置全部为 0，直接返回单零中心，避免重复零桶导致的不稳定行为
        active_abs = np.abs(active_values)
        if np.max(active_abs) <= 1e-30:
            full_labels = np.full(len(weight_flat), -1, dtype=np.int64)
            full_labels[mask_flat] = 0
            full_signs = np.zeros(len(weight_flat), dtype=np.float64)
            indices = torch.from_numpy(full_labels).to(torch.int64).view(shape)
            metadata = {
                'centroids': np.array([0.0]),
                'signs': full_signs,
                'shape': shape,
                'n_clusters': 1,
                'dtype': str(weight.dtype),
                'mask_flat': mask_flat,
            }
            return indices, metadata

        # 提取符号，在绝对值上聚类（对齐 Inshrinkerator）
        active_signs = np.sign(active_values)
        active_abs = np.abs(active_values)

        # 第一阶段：DDSketch 对数直方图（仅正值）
        bucket_centers, bucket_counts, zero_count = _ddsketch_histogram(
            active_abs, alpha=self.alpha)

        # 极少量零值（来自恰好为 0 的非剪枝权重），加入零桶
        if zero_count > 0:
            bucket_centers = np.concatenate([np.array([0.0]), bucket_centers])
            bucket_counts = np.concatenate([np.array([zero_count]), bucket_counts])

        # 计算加权 K-means 的样本权重
        freq_norm = bucket_counts / bucket_counts.sum()
        mag_norm = np.abs(bucket_centers)
        mag_sum = mag_norm.sum()
        if mag_sum > 0:
            mag_norm = mag_norm / mag_sum
        weights = self.sigma * freq_norm + (1 - self.sigma) * mag_norm

        # 第二阶段：在桶上做加权 K-means（所有 centroid 用于幅度）
        k = min(self.n_clusters, len(bucket_centers))
        centroids = _weighted_kmeans(
            bucket_centers, weights, k,
            max_iter=self.max_iter, seed=self.seed)

        # 将活跃参数分配到最近的中心（在绝对值空间）
        sorted_idx = np.argsort(centroids)
        sorted_centroids = centroids[sorted_idx]
        insert_pos = np.searchsorted(sorted_centroids, active_abs)
        insert_pos = np.clip(insert_pos, 1, len(sorted_centroids) - 1)
        left_dist = np.abs(active_abs - sorted_centroids[insert_pos - 1])
        right_dist = np.abs(active_abs - sorted_centroids[insert_pos])
        sorted_labels = np.where(left_dist <= right_dist, insert_pos - 1, insert_pos)
        active_labels = sorted_idx[sorted_labels]

        # 构建完整索引：活跃位置填充聚类索引，非活跃位置填 -1
        full_labels = np.full(len(weight_flat), -1, dtype=np.int64)
        full_labels[mask_flat] = active_labels

        # 构建完整符号数组
        full_signs = np.zeros(len(weight_flat), dtype=np.float64)
        full_signs[mask_flat] = active_signs

        indices = torch.from_numpy(full_labels).to(torch.int64).view(shape)

        metadata = {
            'centroids': centroids,  # 全部为正值（幅度 centroid）
            'signs': full_signs,
            'shape': shape,
            'n_clusters': k,
            'dtype': str(weight.dtype),
            'mask_flat': mask_flat,
        }

        return indices, metadata

    def dequantize(
        self,
        indices: torch.Tensor,
        metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """
        从量化索引恢复权重。

        centroid 表示幅度，乘以存储的符号恢复原始值。
        索引为 -1 的位置恢复为 0（剪枝位置）。

        Args:
            indices: 量化索引张量（-1 表示剪枝位置）
            metadata: 量化元数据（包含 centroids 和 signs）

        Returns:
            weight: 恢复的权重张量
        """
        centroids = metadata['centroids']
        signs = metadata['signs']
        shape = metadata['shape']

        indices_flat = indices.flatten().cpu().numpy()

        # 恢复幅度：-1 位置安全映射到 0
        active_mask = indices_flat >= 0
        weight_flat = np.zeros(len(indices_flat), dtype=np.float64)
        weight_flat[active_mask] = centroids[indices_flat[active_mask]]

        # 乘以符号恢复签名值
        weight_flat *= signs

        weight = torch.from_numpy(weight_flat).view(shape)

        if 'dtype' in metadata:
            dtype = _dtype_from_str(metadata['dtype'])
            weight = weight.to(dtype)

        return weight

    def compute_quantization_error(
        self,
        original: torch.Tensor,
        recovered: torch.Tensor
    ) -> float:
        """
        计算量化误差（相对 MSE）。
        """
        mse = torch.mean((original - recovered) ** 2)
        original_var = torch.var(original)
        relative_mse = (mse / (original_var + 1e-8)).item()
        return relative_mse
