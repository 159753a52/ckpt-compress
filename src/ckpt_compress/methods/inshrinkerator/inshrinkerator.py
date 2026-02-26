"""
Inshrinkerator 压缩器 - 端到端检查点压缩。

实现 Inshrinkerator 算法:
1. 参数分区（保护/剪枝/量化）
2. 基于草图的近似 K-means 量化
3. 带参数重排的增量编码
4. RLE 压缩
"""

import torch
import pickle
import gzip
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass

from .metrics import magnitude, sensitivity
from .partition import partition, PartitionConfig, PartitionResult
from .approx_kmeans import approx_kmeans, quantize_to_centers
from .delta_encoding import (
    delta_encode, delta_decode,
    rearrange_by_prev_bin, restore_rearrangement,
    rle_encode, rle_decode,
)


@dataclass
class InshrinkeratorConfig:
    """Inshrinkerator 压缩配置。"""
    n_bins: int = 16           # 量化桶数
    protect_fraction: float = 0.005  # 保护比例
    prune_fraction: float = 0.2      # 剪枝比例
    use_gzip: bool = True      # 是否使用 gzip 进行最终压缩


class InshrinkeratorCompressor:
    """
    Inshrinkerator 检查点压缩器。

    实现 Inshrinkerator 算法，包含:
    - 三向分区（保护/剪枝/量化）
    - 基于草图的近似 K-means
    - 带 RLE 的增量编码
    """

    def __init__(self, config: Optional[InshrinkeratorConfig] = None):
        """
        初始化 Inshrinkerator 压缩器。

        参数:
            config: 压缩配置（如果为 None 则使用默认值）
        """
        self.config = config or InshrinkeratorConfig()

    def compress(
        self,
        W_t: Dict[str, torch.Tensor],
        grad: Dict[str, torch.Tensor],
        prev_quantized: Optional[Dict[str, torch.Tensor]] = None
    ) -> bytes:
        """
        压缩检查点。

        参数:
            W_t: 当前模型权重状态字典
            grad: 用于敏感度计算的梯度
            prev_quantized: 前一量化索引（用于增量编码）

        返回:
            压缩后的检查点字节数据
        """
        compressed_data = {
            "weights": {},
            "metadata": {
                "config": self.config,
                "is_first": prev_quantized is None,
            }
        }

        for key in W_t:
            W = W_t[key]
            g = grad.get(key, torch.zeros_like(W))

            # 步骤 1: 分区参数
            part_config = PartitionConfig(
                protect_fraction=self.config.protect_fraction,
                prune_fraction=self.config.prune_fraction,
                n_bins=self.config.n_bins,
            )
            part_result = partition(W, g, part_config)

            # 步骤 2: 获取要量化的值
            quantize_values = W[part_result.quantize_mask == 1]

            # 步骤 3: 使用近似 K-means 计算中心
            if quantize_values.numel() > 0:
                centers = approx_kmeans(quantize_values.abs(), self.config.n_bins)
                # 通过使用符号处理有符号值
                signs = torch.sign(quantize_values)
                q_indices = quantize_to_centers(quantize_values.abs(), centers)
            else:
                centers = torch.zeros(self.config.n_bins)
                q_indices = torch.tensor([], dtype=torch.long)
                signs = torch.tensor([])

            # 步骤 4: 如果有前一检查点则进行增量编码
            if prev_quantized is not None and key in prev_quantized:
                prev_q = prev_quantized[key]
                # 仅对量化部分进行增量编码
                prev_q_masked = prev_q[part_result.quantize_mask == 1]
                if prev_q_masked.numel() == q_indices.numel():
                    D = delta_encode(prev_q_masked, q_indices, self.config.n_bins)
                    grouped = rearrange_by_prev_bin(prev_q_masked, D, self.config.n_bins)
                    encoded_groups = [rle_encode(g) for g in grouped]
                    use_delta = True
                else:
                    encoded_groups = [rle_encode(q_indices)]
                    use_delta = False
            else:
                encoded_groups = [rle_encode(q_indices)]
                use_delta = False

            # 步骤 5: 存储受保护值（将 bfloat16 转换为 float16 以供 numpy 使用）
            protected_values = part_result.get_protected_values(W)
            if protected_values.numel() > 0:
                protected_np = protected_values.to(torch.float16).numpy()
            else:
                protected_np = None

            # 存储压缩数据
            compressed_data["weights"][key] = {
                "encoded_groups": encoded_groups,
                "centers": centers.numpy(),
                "signs": signs.numpy() if signs.numel() > 0 else None,
                "protected_values": protected_np,
                "protect_mask": part_result.protect_mask.numpy(),
                "prune_mask": part_result.prune_mask.numpy(),
                "quantize_mask": part_result.quantize_mask.numpy(),
                "shape": list(W.shape),
                "dtype": str(W.dtype),
                "use_delta": use_delta,
                "n_quantized": q_indices.numel(),
            }

        # 序列化
        serialized = pickle.dumps(compressed_data)

        if self.config.use_gzip:
            serialized = gzip.compress(serialized)

        return serialized

    def decompress(
        self,
        compressed: bytes,
        prev_quantized: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        解压检查点。

        参数:
            compressed: 压缩后的检查点字节数据
            prev_quantized: 前一量化索引（用于增量解码）

        返回:
            重建的权重状态字典
        """
        # 如果需要解压 gzip
        try:
            decompressed = gzip.decompress(compressed)
        except gzip.BadGzipFile:
            decompressed = compressed

        data = pickle.loads(decompressed)
        W_hat = {}

        for key in data["weights"]:
            w_data = data["weights"][key]
            shape = tuple(w_data["shape"])
            n_quantized = w_data["n_quantized"]

            # 解码量化索引
            encoded_groups = w_data["encoded_groups"]
            centers = torch.from_numpy(w_data["centers"])

            if w_data["use_delta"] and prev_quantized is not None and key in prev_quantized:
                # 增量解码
                prev_q = prev_quantized[key]
                quantize_mask = torch.from_numpy(w_data["quantize_mask"])
                prev_q_masked = prev_q[quantize_mask == 1]

                # 解码每个组
                grouped = rearrange_by_prev_bin(prev_q_masked, torch.zeros_like(prev_q_masked), len(centers))
                decoded_groups = []
                for i, enc in enumerate(encoded_groups):
                    group_len = (prev_q_masked == i).sum().item()
                    decoded_groups.append(rle_decode(enc, group_len))

                D = restore_rearrangement(prev_q_masked, decoded_groups, len(centers))
                q_indices = delta_decode(prev_q_masked, D, len(centers))
            else:
                # 直接解码
                q_indices = rle_decode(encoded_groups[0], n_quantized)

            # 反量化
            if q_indices.numel() > 0:
                quantized_values = centers[q_indices]
                # 应用符号
                if w_data["signs"] is not None:
                    signs = torch.from_numpy(w_data["signs"])
                    quantized_values = quantized_values * signs
            else:
                quantized_values = torch.tensor([])

            # 重建完整张量
            W_reconstructed = torch.zeros(shape).flatten()

            protect_mask = torch.from_numpy(w_data["protect_mask"]).flatten()
            prune_mask = torch.from_numpy(w_data["prune_mask"]).flatten()
            quantize_mask = torch.from_numpy(w_data["quantize_mask"]).flatten()

            # 填充受保护值
            if w_data["protected_values"] is not None:
                protected = torch.from_numpy(w_data["protected_values"]).float()
                W_reconstructed[protect_mask == 1] = protected

            # 剪枝值保持为零

            # 填充量化值
            if quantized_values.numel() > 0:
                W_reconstructed[quantize_mask == 1] = quantized_values.float()

            W_reconstructed = W_reconstructed.reshape(shape)

            # 转换为原始 dtype
            dtype_str = w_data["dtype"]
            if "float32" in dtype_str:
                W_reconstructed = W_reconstructed.float()
            elif "float16" in dtype_str:
                W_reconstructed = W_reconstructed.half()

            W_hat[key] = W_reconstructed

        return W_hat

    def get_quantized_indices(
        self,
        compressed: bytes
    ) -> Dict[str, torch.Tensor]:
        """
        从压缩数据中提取量化索引用于增量编码。

        参数:
            compressed: 压缩后的检查点字节数据

        返回:
            每个参数的量化索引字典
        """
        try:
            decompressed = gzip.decompress(compressed)
        except gzip.BadGzipFile:
            decompressed = compressed

        data = pickle.loads(decompressed)
        q_indices = {}

        for key in data["weights"]:
            w_data = data["weights"][key]
            shape = tuple(w_data["shape"])
            n_quantized = w_data["n_quantized"]

            # 解码索引
            encoded_groups = w_data["encoded_groups"]
            indices = rle_decode(encoded_groups[0], n_quantized)

            # 创建完整索引张量
            full_indices = torch.zeros(shape, dtype=torch.long).flatten()
            quantize_mask = torch.from_numpy(w_data["quantize_mask"]).flatten()
            full_indices[quantize_mask == 1] = indices

            q_indices[key] = full_indices.reshape(shape)

        return q_indices

    @property
    def name(self) -> str:
        return "Inshrinkerator"
