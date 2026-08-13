"""
Inshrinkerator 压缩器 - 端到端检查点压缩。

实现 Inshrinkerator 算法:
1. 参数分区（保护/剪枝/量化）
2. 基于草图的近似 K-means 量化
3. 带参数重排的增量编码
4. RLE 压缩
"""

import gzip
import hashlib
import math
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch

from .approx_kmeans import approx_kmeans, quantize_to_centers
from .delta_encoding import (
    delta_decode,
    delta_encode,
    rearrange_by_prev_bin,
    restore_rearrangement,
    rle_decode,
    rle_encode,
)
from .metrics import magnitude, sensitivity
from .partition import PartitionConfig, PartitionResult, partition

_DTYPES = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}


@dataclass
class InshrinkeratorConfig:
    """Inshrinkerator 压缩配置。"""

    n_bins: int = 16  # 量化桶数
    protect_fraction: float = 0.005  # 保护比例
    prune_fraction: float = 0.2  # 剪枝比例
    use_gzip: bool = True  # 是否使用 gzip 进行最终压缩
    seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.n_bins, int) or isinstance(self.n_bins, bool):
            raise TypeError("n_bins must be an integer")
        if not 1 <= self.n_bins <= 32768:
            raise ValueError(f"n_bins must be in [1, 32768], got {self.n_bins}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed}")
        for name, value in (
            ("protect_fraction", self.protect_fraction),
            ("prune_fraction", self.prune_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
        if self.protect_fraction + self.prune_fraction > 1.0:
            raise ValueError("protect_fraction + prune_fraction must not exceed 1")


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
        prev_quantized: Optional[Dict[str, torch.Tensor]] = None,
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
        compressed_data: Dict[str, Any] = {
            "weights": {},
            "metadata": {
                "config": self.config,
                "is_first": prev_quantized is None,
            },
        }

        if set(grad) != set(W_t):
            missing = sorted(set(W_t).difference(grad))
            extra = sorted(set(grad).difference(W_t))
            raise ValueError(
                f"Gradient keys must match weight keys: missing={missing}, extra={extra}"
            )

        for key in W_t:
            W = W_t[key]
            g = grad[key]
            self._validate_input_tensors(key, W, g)

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
                centers = approx_kmeans(
                    quantize_values.abs(),
                    self.config.n_bins,
                    seed=self._parameter_seed(key),
                )
                # 通过使用符号处理有符号值
                signs = torch.sign(quantize_values)
                q_indices = quantize_to_centers(quantize_values.abs(), centers)
            else:
                centers = W.new_zeros(self.config.n_bins)
                q_indices = torch.empty(0, dtype=torch.long, device=W.device)
                signs = W.new_empty(0)

            # 步骤 4: 如果有前一检查点则进行增量编码
            if prev_quantized is not None and key in prev_quantized:
                prev_q = prev_quantized[key].to(part_result.quantize_mask.device)
                if prev_q.shape != W.shape:
                    raise ValueError(f"Previous quantized shape does not match weight {key!r}")
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
                storage_dtype = (
                    torch.float32
                    if protected_values.dtype == torch.bfloat16
                    else protected_values.dtype
                )
                protected_np = protected_values.detach().to(storage_dtype).cpu().numpy()
            else:
                protected_np = None

            # 存储压缩数据
            compressed_data["weights"][key] = {
                "encoded_groups": encoded_groups,
                "centers": centers.detach().float().cpu().numpy(),
                "signs": signs.detach().float().cpu().numpy() if signs.numel() > 0 else None,
                "protected_values": protected_np,
                "protect_mask": part_result.protect_mask.detach().cpu().numpy(),
                "prune_mask": part_result.prune_mask.detach().cpu().numpy(),
                "quantize_mask": part_result.quantize_mask.detach().cpu().numpy(),
                "shape": list(W.shape),
                "dtype": str(W.dtype),
                "use_delta": use_delta,
                "n_quantized": q_indices.numel(),
                "encoding_n_bins": self.config.n_bins,
            }

        # 序列化
        serialized = pickle.dumps(compressed_data)

        if self.config.use_gzip:
            serialized = gzip.compress(serialized)

        return serialized

    def decompress(
        self, compressed: bytes, prev_quantized: Optional[Dict[str, torch.Tensor]] = None
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

        # 注意: pickle.loads 可执行任意代码，仅加载可信来源的数据
        data = pickle.loads(decompressed)
        W_hat = {}

        for key in data["weights"]:
            w_data = data["weights"][key]
            shape = tuple(w_data["shape"])
            n_quantized = w_data["n_quantized"]

            centers = torch.from_numpy(w_data["centers"])
            previous = prev_quantized.get(key) if prev_quantized is not None else None
            encoding_n_bins = self._encoding_n_bins(w_data, len(centers))
            q_indices = self._decode_quantized_indices(
                w_data,
                previous,
                encoding_n_bins,
            )
            if q_indices.numel() and q_indices.max().item() >= len(centers):
                raise ValueError("Quantized indices reference a missing center")

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
            if not isinstance(dtype_str, str) or dtype_str not in _DTYPES:
                raise ValueError(
                    f"Unsupported tensor dtype in Inshrinkerator payload: {dtype_str!r}"
                )
            W_reconstructed = W_reconstructed.to(_DTYPES[dtype_str])

            W_hat[key] = W_reconstructed

        return W_hat

    def get_quantized_indices(
        self,
        compressed: bytes,
        prev_quantized: Optional[Dict[str, torch.Tensor]] = None,
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

        # 注意: pickle.loads 可执行任意代码，仅加载可信来源的数据
        data = pickle.loads(decompressed)
        q_indices = {}

        for key in data["weights"]:
            w_data = data["weights"][key]
            shape = tuple(w_data["shape"])
            n_quantized = w_data["n_quantized"]

            centers = w_data["centers"]
            previous = prev_quantized.get(key) if prev_quantized is not None else None
            encoding_n_bins = self._encoding_n_bins(w_data, len(centers))
            indices = self._decode_quantized_indices(
                w_data,
                previous,
                encoding_n_bins,
            )
            if indices.numel() and indices.max().item() >= len(centers):
                raise ValueError("Quantized indices reference a missing center")

            # 创建完整索引张量
            full_indices = torch.zeros(shape, dtype=torch.long).flatten()
            quantize_mask = torch.from_numpy(w_data["quantize_mask"]).flatten()
            full_indices[quantize_mask == 1] = indices

            q_indices[key] = full_indices.reshape(shape)

        return q_indices

    @staticmethod
    def _decode_quantized_indices(
        weight_data: Mapping[str, Any],
        previous: Optional[torch.Tensor],
        n_bins: int,
    ) -> torch.Tensor:
        """Decode one parameter's compact indices for reconstruction or chaining."""
        encoded_groups = weight_data["encoded_groups"]
        n_quantized = int(weight_data["n_quantized"])
        if not weight_data["use_delta"]:
            if len(encoded_groups) != 1:
                raise ValueError("Direct index payload must contain exactly one RLE group")
            return rle_decode(encoded_groups[0], n_quantized)

        if previous is None:
            raise ValueError("Delta-encoded indices require the previous quantized tensor")
        if n_bins < 1 or len(encoded_groups) != n_bins:
            raise ValueError(
                f"Delta index payload has {len(encoded_groups)} groups; expected {n_bins}"
            )

        quantize_mask = torch.from_numpy(weight_data["quantize_mask"]).bool()
        if previous.shape != quantize_mask.shape:
            raise ValueError(
                "Previous quantized tensor shape does not match the current quantize mask"
            )
        previous_masked = previous[quantize_mask]
        if previous_masked.numel() != n_quantized:
            raise ValueError(
                "Previous quantized tensor does not cover the current quantized partition"
            )
        if previous_masked.numel() and (
            previous_masked.min().item() < 0 or previous_masked.max().item() >= n_bins
        ):
            raise ValueError("Previous quantized indices are outside the encoded bin range")

        decoded_groups = [
            rle_decode(encoded, int((previous_masked == bin_index).sum().item()))
            for bin_index, encoded in enumerate(encoded_groups)
        ]
        delta = restore_rearrangement(previous_masked, decoded_groups, n_bins)
        return delta_decode(previous_masked, delta, n_bins)

    @staticmethod
    def _encoding_n_bins(weight_data: Mapping[str, Any], center_count: int) -> int:
        raw_value = weight_data.get("encoding_n_bins", center_count)
        if (
            not isinstance(raw_value, int)
            or isinstance(raw_value, bool)
            or raw_value < max(center_count, 1)
            or raw_value > 32768
        ):
            raise ValueError("Invalid encoding_n_bins in Inshrinkerator payload")
        return raw_value

    @property
    def name(self) -> str:
        return "Inshrinkerator"

    def _parameter_seed(self, name: str) -> int:
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        return (self.config.seed + int.from_bytes(digest[:8], "little")) % (2**63)

    @staticmethod
    def _validate_input_tensors(
        name: str,
        weight: torch.Tensor,
        gradient: torch.Tensor,
    ) -> None:
        for label, tensor in (("weight", weight), ("gradient", gradient)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{label} {name!r} must be a torch.Tensor")
            if not tensor.is_floating_point() or tensor.is_complex():
                raise TypeError(f"{label} {name!r} must be a real floating-point tensor")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{label} {name!r} must contain only finite values")
        if gradient.shape != weight.shape:
            raise ValueError(f"Gradient shape does not match weight {name!r}")
        if gradient.device != weight.device:
            raise ValueError(f"Gradient device does not match weight {name!r}")
