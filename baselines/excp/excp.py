"""
ExCP 压缩器 - 端到端检查点压缩。

实现 ExCP 算法:
1. 计算残差检查点 (ΔW_t = W_t - Ŵ_{t-1})
2. 权重和优化器状态的联合剪枝
3. K-means 量化
4. 序列化（可选通用压缩）
"""

import gzip
import hashlib
import math
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from .pruning import joint_prune
from .quantization import dequantize, kmeans_quantize_nonzero, pack_int4, unpack_int4

_DTYPES = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}


@dataclass
class ExCPConfig:
    """ExCP 压缩配置（论文默认超参数）。"""

    alpha: float = 5e-5  # 权重阈值超参数（论文 §4.2）
    beta: float = 2.0  # 动量阈值超参数（论文 §4.2）
    p: float = 0.0  # 残差百分位剪枝（论文无此步骤，设 0 禁用）
    n_bits: int = 4  # 量化位数
    use_gzip: bool = True  # 是否使用 gzip 进行最终压缩
    seed: int = 42

    def __post_init__(self) -> None:
        for name, value in (("alpha", self.alpha), ("beta", self.beta)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if not math.isfinite(self.p) or not 0.0 <= self.p <= 100.0:
            raise ValueError(f"p must be finite and in [0, 100], got {self.p}")
        if not isinstance(self.n_bits, int) or isinstance(self.n_bits, bool):
            raise TypeError("n_bits must be an integer")
        if not 1 <= self.n_bits <= 4:
            raise ValueError(f"n_bits must be in [1, 4] for int4 packing, got {self.n_bits}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed}")


class ExCPCompressor:
    """
    ExCP 检查点压缩器。

    实现 ExCP 检查点压缩算法，包含:
    - 残差编码
    - 权重-动量联合剪枝
    - K-means 非均匀量化
    """

    def __init__(self, config: Optional[ExCPConfig] = None):
        """
        初始化 ExCP 压缩器。

        参数:
            config: 压缩配置（如果为 None 则使用默认值）
        """
        self.config = config or ExCPConfig()

    def compress(
        self,
        W_t: Dict[str, torch.Tensor],
        O_t: Dict[str, Dict[str, torch.Tensor]],
        prev_W_hat: Optional[Dict[str, torch.Tensor]] = None,
    ) -> bytes:
        """
        压缩检查点。

        参数:
            W_t: 当前模型权重状态字典
            O_t: 当前优化器状态字典 (param_name -> {exp_avg, exp_avg_sq})
            prev_W_hat: 前一重建权重（第一个检查点为 None）

        返回:
            压缩后的检查点字节数据
        """
        compressed_data: Dict[str, Any] = {
            "weights": {},
            "optimizer": {},
            "metadata": {
                "config": self.config,
                "is_first": prev_W_hat is None,
            },
        }

        for key in W_t:
            W = W_t[key]
            self._validate_weight(key, W)

            # 获取此参数的优化器状态
            if key in O_t:
                v_t = O_t[key].get("exp_avg", torch.zeros_like(W))
                m_t = O_t[key].get("exp_avg_sq", torch.ones_like(W))
            else:
                v_t = torch.zeros_like(W)
                m_t = torch.ones_like(W)
            self._validate_state_tensor(key, "exp_avg", v_t, W)
            self._validate_state_tensor(key, "exp_avg_sq", m_t, W)
            if (m_t < 0).any().item():
                raise ValueError(f"exp_avg_sq for {key!r} must be non-negative")

            # 步骤 1: 计算残差（或对第一个检查点使用完整权重）
            if prev_W_hat is not None and key in prev_W_hat:
                self._validate_state_tensor(key, "previous weight", prev_W_hat[key], W)
                dW = W - prev_W_hat[key]
            else:
                dW = W.clone()

            # 步骤 2: 联合剪枝
            dW_pruned, v_pruned = joint_prune(
                dW,
                v_t,
                m_t,
                self.config.alpha,
                self.config.beta,
                self.config.p,
            )

            # 步骤 3: 量化权重
            w_indices, w_centers = kmeans_quantize_nonzero(
                dW_pruned,
                self.config.n_bits,
                seed=self._parameter_seed(key, "weight"),
            )

            # 步骤 4: 量化优化器状态
            o_indices, o_centers = kmeans_quantize_nonzero(
                v_pruned,
                self.config.n_bits,
                seed=self._parameter_seed(key, "optimizer"),
            )

            # 将索引打包为 int4
            w_packed = pack_int4(w_indices)
            o_packed = pack_int4(o_indices)

            # 存储压缩数据
            compressed_data["weights"][key] = {
                "indices": w_packed.cpu().numpy(),
                "centers": w_centers.float().cpu().numpy(),
                "shape": list(W.shape),
                "dtype": str(W.dtype),
            }
            compressed_data["optimizer"][key] = {
                "indices": o_packed.cpu().numpy(),
                "centers": o_centers.float().cpu().numpy(),
                "shape": list(v_t.shape),
            }

        # 序列化
        serialized = pickle.dumps(compressed_data)

        # 可选 gzip 压缩
        if self.config.use_gzip:
            serialized = gzip.compress(serialized)

        return serialized

    def decompress(
        self,
        compressed: bytes,
        prev_W_hat: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
        """
        解压检查点。

        参数:
            compressed: 压缩后的检查点字节数据
            prev_W_hat: 前一重建权重（第一个检查点为 None）

        返回:
            (reconstructed_weights, reconstructed_optimizer) 元组
        """
        # 如果需要解压 gzip
        try:
            decompressed = gzip.decompress(compressed)
        except gzip.BadGzipFile:
            decompressed = compressed

        # pickle 可执行任意代码；这里只允许调用方提供可信的本地实验载荷。
        data = pickle.loads(decompressed)
        if not isinstance(data, Mapping):
            raise ValueError("ExCP payload must be an object")
        weights = data.get("weights")
        optimizer = data.get("optimizer")
        if not isinstance(weights, Mapping) or not isinstance(optimizer, Mapping):
            raise ValueError("ExCP payload must contain weight and optimizer objects")

        W_hat: Dict[str, torch.Tensor] = {}
        O_hat: Dict[str, Dict[str, torch.Tensor]] = {}

        for key, raw_weight_data in weights.items():
            if not isinstance(key, str) or not isinstance(raw_weight_data, Mapping):
                raise ValueError("ExCP weight entries must be named objects")
            raw_optimizer_data = optimizer.get(key)
            if not isinstance(raw_optimizer_data, Mapping):
                raise ValueError(f"ExCP payload is missing optimizer data for {key!r}")
            w_data = raw_weight_data
            o_data = raw_optimizer_data

            # 解包索引
            w_packed = torch.from_numpy(w_data["indices"])
            o_packed = torch.from_numpy(o_data["indices"])

            shape = self._parse_shape(w_data.get("shape"), key)
            optimizer_shape = self._parse_shape(o_data.get("shape"), key)
            if optimizer_shape != shape:
                raise ValueError(f"Optimizer shape does not match weight shape for {key!r}")
            numel = math.prod(shape)

            w_indices = unpack_int4(w_packed, numel).long()
            o_indices = unpack_int4(o_packed, numel).long()

            # 获取中心
            w_centers = torch.from_numpy(w_data["centers"])
            o_centers = torch.from_numpy(o_data["centers"])

            # 反量化
            dW_hat = dequantize(w_indices, w_centers).reshape(shape)
            v_hat = dequantize(o_indices, o_centers).reshape(shape)

            # 重建权重
            if prev_W_hat is not None and key in prev_W_hat:
                if tuple(prev_W_hat[key].shape) != shape:
                    raise ValueError(f"Previous weight shape does not match payload for {key!r}")
                W_hat[key] = prev_W_hat[key].to(dW_hat.device) + dW_hat
            else:
                W_hat[key] = dW_hat

            # 转换为原始 dtype
            dtype_str = w_data.get("dtype")
            if not isinstance(dtype_str, str) or dtype_str not in _DTYPES:
                raise ValueError(f"Unsupported tensor dtype in ExCP payload: {dtype_str!r}")
            W_hat[key] = W_hat[key].to(_DTYPES[dtype_str])

            # This adapter encodes exp_avg only. Do not fabricate an exp_avg_sq state.
            O_hat[key] = {"exp_avg": v_hat}

        return W_hat, O_hat

    @staticmethod
    def _validate_weight(name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Weight {name!r} must be a torch.Tensor")
        if not value.is_floating_point() or value.is_complex():
            raise TypeError(f"Weight {name!r} must be a real floating-point tensor")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Weight {name!r} must contain only finite values")

    @staticmethod
    def _validate_state_tensor(
        name: str,
        state_name: str,
        value: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{state_name} for {name!r} must be a torch.Tensor")
        if value.shape != weight.shape:
            raise ValueError(f"{state_name} shape does not match weight {name!r}")
        if value.device != weight.device:
            raise ValueError(f"{state_name} device does not match weight {name!r}")
        if not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(f"{state_name} for {name!r} must be finite floating point")

    @staticmethod
    def _parse_shape(value: object, name: str) -> Tuple[int, ...]:
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(dimension, int) and not isinstance(dimension, bool) and dimension >= 0
            for dimension in value
        ):
            raise ValueError(f"Invalid tensor shape in ExCP payload for {name!r}")
        return tuple(value)

    @property
    def name(self) -> str:
        return "ExCP"

    def _parameter_seed(self, name: str, component: str) -> int:
        digest = hashlib.sha256(f"{name}:{component}".encode("utf-8")).digest()
        return (self.config.seed + int.from_bytes(digest[:8], "little")) % (2**63)
