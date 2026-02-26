"""
预测残差压缩 - 优化器状态压缩模块。

高效压缩 Adam 优化器状态 (exp_avg, exp_avg_sq):
- exp_avg: 一阶矩，可正可负，使用对称量化
- exp_avg_sq: 二阶矩，始终非负，使用对数尺度量化

核心洞察: 优化器状态通常比权重更大，但具有
可预测的结构，可以利用这一点进行压缩。
"""

import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass


def compress_exp_avg(
    exp_avg: torch.Tensor,
    n_bits: int = 8
) -> Dict[str, Any]:
    """
    使用对称量化压缩一阶矩 (exp_avg)。

    exp_avg 可正可负，因此使用对称量化。

    参数:
        exp_avg: 一阶矩张量
        n_bits: 量化位数

    返回:
        压缩数据字典
    """
    shape = exp_avg.shape
    flat = exp_avg.flatten()

    # 对称量化
    n_levels = 2 ** n_bits
    v_abs_max = flat.abs().max().item()

    if v_abs_max < 1e-10:
        # 全零
        return {
            "indices": torch.zeros_like(flat, dtype=torch.int16),
            "scale": 1.0,
            "shape": shape,
            "is_zero": True,
        }

    scale = v_abs_max / (n_levels // 2 - 1)
    zero_point = n_levels // 2

    # 量化
    indices = torch.round(flat / scale).to(torch.int16) + zero_point
    indices = torch.clamp(indices, 0, n_levels - 1)

    return {
        "indices": indices,
        "scale": scale,
        "zero_point": zero_point,
        "shape": shape,
        "is_zero": False,
    }


def decompress_exp_avg(compressed: Dict[str, Any]) -> torch.Tensor:
    """
    解压一阶矩 (exp_avg)。

    参数:
        compressed: 来自 compress_exp_avg 的压缩数据

    返回:
        重建的 exp_avg 张量
    """
    shape = compressed["shape"]

    if compressed.get("is_zero", False):
        return torch.zeros(shape)

    indices = compressed["indices"]
    scale = compressed["scale"]
    zero_point = compressed["zero_point"]

    values = (indices.float() - zero_point) * scale
    return values.reshape(shape)


def compress_exp_avg_sq(
    exp_avg_sq: torch.Tensor,
    n_bits: int = 8
) -> Dict[str, Any]:
    """
    使用对数尺度量化压缩二阶矩 (exp_avg_sq)。

    exp_avg_sq 始终非负，且通常具有跨越多个数量级的值。
    对数尺度量化在所有尺度上保持相对精度。

    参数:
        exp_avg_sq: 二阶矩张量
        n_bits: 量化位数

    返回:
        压缩数据字典
    """
    shape = exp_avg_sq.shape
    flat = exp_avg_sq.flatten()

    # 处理零和非常小的值
    eps = 1e-10
    flat_safe = torch.clamp(flat, min=eps)

    # 对数尺度量化
    log_values = torch.log(flat_safe)
    log_min = log_values.min().item()
    log_max = log_values.max().item()

    if log_max - log_min < 1e-8:
        # 所有值相同
        return {
            "indices": torch.zeros_like(flat, dtype=torch.int16),
            "log_min": log_min,
            "log_max": log_max,
            "shape": shape,
            "is_uniform": True,
        }

    n_levels = 2 ** n_bits
    scale = (log_max - log_min) / (n_levels - 1)

    # 在对数空间中量化
    indices = torch.round((log_values - log_min) / scale).to(torch.int16)
    indices = torch.clamp(indices, 0, n_levels - 1)

    return {
        "indices": indices,
        "scale": scale,
        "log_min": log_min,
        "shape": shape,
        "is_uniform": False,
    }


def decompress_exp_avg_sq(compressed: Dict[str, Any]) -> torch.Tensor:
    """
    解压二阶矩 (exp_avg_sq)。

    参数:
        compressed: 来自 compress_exp_avg_sq 的压缩数据

    返回:
        重建的 exp_avg_sq 张量（非负）
    """
    shape = compressed["shape"]

    if compressed.get("is_uniform", False):
        # 所有值相同
        value = torch.exp(torch.tensor(compressed["log_min"]))
        return torch.full(shape, value.item())

    indices = compressed["indices"]
    scale = compressed["scale"]
    log_min = compressed["log_min"]

    # 在对数空间中反量化
    log_values = indices.float() * scale + log_min

    # 从对数空间转换回来
    values = torch.exp(log_values)
    return values.reshape(shape)


@dataclass
class OptimizerCompressConfig:
    """优化器状态压缩配置。"""
    n_bits: int = 8
    use_delta: bool = False


class OptimizerStateCompressor:
    """
    Adam 优化器状态压缩器。

    使用适合各自特性的量化策略高效压缩 exp_avg 和 exp_avg_sq。
    """

    def __init__(
        self,
        n_bits: int = 8,
        use_delta: bool = False
    ):
        """
        初始化优化器状态压缩器。

        参数:
            n_bits: 量化位数
            use_delta: 是否在检查点之间使用增量编码
        """
        self.n_bits = n_bits
        self.use_delta = use_delta

    def compress(
        self,
        optimizer_state: Dict[str, Dict[str, torch.Tensor]],
        prev_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        压缩优化器状态。

        参数:
            optimizer_state: 包含 exp_avg 和 exp_avg_sq 的优化器状态字典
            prev_state: 用于增量编码的前一优化器状态

        返回:
            压缩后的优化器状态
        """
        compressed = {}

        for key in optimizer_state:
            state = optimizer_state[key]
            compressed[key] = {}

            # 压缩 exp_avg
            if "exp_avg" in state:
                exp_avg = state["exp_avg"]

                if self.use_delta and prev_state and key in prev_state:
                    # 增量编码
                    prev_exp_avg = prev_state[key].get("exp_avg")
                    if prev_exp_avg is not None:
                        delta = exp_avg - prev_exp_avg
                        compressed[key]["exp_avg"] = compress_exp_avg(delta, self.n_bits)
                        compressed[key]["exp_avg"]["is_delta"] = True
                    else:
                        compressed[key]["exp_avg"] = compress_exp_avg(exp_avg, self.n_bits)
                        compressed[key]["exp_avg"]["is_delta"] = False
                else:
                    compressed[key]["exp_avg"] = compress_exp_avg(exp_avg, self.n_bits)
                    compressed[key]["exp_avg"]["is_delta"] = False

            # 压缩 exp_avg_sq
            if "exp_avg_sq" in state:
                exp_avg_sq = state["exp_avg_sq"]

                if self.use_delta and prev_state and key in prev_state:
                    # 对于 exp_avg_sq，使用比率而非增量（因为它始终为正）
                    prev_exp_avg_sq = prev_state[key].get("exp_avg_sq")
                    if prev_exp_avg_sq is not None:
                        # 在对数空间中存储比率
                        ratio = exp_avg_sq / (prev_exp_avg_sq + 1e-10)
                        compressed[key]["exp_avg_sq"] = compress_exp_avg_sq(ratio, self.n_bits)
                        compressed[key]["exp_avg_sq"]["is_ratio"] = True
                    else:
                        compressed[key]["exp_avg_sq"] = compress_exp_avg_sq(exp_avg_sq, self.n_bits)
                        compressed[key]["exp_avg_sq"]["is_ratio"] = False
                else:
                    compressed[key]["exp_avg_sq"] = compress_exp_avg_sq(exp_avg_sq, self.n_bits)
                    compressed[key]["exp_avg_sq"]["is_ratio"] = False

        return compressed

    def decompress(
        self,
        compressed: Dict[str, Dict[str, Any]],
        prev_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        解压优化器状态。

        参数:
            compressed: 压缩后的优化器状态
            prev_state: 用于增量解码的前一优化器状态

        返回:
            重建的优化器状态
        """
        reconstructed = {}

        for key in compressed:
            reconstructed[key] = {}

            # 解压 exp_avg
            if "exp_avg" in compressed[key]:
                data = compressed[key]["exp_avg"]
                value = decompress_exp_avg(data)

                if data.get("is_delta", False) and prev_state and key in prev_state:
                    prev_exp_avg = prev_state[key].get("exp_avg")
                    if prev_exp_avg is not None:
                        value = value + prev_exp_avg

                reconstructed[key]["exp_avg"] = value

            # 解压 exp_avg_sq
            if "exp_avg_sq" in compressed[key]:
                data = compressed[key]["exp_avg_sq"]
                value = decompress_exp_avg_sq(data)

                if data.get("is_ratio", False) and prev_state and key in prev_state:
                    prev_exp_avg_sq = prev_state[key].get("exp_avg_sq")
                    if prev_exp_avg_sq is not None:
                        value = value * prev_exp_avg_sq

                reconstructed[key]["exp_avg_sq"] = value

        return reconstructed
