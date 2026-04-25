"""
ExCP 压缩器 - 端到端检查点压缩。

实现 ExCP 算法:
1. 计算残差检查点 (ΔW_t = W_t - Ŵ_{t-1})
2. 权重和优化器状态的联合剪枝
3. K-means 量化
4. 序列化（可选通用压缩）
"""

import torch
import pickle
import gzip
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .residual import compute_residual_state_dict, reconstruct_state_dict
from .pruning import joint_prune
from .quantization import kmeans_quantize_nonzero, dequantize, pack_int4, unpack_int4


@dataclass
class ExCPConfig:
    """ExCP 压缩配置（论文默认超参数）。"""
    alpha: float = 5e-5  # 权重阈值超参数（论文 §4.2）
    beta: float = 2.0    # 动量阈值超参数（论文 §4.2）
    p: float = 0.0       # 残差百分位剪枝（论文无此步骤，设 0 禁用）
    n_bits: int = 4      # 量化位数
    use_gzip: bool = True  # 是否使用 gzip 进行最终压缩


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
        prev_W_hat: Optional[Dict[str, torch.Tensor]] = None
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
        compressed_data = {
            "weights": {},
            "optimizer": {},
            "metadata": {
                "config": self.config,
                "is_first": prev_W_hat is None,
            }
        }

        for key in W_t:
            W = W_t[key]

            # 获取此参数的优化器状态
            if key in O_t:
                v_t = O_t[key].get("exp_avg", torch.zeros_like(W))
                m_t = O_t[key].get("exp_avg_sq", torch.ones_like(W))
            else:
                v_t = torch.zeros_like(W)
                m_t = torch.ones_like(W)

            # 步骤 1: 计算残差（或对第一个检查点使用完整权重）
            if prev_W_hat is not None and key in prev_W_hat:
                dW = W - prev_W_hat[key]
            else:
                dW = W.clone()

            # 步骤 2: 联合剪枝
            dW_pruned, v_pruned = joint_prune(
                dW, v_t, m_t,
                self.config.alpha,
                self.config.beta,
                self.config.p
            )

            # 步骤 3: 量化权重
            w_indices, w_centers = kmeans_quantize_nonzero(
                dW_pruned, self.config.n_bits
            )

            # 步骤 4: 量化优化器状态
            o_indices, o_centers = kmeans_quantize_nonzero(
                v_pruned, self.config.n_bits
            )

            # 将索引打包为 int4
            w_packed = pack_int4(w_indices.flatten().to(torch.uint8))
            o_packed = pack_int4(o_indices.flatten().to(torch.uint8))

            # 存储压缩数据
            compressed_data["weights"][key] = {
                "indices": w_packed.numpy(),
                "centers": w_centers.numpy(),
                "shape": list(W.shape),
                "dtype": str(W.dtype),
            }
            compressed_data["optimizer"][key] = {
                "indices": o_packed.numpy(),
                "centers": o_centers.numpy(),
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
        prev_W_hat: Optional[Dict[str, torch.Tensor]] = None
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

        # 反序列化
        data = pickle.loads(decompressed)

        W_hat = {}
        O_hat = {}

        for key in data["weights"]:
            w_data = data["weights"][key]
            o_data = data["optimizer"][key]

            # 解包索引
            w_packed = torch.from_numpy(w_data["indices"])
            o_packed = torch.from_numpy(o_data["indices"])

            shape = tuple(w_data["shape"])
            numel = 1
            for s in shape:
                numel *= s

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
                W_hat[key] = prev_W_hat[key] + dW_hat
            else:
                W_hat[key] = dW_hat

            # 转换为原始 dtype
            dtype_str = w_data["dtype"]
            if "float32" in dtype_str:
                W_hat[key] = W_hat[key].float()
            elif "float16" in dtype_str:
                W_hat[key] = W_hat[key].half()

            # 存储优化器状态
            O_hat[key] = {
                "exp_avg": v_hat,
                "exp_avg_sq": torch.ones_like(v_hat) * 0.1,  # 占位符
            }

        return W_hat, O_hat

    @property
    def name(self) -> str:
        return "ExCP"
