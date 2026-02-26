"""
预测残差压缩 - 端到端压缩器。

组合:
- 基于 Adam 的权重预测以获得更小的残差
- 基于敏感度的自适应量化
- 优化器状态压缩

相比 ExCP 和 Inshrinkerator 的关键优势:
1. 基于预测的残差编码（比直接增量更小）
2. 敏感度感知的位数分配（保留重要参数）
3. 同时压缩权重和优化器状态
"""

import torch
import pickle
import gzip
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .predictor import AdamPredictor
from .adaptive_quantization import AdaptiveQuantizer
from .optimizer_compress import OptimizerStateCompressor


@dataclass
class PredictiveConfig:
    """预测残差压缩器配置。"""
    # 预测器配置
    lr: float = 0.001
    eps: float = 1e-8

    # 量化配置
    min_bits: int = 4
    max_bits: int = 8

    # 优化器压缩配置
    optimizer_bits: int = 8

    # 通用配置
    use_gzip: bool = True


class PredictiveCompressor:
    """
    预测残差检查点压缩器。

    此方法结合了 ExCP 和 Inshrinkerator 的优势:
    - 类似 ExCP: 使用残差编码进行压缩
    - 类似 Inshrinkerator: 使用敏感度保留重要参数
    - 创新点: 使用优化器状态预测下一个权重，使残差更小

    核心洞察是，如果我们能从 W_{t-1} 和优化器状态预测 W_t，
    预测残差 R = W_t - W_pred 将比直接增量 W_t - W_{t-1} 小得多，
    从而实现更好的压缩。
    """

    def __init__(self, config: Optional[PredictiveConfig] = None):
        """
        初始化预测残差压缩器。

        参数:
            config: 压缩配置（如果为 None 则使用默认值）
        """
        self.config = config or PredictiveConfig()

        # 初始化组件
        self.predictor = AdamPredictor(
            lr=self.config.lr,
            eps=self.config.eps
        )
        self.quantizer = AdaptiveQuantizer(
            min_bits=self.config.min_bits,
            max_bits=self.config.max_bits
        )
        self.optimizer_compressor = OptimizerStateCompressor(
            n_bits=self.config.optimizer_bits,
            use_delta=True
        )

    def compress(
        self,
        W_t: Dict[str, torch.Tensor],
        O_t: Dict[str, Dict[str, torch.Tensor]],
        grad: Dict[str, torch.Tensor],
        prev_checkpoint: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        压缩检查点。

        参数:
            W_t: 当前模型权重状态字典
            O_t: 当前优化器状态字典
            grad: 用于敏感度计算的梯度
            prev_checkpoint: 用于链式压缩的前一检查点数据
                            {"W": prev_weights, "O": prev_optimizer_state}

        返回:
            压缩后的检查点字节数据
        """
        compressed_data = {
            "weights": {},
            "optimizer": {},
            "metadata": {
                "config": self.config,
                "is_first": prev_checkpoint is None,
            }
        }

        # 如果可用则提取前一状态
        W_prev = prev_checkpoint["W"] if prev_checkpoint else None
        O_prev = prev_checkpoint["O"] if prev_checkpoint else None

        # 步骤 1: 计算预测残差（如果是第一个检查点则直接存储权重）
        if W_prev is not None and O_prev is not None:
            # 使用基于预测的残差
            residual = self.predictor.compute_residual(W_t, W_prev, O_prev)
            compressed_data["metadata"]["use_prediction"] = True
        else:
            # 第一个检查点，直接存储权重
            residual = W_t
            compressed_data["metadata"]["use_prediction"] = False

        # 步骤 2: 使用敏感度感知位数分配量化残差
        for key in residual:
            tensor = residual[key]
            g = grad.get(key, torch.zeros_like(tensor))

            # 量化
            q_result = self.quantizer.quantize(tensor, g)

            # 将张量转换为 numpy 以便序列化
            compressed_data["weights"][key] = {
                "indices": q_result["indices"].numpy(),
                "scale": q_result["scale"],
                "zero_point": q_result["zero_point"],
                "bits": q_result["bits"],
                "shape": list(q_result["shape"]),
            }

        # 步骤 3: 压缩优化器状态
        if O_t:
            compressed_opt = self.optimizer_compressor.compress(O_t, O_prev)

            for key in compressed_opt:
                compressed_data["optimizer"][key] = {}

                if "exp_avg" in compressed_opt[key]:
                    exp_avg_data = compressed_opt[key]["exp_avg"]
                    compressed_data["optimizer"][key]["exp_avg"] = {
                        "indices": exp_avg_data["indices"].numpy(),
                        "scale": exp_avg_data["scale"],
                        "shape": list(exp_avg_data["shape"]),
                        "is_zero": exp_avg_data.get("is_zero", False),
                        "is_delta": exp_avg_data.get("is_delta", False),
                    }
                    if "zero_point" in exp_avg_data:
                        compressed_data["optimizer"][key]["exp_avg"]["zero_point"] = exp_avg_data["zero_point"]

                if "exp_avg_sq" in compressed_opt[key]:
                    exp_avg_sq_data = compressed_opt[key]["exp_avg_sq"]
                    compressed_data["optimizer"][key]["exp_avg_sq"] = {
                        "indices": exp_avg_sq_data["indices"].numpy(),
                        "scale": exp_avg_sq_data["scale"],
                        "log_min": exp_avg_sq_data["log_min"],
                        "shape": list(exp_avg_sq_data["shape"]),
                        "is_uniform": exp_avg_sq_data.get("is_uniform", False),
                        "is_ratio": exp_avg_sq_data.get("is_ratio", False),
                    }

        # 序列化
        serialized = pickle.dumps(compressed_data)

        if self.config.use_gzip:
            serialized = gzip.compress(serialized)

        return serialized

    def decompress(
        self,
        compressed: bytes,
        prev_checkpoint: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
        """
        解压检查点。

        参数:
            compressed: 压缩后的检查点字节数据
            prev_checkpoint: 用于链式解压的前一检查点数据

        返回:
            (权重字典, 优化器状态字典) 元组
        """
        # 如果需要解压 gzip
        try:
            decompressed = gzip.decompress(compressed)
        except gzip.BadGzipFile:
            decompressed = compressed

        data = pickle.loads(decompressed)

        # 如果可用则提取前一状态
        W_prev = prev_checkpoint["W"] if prev_checkpoint else None
        O_prev = prev_checkpoint["O"] if prev_checkpoint else None

        # 步骤 1: 反量化权重/残差
        residual = {}
        for key in data["weights"]:
            w_data = data["weights"][key]

            q_result = {
                "indices": torch.from_numpy(w_data["indices"]),
                "scale": w_data["scale"],
                "zero_point": w_data["zero_point"],
                "bits": w_data["bits"],
                "shape": tuple(w_data["shape"]),
            }

            residual[key] = self.quantizer.dequantize(q_result)

        # 步骤 2: 从残差重建权重
        if data["metadata"].get("use_prediction", False) and W_prev is not None and O_prev is not None:
            W_hat = self.predictor.reconstruct(residual, W_prev, O_prev)
        else:
            W_hat = residual

        # 步骤 3: 解压优化器状态
        O_hat = {}
        if "optimizer" in data and data["optimizer"]:
            # 为优化器解压器重建压缩格式
            compressed_opt = {}
            for key in data["optimizer"]:
                compressed_opt[key] = {}

                if "exp_avg" in data["optimizer"][key]:
                    exp_avg_data = data["optimizer"][key]["exp_avg"]
                    compressed_opt[key]["exp_avg"] = {
                        "indices": torch.from_numpy(exp_avg_data["indices"]),
                        "scale": exp_avg_data["scale"],
                        "shape": tuple(exp_avg_data["shape"]),
                        "is_zero": exp_avg_data.get("is_zero", False),
                        "is_delta": exp_avg_data.get("is_delta", False),
                    }
                    if "zero_point" in exp_avg_data:
                        compressed_opt[key]["exp_avg"]["zero_point"] = exp_avg_data["zero_point"]

                if "exp_avg_sq" in data["optimizer"][key]:
                    exp_avg_sq_data = data["optimizer"][key]["exp_avg_sq"]
                    compressed_opt[key]["exp_avg_sq"] = {
                        "indices": torch.from_numpy(exp_avg_sq_data["indices"]),
                        "scale": exp_avg_sq_data["scale"],
                        "log_min": exp_avg_sq_data["log_min"],
                        "shape": tuple(exp_avg_sq_data["shape"]),
                        "is_uniform": exp_avg_sq_data.get("is_uniform", False),
                        "is_ratio": exp_avg_sq_data.get("is_ratio", False),
                    }

            O_hat = self.optimizer_compressor.decompress(compressed_opt, O_prev)

        return W_hat, O_hat

    @property
    def name(self) -> str:
        return "PredictiveResidual"
