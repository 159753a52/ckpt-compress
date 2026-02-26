"""
预测残差压缩 - 端到端压缩器测试。

TDD: 先写测试，再实现。

这是主压缩器，组合了:
- 基于 Adam 的权重预测
- 基于敏感度的自适应量化
- 优化器状态压缩
"""

import pytest
import torch

from ckpt_compress.methods.predictive.predictive import (
    PredictiveCompressor,
    PredictiveConfig,
)


class TestPredictiveCompressorBasic:
    """PredictiveCompressor 的基础测试。"""

    def test_compressor_init(self):
        """压缩器应该使用默认配置初始化。"""
        compressor = PredictiveCompressor()

        assert compressor.config is not None
        assert compressor.name == "PredictiveResidual"

    def test_compressor_custom_config(self):
        """压缩器应该接受自定义配置。"""
        config = PredictiveConfig(
            lr=0.01,
            min_bits=2,
            max_bits=8,
            use_gzip=True
        )
        compressor = PredictiveCompressor(config)

        assert compressor.config.lr == 0.01
        assert compressor.config.min_bits == 2
        assert compressor.config.max_bits == 8


class TestPredictiveCompressorRoundtrip:
    """压缩/解压往返测试。"""

    def test_compress_single_checkpoint(self):
        """应该压缩单个检查点。"""
        torch.manual_seed(42)
        compressor = PredictiveCompressor()

        W_t = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
        }
        O_t = {
            "layer1.weight": {
                "exp_avg": torch.randn(64, 32) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(64, 32)) * 0.01,
            },
            "layer1.bias": {
                "exp_avg": torch.randn(64) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(64)) * 0.01,
            },
        }
        grad = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
        }

        compressed = compressor.compress(W_t, O_t, grad, prev_checkpoint=None)

        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

    def test_decompress_single_checkpoint(self):
        """应该解压单个检查点。"""
        torch.manual_seed(42)
        compressor = PredictiveCompressor()

        W_t = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
        }
        O_t = {
            "layer1.weight": {
                "exp_avg": torch.randn(64, 32) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(64, 32)) * 0.01,
            },
            "layer1.bias": {
                "exp_avg": torch.randn(64) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(64)) * 0.01,
            },
        }
        grad = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
        }

        compressed = compressor.compress(W_t, O_t, grad, prev_checkpoint=None)
        W_hat, O_hat = compressor.decompress(compressed, prev_checkpoint=None)

        assert set(W_hat.keys()) == set(W_t.keys())
        for key in W_t:
            assert W_hat[key].shape == W_t[key].shape

    def test_roundtrip_preserves_structure(self):
        """压缩-解压应该保持张量结构。"""
        torch.manual_seed(42)
        compressor = PredictiveCompressor()

        W_t = {
            "fc1.weight": torch.randn(128, 64),
            "fc1.bias": torch.randn(128),
            "fc2.weight": torch.randn(64, 128),
            "fc2.bias": torch.randn(64),
        }
        O_t = {
            "fc1.weight": {
                "exp_avg": torch.randn(128, 64) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(128, 64)) * 0.01,
            },
            "fc1.bias": {
                "exp_avg": torch.randn(128) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(128)) * 0.01,
            },
            "fc2.weight": {
                "exp_avg": torch.randn(64, 128) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(64, 128)) * 0.01,
            },
            "fc2.bias": {
                "exp_avg": torch.randn(64) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(64)) * 0.01,
            },
        }
        grad = {key: torch.randn_like(W_t[key]) for key in W_t}

        compressed = compressor.compress(W_t, O_t, grad, prev_checkpoint=None)
        W_hat, O_hat = compressor.decompress(compressed, prev_checkpoint=None)

        assert set(W_hat.keys()) == set(W_t.keys())
        assert set(O_hat.keys()) == set(O_t.keys())


class TestPredictiveCompressorChain:
    """链式压缩（多个检查点）测试。"""

    def test_chain_two_checkpoints(self):
        """应该压缩两个检查点的链。"""
        torch.manual_seed(42)
        compressor = PredictiveCompressor()

        # 第一个检查点
        W_1 = {"weight": torch.randn(32, 32)}
        O_1 = {
            "weight": {
                "exp_avg": torch.randn(32, 32) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(32, 32)) * 0.01,
            }
        }
        grad_1 = {"weight": torch.randn(32, 32)}

        # 第二个检查点（从第一个演化而来）
        W_2 = {"weight": W_1["weight"] + torch.randn(32, 32) * 0.1}
        O_2 = {
            "weight": {
                "exp_avg": O_1["weight"]["exp_avg"] * 0.9 + torch.randn(32, 32) * 0.01,
                "exp_avg_sq": O_1["weight"]["exp_avg_sq"] * 0.999 + torch.abs(torch.randn(32, 32)) * 0.001,
            }
        }
        grad_2 = {"weight": torch.randn(32, 32)}

        # 压缩第一个检查点
        compressed_1 = compressor.compress(W_1, O_1, grad_1, prev_checkpoint=None)
        W_1_hat, O_1_hat = compressor.decompress(compressed_1, prev_checkpoint=None)

        # 为链式压缩创建 prev_checkpoint
        prev_ckpt = {"W": W_1_hat, "O": O_1_hat}

        # 压缩第二个检查点
        compressed_2 = compressor.compress(W_2, O_2, grad_2, prev_checkpoint=prev_ckpt)
        W_2_hat, O_2_hat = compressor.decompress(compressed_2, prev_checkpoint=prev_ckpt)

        assert W_2_hat["weight"].shape == W_2["weight"].shape

    def test_chain_compression_smaller_than_independent(self):
        """链式压缩应该比独立压缩更小。"""
        torch.manual_seed(42)
        compressor = PredictiveCompressor()

        # 第一个检查点
        W_1 = {"weight": torch.randn(64, 64)}
        O_1 = {
            "weight": {
                "exp_avg": torch.randn(64, 64) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(64, 64)) * 0.01,
            }
        }
        grad_1 = {"weight": torch.randn(64, 64)}

        # 第二个检查点（与第一个相比变化很小）
        W_2 = {"weight": W_1["weight"] + torch.randn(64, 64) * 0.01}
        O_2 = {
            "weight": {
                "exp_avg": O_1["weight"]["exp_avg"] * 0.9 + torch.randn(64, 64) * 0.001,
                "exp_avg_sq": O_1["weight"]["exp_avg_sq"] * 0.999 + torch.abs(torch.randn(64, 64)) * 0.0001,
            }
        }
        grad_2 = {"weight": torch.randn(64, 64)}

        # 压缩第一个检查点
        compressed_1 = compressor.compress(W_1, O_1, grad_1, prev_checkpoint=None)
        W_1_hat, O_1_hat = compressor.decompress(compressed_1, prev_checkpoint=None)

        # 链式压缩
        prev_ckpt = {"W": W_1_hat, "O": O_1_hat}
        compressed_2_chain = compressor.compress(W_2, O_2, grad_2, prev_checkpoint=prev_ckpt)

        # 独立压缩
        compressed_2_independent = compressor.compress(W_2, O_2, grad_2, prev_checkpoint=None)

        # 链式压缩应该更小（预测残差更小）
        assert len(compressed_2_chain) <= len(compressed_2_independent)


class TestPredictiveCompressorQuality:
    """压缩质量测试。"""

    def test_reconstruction_error_bounded(self):
        """重建误差应该有界。"""
        torch.manual_seed(42)
        compressor = PredictiveCompressor(PredictiveConfig(min_bits=4, max_bits=8))

        W_t = {"weight": torch.randn(64, 64)}
        O_t = {
            "weight": {
                "exp_avg": torch.randn(64, 64) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(64, 64)) * 0.01,
            }
        }
        grad = {"weight": torch.randn(64, 64)}

        compressed = compressor.compress(W_t, O_t, grad, prev_checkpoint=None)
        W_hat, O_hat = compressor.decompress(compressed, prev_checkpoint=None)

        # 权重重建误差
        mse = torch.mean((W_t["weight"] - W_hat["weight"]) ** 2)
        variance = torch.var(W_t["weight"])
        relative_error = mse / variance

        assert relative_error < 0.3  # 相对误差小于 30%

    def test_compression_ratio(self):
        """应该实现有意义的压缩。"""
        torch.manual_seed(42)
        compressor = PredictiveCompressor(PredictiveConfig(min_bits=4, max_bits=8))

        W_t = {
            "layer1.weight": torch.randn(256, 128),
            "layer2.weight": torch.randn(128, 64),
        }
        O_t = {
            "layer1.weight": {
                "exp_avg": torch.randn(256, 128) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(256, 128)) * 0.01,
            },
            "layer2.weight": {
                "exp_avg": torch.randn(128, 64) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(128, 64)) * 0.01,
            },
        }
        grad = {key: torch.randn_like(W_t[key]) for key in W_t}

        compressed = compressor.compress(W_t, O_t, grad, prev_checkpoint=None)

        # 原始大小: 权重 + 优化器状态
        original_size = sum(t.numel() * 4 for t in W_t.values())
        original_size += sum(
            s["exp_avg"].numel() * 4 + s["exp_avg_sq"].numel() * 4
            for s in O_t.values()
        )

        compression_ratio = original_size / len(compressed)

        # 应该至少实现 2 倍压缩
        assert compression_ratio > 2.0

    def test_higher_bits_lower_error(self):
        """更多位数应该产生更低的重建误差。"""
        torch.manual_seed(42)

        W_t = {"weight": torch.randn(64, 64)}
        O_t = {
            "weight": {
                "exp_avg": torch.randn(64, 64) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(64, 64)) * 0.01,
            }
        }
        grad = {"weight": torch.randn(64, 64)}

        # 低位数
        compressor_low = PredictiveCompressor(PredictiveConfig(min_bits=2, max_bits=4))
        compressed_low = compressor_low.compress(W_t, O_t, grad, prev_checkpoint=None)
        W_hat_low, _ = compressor_low.decompress(compressed_low, prev_checkpoint=None)
        mse_low = torch.mean((W_t["weight"] - W_hat_low["weight"]) ** 2)

        # 高位数
        compressor_high = PredictiveCompressor(PredictiveConfig(min_bits=6, max_bits=8))
        compressed_high = compressor_high.compress(W_t, O_t, grad, prev_checkpoint=None)
        W_hat_high, _ = compressor_high.decompress(compressed_high, prev_checkpoint=None)
        mse_high = torch.mean((W_t["weight"] - W_hat_high["weight"]) ** 2)

        assert mse_high < mse_low


class TestPredictiveCompressorEdgeCases:
    """边界情况测试。"""

    def test_empty_optimizer_state(self):
        """应该处理空优化器状态。"""
        torch.manual_seed(42)
        compressor = PredictiveCompressor()

        W_t = {"weight": torch.randn(32, 32)}
        O_t = {}  # 空优化器状态
        grad = {"weight": torch.randn(32, 32)}

        compressed = compressor.compress(W_t, O_t, grad, prev_checkpoint=None)
        W_hat, O_hat = compressor.decompress(compressed, prev_checkpoint=None)

        assert W_hat["weight"].shape == W_t["weight"].shape

    def test_missing_gradient(self):
        """应该处理缺失的梯度（使用零）。"""
        torch.manual_seed(42)
        compressor = PredictiveCompressor()

        W_t = {"weight": torch.randn(32, 32)}
        O_t = {
            "weight": {
                "exp_avg": torch.randn(32, 32) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(32, 32)) * 0.01,
            }
        }
        grad = {}  # 空梯度

        compressed = compressor.compress(W_t, O_t, grad, prev_checkpoint=None)
        W_hat, O_hat = compressor.decompress(compressed, prev_checkpoint=None)

        assert W_hat["weight"].shape == W_t["weight"].shape

    def test_1d_tensor(self):
        """应该处理一维张量（偏置）。"""
        torch.manual_seed(42)
        compressor = PredictiveCompressor()

        W_t = {"bias": torch.randn(64)}
        O_t = {
            "bias": {
                "exp_avg": torch.randn(64) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(64)) * 0.01,
            }
        }
        grad = {"bias": torch.randn(64)}

        compressed = compressor.compress(W_t, O_t, grad, prev_checkpoint=None)
        W_hat, O_hat = compressor.decompress(compressed, prev_checkpoint=None)

        assert W_hat["bias"].shape == W_t["bias"].shape
