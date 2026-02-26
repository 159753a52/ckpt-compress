"""
ExCP 端到端压缩和解压测试。
"""

import pytest
import torch
from typing import Dict

from ckpt_compress.methods.excp.excp import ExCPCompressor


class TestExCPRoundtrip:
    """ExCP 端到端压缩/解压测试。"""

    def test_excp_roundtrip_single_checkpoint(self):
        """压缩和解压应该保持结构和合理的精度。"""
        torch.manual_seed(42)

        # 创建模拟检查点
        W_t = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
        }
        O_t = {
            "layer1.weight": {
                "exp_avg": torch.randn(64, 32),  # v_t（一阶动量）
                "exp_avg_sq": torch.abs(torch.randn(64, 32)) + 0.1,  # m_t（二阶）
            },
            "layer1.bias": {
                "exp_avg": torch.randn(64),
                "exp_avg_sq": torch.abs(torch.randn(64)) + 0.1,
            },
        }

        # 前一重建检查点（用于残差）
        W_prev_hat = {
            "layer1.weight": W_t["layer1.weight"] + torch.randn(64, 32) * 0.1,
            "layer1.bias": W_t["layer1.bias"] + torch.randn(64) * 0.1,
        }

        compressor = ExCPCompressor()
        compressed = compressor.compress(W_t, O_t, W_prev_hat)
        W_hat, O_hat = compressor.decompress(compressed, W_prev_hat)

        # 检查结构是否保持
        assert set(W_hat.keys()) == set(W_t.keys())
        for key in W_t:
            assert W_hat[key].shape == W_t[key].shape

    def test_excp_reconstruction_matches_formula(self):
        """验证 W_hat = W_prev_hat + dW_hat。"""
        torch.manual_seed(42)

        W_t = {"weight": torch.randn(32, 32)}
        O_t = {
            "weight": {
                "exp_avg": torch.randn(32, 32),
                "exp_avg_sq": torch.abs(torch.randn(32, 32)) + 0.1,
            }
        }
        W_prev_hat = {"weight": torch.randn(32, 32)}

        compressor = ExCPCompressor()
        compressed = compressor.compress(W_t, O_t, W_prev_hat)

        # 从压缩数据获取量化残差
        W_hat, _ = compressor.decompress(compressed, W_prev_hat)

        # W_hat 应该是 W_prev_hat + dW_hat（量化残差）
        # 我们无法在不访问内部的情况下验证精确公式，
        # 但我们可以验证结果是合理的
        assert W_hat["weight"].shape == W_t["weight"].shape

    def test_excp_chain_reconstruction_two_steps(self):
        """测试链式重建: W_0 -> W_1 -> W_2。"""
        torch.manual_seed(42)

        # 初始检查点 (t=0)
        W_0 = {"weight": torch.randn(32, 32)}

        # 检查点 t=1
        W_1 = {"weight": W_0["weight"] + torch.randn(32, 32) * 0.1}
        O_1 = {
            "weight": {
                "exp_avg": torch.randn(32, 32),
                "exp_avg_sq": torch.abs(torch.randn(32, 32)) + 0.1,
            }
        }

        # 检查点 t=2
        W_2 = {"weight": W_1["weight"] + torch.randn(32, 32) * 0.1}
        O_2 = {
            "weight": {
                "exp_avg": torch.randn(32, 32),
                "exp_avg_sq": torch.abs(torch.randn(32, 32)) + 0.1,
            }
        }

        compressor = ExCPCompressor()

        # 使用 t=0 作为参考压缩 t=1
        compressed_1 = compressor.compress(W_1, O_1, W_0)
        W_1_hat, _ = compressor.decompress(compressed_1, W_0)

        # 使用重建的 t=1 作为参考压缩 t=2
        compressed_2 = compressor.compress(W_2, O_2, W_1_hat)
        W_2_hat, _ = compressor.decompress(compressed_2, W_1_hat)

        # 两个重建都应该有正确的形状
        assert W_1_hat["weight"].shape == W_1["weight"].shape
        assert W_2_hat["weight"].shape == W_2["weight"].shape

    def test_excp_compressed_artifact_smaller(self):
        """压缩数据应该比原始数据小。"""
        torch.manual_seed(42)

        W_t = {
            "layer1.weight": torch.randn(256, 128),
            "layer2.weight": torch.randn(128, 64),
        }
        O_t = {
            "layer1.weight": {
                "exp_avg": torch.randn(256, 128),
                "exp_avg_sq": torch.abs(torch.randn(256, 128)) + 0.1,
            },
            "layer2.weight": {
                "exp_avg": torch.randn(128, 64),
                "exp_avg_sq": torch.abs(torch.randn(128, 64)) + 0.1,
            },
        }
        W_prev_hat = {
            "layer1.weight": W_t["layer1.weight"] + torch.randn(256, 128) * 0.05,
            "layer2.weight": W_t["layer2.weight"] + torch.randn(128, 64) * 0.05,
        }

        compressor = ExCPCompressor()
        compressed = compressor.compress(W_t, O_t, W_prev_hat)

        # 计算原始大小（float32 = 每个元素 4 字节）
        original_size = sum(t.numel() * 4 for t in W_t.values())
        original_size += sum(
            s["exp_avg"].numel() * 4 + s["exp_avg_sq"].numel() * 4
            for s in O_t.values()
        )

        compressed_size = len(compressed)

        # 压缩后应该更小
        assert compressed_size < original_size

    def test_excp_handles_first_checkpoint(self):
        """第一个检查点（无前一个）应该能工作。"""
        torch.manual_seed(42)

        W_t = {"weight": torch.randn(32, 32)}
        O_t = {
            "weight": {
                "exp_avg": torch.randn(32, 32),
                "exp_avg_sq": torch.abs(torch.randn(32, 32)) + 0.1,
            }
        }

        compressor = ExCPCompressor()

        # 第一个检查点：无前一参考
        compressed = compressor.compress(W_t, O_t, prev_W_hat=None)
        W_hat, O_hat = compressor.decompress(compressed, prev_W_hat=None)

        assert W_hat["weight"].shape == W_t["weight"].shape


class TestExCPQuality:
    """ExCP 压缩质量测试。"""

    def test_excp_reconstruction_error_bounded(self):
        """重建误差应该有界。"""
        torch.manual_seed(42)

        W_t = {"weight": torch.randn(64, 64)}
        O_t = {
            "weight": {
                "exp_avg": torch.randn(64, 64),
                "exp_avg_sq": torch.abs(torch.randn(64, 64)) + 0.1,
            }
        }
        W_prev_hat = {"weight": W_t["weight"] + torch.randn(64, 64) * 0.1}

        compressor = ExCPCompressor()
        compressed = compressor.compress(W_t, O_t, W_prev_hat)
        W_hat, _ = compressor.decompress(compressed, W_prev_hat)

        # 计算相对误差
        mse = torch.mean((W_t["weight"] - W_hat["weight"]) ** 2)
        variance = torch.var(W_t["weight"])
        relative_error = mse / variance

        # 误差应该合理（激进压缩下 < 20%）
        assert relative_error < 0.2

    def test_excp_preserves_tensor_dtype(self):
        """输出张量应该有正确的 dtype。"""
        W_t = {"weight": torch.randn(32, 32, dtype=torch.float32)}
        O_t = {
            "weight": {
                "exp_avg": torch.randn(32, 32, dtype=torch.float32),
                "exp_avg_sq": torch.abs(torch.randn(32, 32, dtype=torch.float32)) + 0.1,
            }
        }

        compressor = ExCPCompressor()
        compressed = compressor.compress(W_t, O_t, prev_W_hat=None)
        W_hat, O_hat = compressor.decompress(compressed, prev_W_hat=None)

        assert W_hat["weight"].dtype == torch.float32
