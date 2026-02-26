"""
Inshrinkerator 分区模块测试（保护/剪枝/量化）。
"""

import pytest
import torch

from ckpt_compress.methods.inshrinkerator.partition import (
    partition,
    compute_thresholds,
    PartitionConfig,
    PartitionResult,
)


class TestPartitionBasic:
    """参数分区基本测试。"""

    def test_partition_protect_top_fraction_both_metrics(self):
        """高幅度或高敏感度应被保护。"""
        # 创建一些高幅度、一些高敏感度的值
        values = torch.tensor([10.0, 0.1, 5.0, 0.2])  # [高幅度, 低幅度, 中幅度, 低幅度]
        grad = torch.tensor([0.01, 10.0, 0.1, 0.1])   # [低梯度, 高梯度, 低梯度, 低梯度]

        config = PartitionConfig(
            protect_fraction=0.5,  # 保护前 50%
            prune_fraction=0.0,
            n_bins=8,
        )

        result = partition(values, grad, config)

        # 索引 0（高幅度）和索引 1（高敏感度）应被保护
        assert result.protect_mask[0] == 1 or result.protect_mask[1] == 1

    def test_partition_prune_bottom_fraction(self):
        """按幅度的底部比例应被剪枝。"""
        values = torch.tensor([0.01, 0.02, 1.0, 2.0, 3.0])
        grad = torch.zeros(5)

        config = PartitionConfig(
            protect_fraction=0.0,
            prune_fraction=0.4,  # 剪枝底部 40%
            n_bins=8,
        )

        result = partition(values, grad, config)

        # 前两个（最小的）应被剪枝
        assert result.prune_mask[0] == 1
        assert result.prune_mask[1] == 1

    def test_partition_quantize_remaining(self):
        """未被剪枝或保护的值应被量化。"""
        values = torch.tensor([0.01, 0.5, 1.0, 10.0])
        grad = torch.zeros(4)

        config = PartitionConfig(
            protect_fraction=0.25,  # 保护前 25%
            prune_fraction=0.25,    # 剪枝底部 25%
            n_bins=8,
        )

        result = partition(values, grad, config)

        # 中间值应被量化
        quantize_count = result.quantize_mask.sum().item()
        assert quantize_count >= 1  # 至少有一些应被量化


class TestPartitionMasks:
    """分区掩码属性测试。"""

    def test_masks_are_disjoint(self):
        """保护、剪枝和量化掩码不应重叠。"""
        values = torch.randn(100).abs()
        grad = torch.randn(100).abs()

        config = PartitionConfig(
            protect_fraction=0.1,
            prune_fraction=0.2,
            n_bins=8,
        )

        result = partition(values, grad, config)

        # 检查无重叠
        overlap_prot_prune = (result.protect_mask & result.prune_mask).sum()
        overlap_prot_quant = (result.protect_mask & result.quantize_mask).sum()
        overlap_prune_quant = (result.prune_mask & result.quantize_mask).sum()

        assert overlap_prot_prune == 0
        assert overlap_prot_quant == 0
        assert overlap_prune_quant == 0

    def test_masks_cover_all(self):
        """所有元素应恰好在一个分区中。"""
        values = torch.randn(100).abs()
        grad = torch.randn(100).abs()

        config = PartitionConfig(
            protect_fraction=0.1,
            prune_fraction=0.2,
            n_bins=8,
        )

        result = partition(values, grad, config)

        total = result.protect_mask + result.prune_mask + result.quantize_mask
        assert torch.all(total == 1)

    def test_prune_sets_exact_zero(self):
        """被剪枝的值应精确为零。"""
        values = torch.tensor([0.01, 0.02, 1.0, 2.0])
        grad = torch.zeros(4)

        config = PartitionConfig(
            protect_fraction=0.0,
            prune_fraction=0.5,
            n_bins=8,
        )

        result = partition(values, grad, config)
        pruned_values = result.get_pruned_values(values)

        # 被剪枝的位置应为零
        assert torch.all(pruned_values[result.prune_mask == 1] == 0)


class TestPartitionEdgeCases:
    """分区边界情况测试。"""

    def test_fraction_edge_case_zero_protect(self):
        """零保护比例不应保护任何内容。"""
        values = torch.randn(100).abs()
        grad = torch.randn(100).abs()

        config = PartitionConfig(
            protect_fraction=0.0,
            prune_fraction=0.2,
            n_bins=8,
        )

        result = partition(values, grad, config)

        assert result.protect_mask.sum() == 0

    def test_fraction_edge_case_zero_prune(self):
        """零剪枝比例不应剪枝任何内容。"""
        values = torch.randn(100).abs()
        grad = torch.randn(100).abs()

        config = PartitionConfig(
            protect_fraction=0.1,
            prune_fraction=0.0,
            n_bins=8,
        )

        result = partition(values, grad, config)

        assert result.prune_mask.sum() == 0

    def test_fraction_edge_case_full_prune(self):
        """完全剪枝比例应剪枝所有未保护的内容。"""
        values = torch.randn(100).abs()
        grad = torch.randn(100).abs()

        config = PartitionConfig(
            protect_fraction=0.1,
            prune_fraction=0.9,  # 剪枝 90%
            n_bins=8,
        )

        result = partition(values, grad, config)

        # 大部分应被剪枝
        assert result.prune_mask.sum() >= 80

    def test_empty_tensor(self):
        """应处理空张量。"""
        values = torch.tensor([])
        grad = torch.tensor([])

        config = PartitionConfig(
            protect_fraction=0.1,
            prune_fraction=0.2,
            n_bins=8,
        )

        result = partition(values, grad, config)

        assert result.protect_mask.numel() == 0


class TestProtectedStorage:
    """受保护值存储测试。"""

    def test_protect_stored_in_bfloat16(self):
        """受保护的值应以 bfloat16 存储。"""
        values = torch.tensor([10.0, 0.1, 0.2, 0.3], dtype=torch.float32)
        grad = torch.zeros(4)

        config = PartitionConfig(
            protect_fraction=0.25,
            prune_fraction=0.0,
            n_bins=8,
        )

        result = partition(values, grad, config)
        protected = result.get_protected_values(values)

        # 受保护的值应为 bfloat16
        assert protected.dtype == torch.bfloat16
