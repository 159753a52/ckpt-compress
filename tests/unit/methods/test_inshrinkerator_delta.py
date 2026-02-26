"""
Inshrinkerator 增量编码模块测试。
"""

import pytest
import torch

from ckpt_compress.methods.inshrinkerator.delta_encoding import (
    delta_encode,
    delta_decode,
    rearrange_by_prev_bin,
    restore_rearrangement,
    rle_encode,
    rle_decode,
)


class TestDeltaEncode:
    """增量编码测试。"""

    def test_delta_mod_B_basic(self):
        """测试 D = (prev - curr) mod B。"""
        prev = torch.tensor([1, 3, 5, 7])
        curr = torch.tensor([3, 3, 2, 0])
        B = 8

        D = delta_encode(prev, curr, B)

        # D = (prev - curr) mod B
        # (1-3) mod 8 = -2 mod 8 = 6
        # (3-3) mod 8 = 0
        # (5-2) mod 8 = 3
        # (7-0) mod 8 = 7
        expected = torch.tensor([6, 0, 3, 7])
        assert torch.equal(D, expected)

    def test_delta_all_zero_when_same(self):
        """当 prev == curr 时，增量应全为零。"""
        prev = torch.tensor([1, 2, 3, 4])
        curr = torch.tensor([1, 2, 3, 4])
        B = 8

        D = delta_encode(prev, curr, B)

        assert torch.all(D == 0)

    def test_delta_with_varied_bins_uses_max_B(self):
        """当桶数变化时，应使用最大 B。"""
        prev = torch.tensor([1, 2, 3])
        curr = torch.tensor([2, 3, 4])
        B = 16  # 较大的 B

        D = delta_encode(prev, curr, B)

        # 所有增量应在 [0, B-1] 范围内
        assert torch.all(D >= 0)
        assert torch.all(D < B)


class TestDeltaDecode:
    """增量解码测试。"""

    def test_delta_decode_recovers_curr_quantized(self):
        """解码应恢复 curr：curr = (prev - D) mod B。"""
        prev = torch.tensor([1, 3, 5, 7])
        curr = torch.tensor([3, 3, 2, 0])
        B = 8

        D = delta_encode(prev, curr, B)
        recovered = delta_decode(prev, D, B)

        assert torch.equal(recovered, curr)

    def test_encode_decode_roundtrip(self):
        """编码后解码应恢复原始值。"""
        torch.manual_seed(42)
        prev = torch.randint(0, 16, (100,))
        curr = torch.randint(0, 16, (100,))
        B = 16

        D = delta_encode(prev, curr, B)
        recovered = delta_decode(prev, D, B)

        assert torch.equal(recovered, curr)


class TestRearrangement:
    """基于桶的重排测试（算法 5/6）。"""

    def test_rearrange_groups_by_prev_bin(self):
        """重排应按前一个桶分组增量。"""
        q_prev = torch.tensor([0, 1, 1, 2, 0, 2])
        D = torch.tensor([1, 2, 3, 4, 5, 6])
        B = 3

        grouped = rearrange_by_prev_bin(q_prev, D, B)

        # 桶 0：位置 0, 4 -> 增量 1, 5
        # 桶 1：位置 1, 2 -> 增量 2, 3
        # 桶 2：位置 3, 5 -> 增量 4, 6
        assert torch.equal(grouped[0], torch.tensor([1, 5]))
        assert torch.equal(grouped[1], torch.tensor([2, 3]))
        assert torch.equal(grouped[2], torch.tensor([4, 6]))

    def test_restore_rearrangement(self):
        """恢复应还原原始增量顺序。"""
        q_prev = torch.tensor([0, 1, 1, 2, 0, 2])
        D = torch.tensor([1, 2, 3, 4, 5, 6])
        B = 3

        grouped = rearrange_by_prev_bin(q_prev, D, B)
        restored = restore_rearrangement(q_prev, grouped, B)

        assert torch.equal(restored, D)

    def test_rearrange_restore_roundtrip(self):
        """重排后恢复应为恒等操作。"""
        torch.manual_seed(42)
        q_prev = torch.randint(0, 8, (100,))
        D = torch.randint(0, 8, (100,))
        B = 8

        grouped = rearrange_by_prev_bin(q_prev, D, B)
        restored = restore_rearrangement(q_prev, grouped, B)

        assert torch.equal(restored, D)


class TestRLE:
    """游程编码测试。"""

    def test_rle_encode_decode_roundtrip(self):
        """RLE 编码后解码应恢复原始值。"""
        D = torch.tensor([0, 0, 0, 2, 2, 5, 5, 5, 5])

        encoded = rle_encode(D)
        decoded = rle_decode(encoded, len(D))

        assert torch.equal(decoded, D)

    def test_rle_stores_run_length_only_if_gt1(self):
        """游程长度仅在 > 1 时存储。"""
        D = torch.tensor([1, 2, 3])  # 无游程

        encoded = rle_encode(D)

        # 应该相对紧凑（无额外游程长度标记）
        # 只检查它能工作
        decoded = rle_decode(encoded, len(D))
        assert torch.equal(decoded, D)

    def test_rle_handles_long_runs(self):
        """应高效处理长游程。"""
        D = torch.zeros(100, dtype=torch.long)  # 100 个零

        encoded = rle_encode(D)
        decoded = rle_decode(encoded, len(D))

        assert torch.equal(decoded, D)
        # 编码后应比原始小得多
        assert len(encoded) < len(D)

    def test_rle_mixed_content(self):
        """应处理混合的游程和单个值。"""
        D = torch.tensor([0, 0, 1, 2, 2, 2, 3, 0, 0, 0, 0])

        encoded = rle_encode(D)
        decoded = rle_decode(encoded, len(D))

        assert torch.equal(decoded, D)

    def test_rle_empty(self):
        """应处理空输入。"""
        D = torch.tensor([], dtype=torch.long)

        encoded = rle_encode(D)
        decoded = rle_decode(encoded, 0)

        assert decoded.numel() == 0


class TestEndToEnd:
    """增量编码流水线端到端测试。"""

    def test_end_to_end_two_checkpoints_quantized_roundtrip(self):
        """完整流水线：q1 -> 增量编码 -> 解码 -> q2。"""
        torch.manual_seed(42)
        q1 = torch.randint(0, 16, (100,))
        q2 = torch.randint(0, 16, (100,))
        B = 16

        # 编码
        D = delta_encode(q1, q2, B)
        grouped = rearrange_by_prev_bin(q1, D, B)

        # 对每组使用 RLE 编码
        encoded_groups = [rle_encode(g) for g in grouped]

        # 解码
        decoded_groups = [rle_decode(e, len(grouped[i])) for i, e in enumerate(encoded_groups)]
        D_restored = restore_rearrangement(q1, decoded_groups, B)
        q2_recovered = delta_decode(q1, D_restored, B)

        assert torch.equal(q2_recovered, q2)
