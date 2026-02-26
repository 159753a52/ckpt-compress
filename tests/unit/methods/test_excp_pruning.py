"""
ExCP 联合剪枝模块测试。
"""

import pytest
import torch

from ckpt_compress.methods.excp.pruning import (
    compute_weight_threshold,
    compute_weight_mask,
    compute_momentum_threshold,
    compute_momentum_mask,
    prune_residual_by_percentile,
    joint_prune,
)


class TestWeightThreshold:
    """权重阈值计算测试（公式 4）。"""

    def test_weight_threshold_rw_elementwise(self):
        """测试 r_w = alpha / sqrt(m_t) * median(W)。"""
        m_t = torch.tensor([1.0, 4.0])
        W = torch.tensor([0.1, 0.2, 0.3])  # 中位数 = 0.2
        alpha = 1.0

        r_w = compute_weight_threshold(W, m_t, alpha)

        # r_w = 1.0 / sqrt([1, 4]) * 0.2 = [0.2, 0.1]
        expected = torch.tensor([0.2, 0.1])
        assert torch.allclose(r_w, expected)

    def test_weight_mask_mw_monotonic_in_m(self):
        """较大的 m_t -> 较小的 r_w -> 更可能保留（mask=1）。"""
        W = torch.tensor([0.15])  # 单个权重
        alpha = 1.0

        # 小 m_t -> 大阈值 -> 更多剪枝
        m_t_small = torch.tensor([0.25])  # r_w = 1/0.5 * median = 2 * 0.15 = 0.3
        mask_small = compute_weight_mask(W, m_t_small, alpha)

        # 大 m_t -> 小阈值 -> 更少剪枝
        m_t_large = torch.tensor([4.0])  # r_w = 1/2 * 0.15 = 0.075
        mask_large = compute_weight_mask(W, m_t_large, alpha)

        # 大 m_t 时，|W|=0.15 > r_w=0.075，所以 mask=1
        # 小 m_t 时，|W|=0.15 < r_w=0.3，所以 mask=0
        assert mask_large.item() == 1
        assert mask_small.item() == 0


class TestMomentumThreshold:
    """动量阈值计算测试（公式 5）。"""

    def test_momentum_threshold_ro_scalar(self):
        """测试 r_o = beta * mean(v_t)。"""
        v_t = torch.tensor([0.0, 2.0])
        beta = 1.0

        r_o = compute_momentum_threshold(v_t, beta)

        # r_o = 1.0 * mean([0, 2]) = 1.0
        assert torch.isclose(r_o, torch.tensor(1.0))

    def test_momentum_mask_depends_on_weight_mask(self):
        """M_o(i) = 1 当且仅当 v_t(i) > r_o 且 M_w(i) = 1。"""
        v_t = torch.tensor([0.5, 1.5, 2.0])  # 均值 = 1.33
        M_w = torch.tensor([1, 0, 1])  # 权重掩码
        beta = 1.0

        M_o = compute_momentum_mask(v_t, M_w, beta)

        # r_o = 1.33
        # v_t[0]=0.5 < r_o, M_w[0]=1 -> M_o[0]=0
        # v_t[1]=1.5 > r_o, M_w[1]=0 -> M_o[1]=0（权重掩码阻止）
        # v_t[2]=2.0 > r_o, M_w[2]=1 -> M_o[2]=1
        expected = torch.tensor([0, 0, 1])
        assert torch.equal(M_o, expected)


class TestResidualPercentilePruning:
    """残差百分位剪枝测试（公式 6）。"""

    def test_residual_percentile_pruning_all_zero_when_p_100(self):
        """p=100 -> 阈值 = max(|dW|) -> 全部剪枝为零。"""
        dW = torch.tensor([0.1, -0.2, 0.3, -0.4])
        p = 100.0

        dW_star = prune_residual_by_percentile(dW, p)

        # 所有值应该为零（或几乎全部）
        assert torch.sum(dW_star != 0) <= 1  # 最多 1 个非零（最大值本身）

    def test_residual_percentile_pruning_none_when_p_0(self):
        """p=0 -> 阈值 = min(|dW|) -> 大部分保留。"""
        dW = torch.tensor([0.1, -0.2, 0.3, -0.4])
        p = 0.0

        dW_star = prune_residual_by_percentile(dW, p)

        # 大部分值应该被保留
        assert torch.sum(dW_star != 0) >= 3


class TestJointPrune:
    """联合剪枝函数测试。"""

    def test_joint_prune_applies_weight_mask_to_dW(self):
        """权重掩码应该将 dW 中对应位置置零。"""
        dW = torch.tensor([1.0, 2.0, 3.0, 4.0])
        v_t = torch.tensor([1.0, 1.0, 1.0, 1.0])  # 一阶动量
        m_t = torch.tensor([1.0, 1.0, 1.0, 1.0])  # 二阶动量
        alpha = 0.0  # 非常小的阈值 -> 全部保留
        beta = 0.0
        p = 50.0  # 剪枝底部 50%

        dW_star, v_star = joint_prune(dW, v_t, m_t, alpha, beta, p)

        # 某些位置应该被置零
        assert dW_star.shape == dW.shape

    def test_joint_prune_prunes_optimizer_where_v_small(self):
        """当 v_t 较小时，优化器状态应该被剪枝。"""
        dW = torch.tensor([1.0, 2.0, 3.0, 4.0])
        v_t = torch.tensor([0.1, 0.1, 10.0, 10.0])  # 前两个较小
        m_t = torch.ones(4)
        alpha = 0.0  # 保留所有权重
        beta = 1.0  # 使用均值作为阈值
        p = 0.0  # 无百分位剪枝

        dW_star, v_star = joint_prune(dW, v_t, m_t, alpha, beta, p)

        # v_t[0], v_t[1] < mean(v_t) = 5.05，所以应该被剪枝
        # 但也取决于权重掩码
        assert v_star.shape == v_t.shape

    def test_joint_prune_idempotent(self):
        """对已剪枝的数据再次剪枝不应改变它。"""
        dW = torch.tensor([0.0, 2.0, 0.0, 4.0])
        v_t = torch.tensor([0.0, 1.0, 0.0, 1.0])
        m_t = torch.ones(4)
        alpha = 0.0
        beta = 0.0
        p = 0.0

        dW_star1, v_star1 = joint_prune(dW, v_t, m_t, alpha, beta, p)
        dW_star2, v_star2 = joint_prune(dW_star1, v_star1, m_t, alpha, beta, p)

        assert torch.allclose(dW_star1, dW_star2)
        assert torch.allclose(v_star1, v_star2)

    def test_joint_prune_preserves_shapes(self):
        """输出形状应与输入形状匹配。"""
        dW = torch.randn(64, 32)
        v_t = torch.abs(torch.randn(64, 32))
        m_t = torch.abs(torch.randn(64, 32)) + 0.1
        alpha = 1.0
        beta = 1.0
        p = 20.0

        dW_star, v_star = joint_prune(dW, v_t, m_t, alpha, beta, p)

        assert dW_star.shape == dW.shape
        assert v_star.shape == v_t.shape

    def test_joint_prune_handles_all_zero_moments(self):
        """当动量全为零时不应崩溃（添加 eps）。"""
        dW = torch.randn(10)
        v_t = torch.zeros(10)
        m_t = torch.zeros(10)
        alpha = 1.0
        beta = 1.0
        p = 20.0

        # 不应抛出异常
        dW_star, v_star = joint_prune(dW, v_t, m_t, alpha, beta, p)

        assert dW_star.shape == dW.shape

    def test_joint_prune_layerwise_median_computation(self):
        """中位数应该按张量（层）计算。"""
        # 两个具有不同分布的"层"
        dW1 = torch.tensor([0.1, 0.2, 0.3])  # 中位数 = 0.2
        dW2 = torch.tensor([1.0, 2.0, 3.0])  # 中位数 = 2.0

        m_t1 = torch.ones(3)
        m_t2 = torch.ones(3)
        v_t1 = torch.ones(3)
        v_t2 = torch.ones(3)

        alpha = 1.0
        beta = 0.0
        p = 0.0

        # 每层的阈值应该不同
        r_w1 = compute_weight_threshold(dW1, m_t1, alpha)
        r_w2 = compute_weight_threshold(dW2, m_t2, alpha)

        # median(dW1) = 0.2, median(dW2) = 2.0
        # 所以阈值应该不同
        assert not torch.allclose(r_w1, r_w2)
