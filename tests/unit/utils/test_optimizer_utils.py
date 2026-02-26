"""
优化器状态工具测试。

TDD：先写测试，再实现。

关键功能：
- 从 PyTorch 优化器提取优化器状态
- 从压缩数据恢复优化器状态
- 支持 Adam 优化器
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from ckpt_compress.utils.optimizer_utils import (
    extract_optimizer_state,
    restore_optimizer_state,
    create_optimizer_with_state,
)


class SimpleModel(nn.Module):
    """用于测试的简单模型。"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestExtractOptimizerState:
    """提取优化器状态测试。"""

    def test_extract_adam_state(self):
        """应从 Adam 提取 exp_avg 和 exp_avg_sq。"""
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 运行几步以填充优化器状态
        for _ in range(3):
            x = torch.randn(4, 10)
            loss = model(x).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = extract_optimizer_state(model, optimizer)

        # 应有每个参数的状态
        assert "fc1.weight" in state
        assert "fc1.bias" in state
        assert "fc2.weight" in state
        assert "fc2.bias" in state

        # 每个应有 exp_avg 和 exp_avg_sq
        assert "exp_avg" in state["fc1.weight"]
        assert "exp_avg_sq" in state["fc1.weight"]

    def test_extract_state_shapes_match(self):
        """提取的状态形状应与参数形状匹配。"""
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 运行一步
        x = torch.randn(4, 10)
        loss = model(x).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = extract_optimizer_state(model, optimizer)

        # 检查形状
        assert state["fc1.weight"]["exp_avg"].shape == (20, 10)
        assert state["fc1.bias"]["exp_avg"].shape == (20,)

    def test_extract_empty_state_before_step(self):
        """任何步骤之前，优化器状态应为空。"""
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        state = extract_optimizer_state(model, optimizer)

        # 状态应为空或没有 exp_avg
        assert len(state) == 0 or all(
            "exp_avg" not in s for s in state.values()
        )


class TestRestoreOptimizerState:
    """恢复优化器状态测试。"""

    def test_restore_adam_state(self):
        """应将 exp_avg 和 exp_avg_sq 恢复到 Adam。"""
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 创建假状态
        state = {
            "fc1.weight": {
                "exp_avg": torch.randn(20, 10) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(20, 10)) * 0.01,
            },
            "fc1.bias": {
                "exp_avg": torch.randn(20) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(20)) * 0.01,
            },
            "fc2.weight": {
                "exp_avg": torch.randn(5, 20) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(5, 20)) * 0.01,
            },
            "fc2.bias": {
                "exp_avg": torch.randn(5) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(5)) * 0.01,
            },
        }

        restore_optimizer_state(model, optimizer, state)

        # 验证状态已恢复
        extracted = extract_optimizer_state(model, optimizer)
        assert torch.allclose(
            extracted["fc1.weight"]["exp_avg"],
            state["fc1.weight"]["exp_avg"]
        )

    def test_restore_preserves_lr(self):
        """恢复状态应保持学习率。"""
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        state = {
            "fc1.weight": {
                "exp_avg": torch.randn(20, 10) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(20, 10)) * 0.01,
            },
            "fc1.bias": {
                "exp_avg": torch.randn(20) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(20)) * 0.01,
            },
            "fc2.weight": {
                "exp_avg": torch.randn(5, 20) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(5, 20)) * 0.01,
            },
            "fc2.bias": {
                "exp_avg": torch.randn(5) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(5)) * 0.01,
            },
        }

        restore_optimizer_state(model, optimizer, state)

        # 学习率应仍为 0.01
        assert optimizer.param_groups[0]["lr"] == 0.01


class TestCreateOptimizerWithState:
    """创建带预加载状态的优化器测试。"""

    def test_create_adam_with_state(self):
        """应创建带恢复状态的 Adam 优化器。"""
        model = SimpleModel()

        state = {
            "fc1.weight": {
                "exp_avg": torch.randn(20, 10) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(20, 10)) * 0.01,
            },
            "fc1.bias": {
                "exp_avg": torch.randn(20) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(20)) * 0.01,
            },
            "fc2.weight": {
                "exp_avg": torch.randn(5, 20) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(5, 20)) * 0.01,
            },
            "fc2.bias": {
                "exp_avg": torch.randn(5) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(5)) * 0.01,
            },
        }

        optimizer = create_optimizer_with_state(
            model, state, lr=0.001, optimizer_type="adam"
        )

        assert isinstance(optimizer, optim.Adam)

        # 验证状态已加载
        extracted = extract_optimizer_state(model, optimizer)
        assert torch.allclose(
            extracted["fc1.weight"]["exp_avg"],
            state["fc1.weight"]["exp_avg"]
        )

    def test_create_with_empty_state(self):
        """即使状态为空也应创建优化器。"""
        model = SimpleModel()

        optimizer = create_optimizer_with_state(
            model, {}, lr=0.001, optimizer_type="adam"
        )

        assert isinstance(optimizer, optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 0.001

    def test_training_continues_correctly(self):
        """状态恢复后训练应正确继续。"""
        torch.manual_seed(42)
        model1 = SimpleModel()
        optimizer1 = optim.Adam(model1.parameters(), lr=0.001)

        # 训练几步
        for _ in range(5):
            x = torch.randn(4, 10)
            loss = model1(x).sum()
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

        # 提取状态
        state = extract_optimizer_state(model1, optimizer1)
        weights = {k: v.clone() for k, v in model1.state_dict().items()}

        # 创建新模型和带恢复状态的优化器
        torch.manual_seed(42)
        model2 = SimpleModel()
        model2.load_state_dict(weights)
        optimizer2 = create_optimizer_with_state(
            model2, state, lr=0.001, optimizer_type="adam"
        )

        # 继续训练两者
        torch.manual_seed(123)
        x = torch.randn(4, 10)

        # 模型 1
        loss1 = model1(x).sum()
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        # 模型 2
        loss2 = model2(x).sum()
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        # 权重应非常接近（允许小的浮点差异）
        for key in model1.state_dict():
            assert torch.allclose(
                model1.state_dict()[key],
                model2.state_dict()[key],
                atol=1e-4
            )
