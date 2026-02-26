"""
内存高效剪枝评估模块的测试。

使用 TDD 模式开发，测试覆盖率目标 90%+。
"""

import pytest
import torch
import torch.nn as nn
import gc
from typing import Dict, List
from unittest.mock import Mock, patch, MagicMock


class SimpleModel(nn.Module):
    """用于测试的简单模型。"""

    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, labels=None):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        # 返回类似 HuggingFace 模型的输出
        output = Mock()
        output.loss = loss
        output.logits = logits
        return output


class TestMemoryEfficientEvaluator:
    """测试 MemoryEfficientEvaluator 类。"""

    @pytest.fixture
    def model(self):
        """创建测试模型。"""
        return SimpleModel()

    @pytest.fixture
    def weights(self, model):
        """创建测试权重。"""
        return {name: param.data.clone() for name, param in model.named_parameters()}

    @pytest.fixture
    def importance_scores(self, model):
        """创建测试重要性得分。"""
        return {name: torch.rand_like(param) for name, param in model.named_parameters()}

    @pytest.fixture
    def cached_batches(self):
        """创建测试批次数据。"""
        return [
            {
                'input_ids': torch.randn(2, 10),
                'attention_mask': torch.ones(2, 10),
                'labels': torch.randint(0, 5, (2,))
            }
            for _ in range(3)
        ]

    def test_init(self, model):
        """测试初始化。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import MemoryEfficientEvaluator

        evaluator = MemoryEfficientEvaluator(model)
        assert evaluator.model is model
        assert evaluator._original_weights is None

    def test_save_original_weights(self, model):
        """测试保存原始权重。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import MemoryEfficientEvaluator

        evaluator = MemoryEfficientEvaluator(model)
        evaluator.save_original_weights()

        assert evaluator._original_weights is not None
        assert len(evaluator._original_weights) == len(list(model.named_parameters()))

        # 验证权重是克隆的，不是引用
        for name, param in model.named_parameters():
            assert name in evaluator._original_weights
            assert evaluator._original_weights[name] is not param.data
            assert torch.equal(evaluator._original_weights[name], param.data)

    def test_apply_pruned_weights(self, model, weights, importance_scores):
        """测试应用剪枝权重。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import MemoryEfficientEvaluator
        from ckpt_compress.methods.adam_prune.layer_pruning import prune_by_global_sparsity

        evaluator = MemoryEfficientEvaluator(model)
        evaluator.save_original_weights()

        # 剪枝
        pruned_weights, masks = prune_by_global_sparsity(weights, importance_scores, 0.5)

        # 应用剪枝权重
        evaluator.apply_pruned_weights(pruned_weights)

        # 验证权重已更改
        for name, param in model.named_parameters():
            if name in pruned_weights:
                assert torch.equal(param.data, pruned_weights[name])

    def test_restore_original_weights(self, model, weights, importance_scores):
        """测试恢复原始权重。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import MemoryEfficientEvaluator
        from ckpt_compress.methods.adam_prune.layer_pruning import prune_by_global_sparsity

        evaluator = MemoryEfficientEvaluator(model)
        evaluator.save_original_weights()

        # 保存原始权重的副本用于比较
        original_weights_copy = {name: param.data.clone() for name, param in model.named_parameters()}

        # 剪枝并应用
        pruned_weights, masks = prune_by_global_sparsity(weights, importance_scores, 0.5)
        evaluator.apply_pruned_weights(pruned_weights)

        # 恢复
        evaluator.restore_original_weights()

        # 验证权重已恢复
        for name, param in model.named_parameters():
            assert torch.equal(param.data, original_weights_copy[name])

    def test_restore_without_save_raises_error(self, model):
        """测试未保存时恢复会抛出错误。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import MemoryEfficientEvaluator

        evaluator = MemoryEfficientEvaluator(model)

        with pytest.raises(RuntimeError, match="No original weights saved"):
            evaluator.restore_original_weights()

    def test_evaluate_loss(self, model):
        """测试评估损失。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import MemoryEfficientEvaluator

        evaluator = MemoryEfficientEvaluator(model)

        # 创建简单的测试数据
        batches = [
            {
                'input_ids': torch.randn(2, 10),
                'labels': torch.randint(0, 5, (2,))
            }
            for _ in range(3)
        ]

        loss = evaluator.evaluate_loss(batches)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_evaluate_loss_increase(self, model, weights, importance_scores):
        """测试评估损失增量。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import MemoryEfficientEvaluator
        from ckpt_compress.methods.adam_prune.layer_pruning import prune_by_global_sparsity

        evaluator = MemoryEfficientEvaluator(model)
        evaluator.save_original_weights()

        # 创建测试数据
        batches = [
            {
                'input_ids': torch.randn(2, 10),
                'labels': torch.randint(0, 5, (2,))
            }
            for _ in range(3)
        ]

        # 剪枝
        pruned_weights, masks = prune_by_global_sparsity(weights, importance_scores, 0.5)

        # 评估损失增量
        loss_increase = evaluator.evaluate_loss_increase(pruned_weights, batches)

        assert isinstance(loss_increase, float)
        # 剪枝后损失应该增加（或至少不会大幅减少）
        # 由于随机初始化，我们只检查返回值类型

    def test_context_manager(self, model, weights, importance_scores):
        """测试上下文管理器。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import MemoryEfficientEvaluator
        from ckpt_compress.methods.adam_prune.layer_pruning import prune_by_global_sparsity

        evaluator = MemoryEfficientEvaluator(model)

        # 保存原始权重的副本
        original_weights_copy = {name: param.data.clone() for name, param in model.named_parameters()}

        # 剪枝
        pruned_weights, masks = prune_by_global_sparsity(weights, importance_scores, 0.5)

        # 使用上下文管理器
        with evaluator.temporary_weights(pruned_weights):
            # 在上下文中，权重应该是剪枝后的
            for name, param in model.named_parameters():
                if name in pruned_weights:
                    assert torch.equal(param.data, pruned_weights[name])

        # 退出上下文后，权重应该恢复
        for name, param in model.named_parameters():
            assert torch.equal(param.data, original_weights_copy[name])

    def test_cleanup(self, model):
        """测试清理方法。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import MemoryEfficientEvaluator

        evaluator = MemoryEfficientEvaluator(model)
        evaluator.save_original_weights()

        assert evaluator._original_weights is not None

        evaluator.cleanup()

        assert evaluator._original_weights is None


class TestMemoryEfficientDataCollector:
    """测试 MemoryEfficientDataCollector 类。"""

    @pytest.fixture
    def model(self):
        """创建测试模型。"""
        return SimpleModel()

    @pytest.fixture
    def optimizer(self, model):
        """创建测试优化器。"""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # 运行一步以初始化优化器状态
        x = torch.randn(2, 10)
        labels = torch.randint(0, 5, (2,))
        output = model(x, labels)
        output.loss.backward()
        optimizer.step()
        return optimizer

    def test_collect_to_cpu(self, model, optimizer):
        """测试收集数据到 CPU。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import MemoryEfficientDataCollector

        collector = MemoryEfficientDataCollector()
        weights, gradients, exp_avg_sq = collector.collect(model, optimizer, to_cpu=True)

        assert len(weights) > 0
        assert len(gradients) > 0
        assert len(exp_avg_sq) > 0

        # 验证数据在 CPU 上
        for name, tensor in weights.items():
            assert tensor.device.type == 'cpu'
        for name, tensor in gradients.items():
            assert tensor.device.type == 'cpu'
        for name, tensor in exp_avg_sq.items():
            assert tensor.device.type == 'cpu'

    def test_collect_preserves_device(self, model, optimizer):
        """测试收集数据保持设备。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import MemoryEfficientDataCollector

        collector = MemoryEfficientDataCollector()
        weights, gradients, exp_avg_sq = collector.collect(model, optimizer, to_cpu=False)

        # 验证数据设备与模型一致
        model_device = next(model.parameters()).device
        for name, tensor in weights.items():
            assert tensor.device == model_device


class TestForceGarbageCollection:
    """测试强制垃圾回收函数。"""

    def test_force_gc(self):
        """测试强制垃圾回收。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import force_garbage_collection

        # 创建一些临时对象
        temp_tensors = [torch.randn(100, 100) for _ in range(10)]
        del temp_tensors

        # 调用强制垃圾回收
        force_garbage_collection()

        # 验证没有异常
        assert True

    @patch('gc.collect')
    def test_force_gc_calls_gc_collect(self, mock_gc_collect):
        """测试强制垃圾回收调用 gc.collect。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import force_garbage_collection

        force_garbage_collection()

        mock_gc_collect.assert_called()


class TestMemoryEfficientSparsityExperiment:
    """测试内存高效的稀疏度实验函数。"""

    @pytest.fixture
    def model(self):
        """创建测试模型。"""
        return SimpleModel()

    @pytest.fixture
    def weights(self, model):
        """创建测试权重。"""
        return {name: param.data.clone() for name, param in model.named_parameters()}

    @pytest.fixture
    def importance_scores(self, model):
        """创建测试重要性得分。"""
        return {name: torch.rand_like(param) for name, param in model.named_parameters()}

    @pytest.fixture
    def cached_batches(self):
        """创建测试批次数据。"""
        return [
            {
                'input_ids': torch.randn(2, 10),
                'labels': torch.randint(0, 5, (2,))
            }
            for _ in range(3)
        ]

    def test_run_single_sparsity(self, model, weights, importance_scores, cached_batches):
        """测试单个稀疏度实验。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import (
            MemoryEfficientEvaluator,
            run_single_sparsity_experiment
        )

        evaluator = MemoryEfficientEvaluator(model)
        evaluator.save_original_weights()

        baseline_loss = evaluator.evaluate_loss(cached_batches)

        result = run_single_sparsity_experiment(
            evaluator=evaluator,
            weights=weights,
            importance_scores=importance_scores,
            cached_batches=cached_batches,
            sparsity=0.1,
            baseline_loss=baseline_loss
        )

        assert 'sparsity' in result
        assert 'x' in result
        assert 'y' in result
        assert 'sum_pruned_scores' in result
        assert 'actual_loss_increase' in result

        assert result['sparsity'] == 0.1
        assert isinstance(result['x'], float)
        assert isinstance(result['y'], float)

    def test_run_sparsity_grid_memory_efficient(self, model, weights, importance_scores, cached_batches):
        """测试内存高效的稀疏度网格实验。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import (
            run_sparsity_grid_memory_efficient
        )

        sparsities = [0.05, 0.1, 0.2]

        results = run_sparsity_grid_memory_efficient(
            model=model,
            weights=weights,
            importance_scores=importance_scores,
            cached_batches=cached_batches,
            sparsities=sparsities
        )

        assert 'sparsities' in results
        assert 'xs' in results
        assert 'ys' in results
        assert 'baseline_loss' in results

        assert len(results['sparsities']) == len(sparsities)
        assert len(results['xs']) == len(sparsities)
        assert len(results['ys']) == len(sparsities)


class TestMemoryMonitor:
    """测试内存监控类。"""

    def test_get_memory_usage(self):
        """测试获取内存使用量。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import MemoryMonitor

        monitor = MemoryMonitor()
        usage = monitor.get_memory_usage()

        assert 'rss_mb' in usage
        assert 'vms_mb' in usage
        assert usage['rss_mb'] > 0

    def test_check_memory_limit(self):
        """测试检查内存限制。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import MemoryMonitor

        monitor = MemoryMonitor(memory_limit_gb=100)  # 设置一个很高的限制

        # 不应该抛出异常
        monitor.check_memory_limit()

    def test_check_memory_limit_exceeded(self):
        """测试内存限制超出时的行为。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import MemoryMonitor

        monitor = MemoryMonitor(memory_limit_gb=0.001)  # 设置一个很低的限制

        with pytest.raises(MemoryError, match="Memory limit exceeded"):
            monitor.check_memory_limit()

    def test_log_memory_usage(self, capsys):
        """测试记录内存使用量。"""
        from ckpt_compress.methods.adam_prune.memory_efficient import MemoryMonitor

        monitor = MemoryMonitor()
        monitor.log_memory_usage("Test checkpoint")

        captured = capsys.readouterr()
        assert "Test checkpoint" in captured.out
        assert "MB" in captured.out
