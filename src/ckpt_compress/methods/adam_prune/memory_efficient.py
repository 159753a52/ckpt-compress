"""
内存高效的剪枝评估模块。

提供内存优化的模型评估和剪枝实验功能，避免 deepcopy 导致的内存泄漏。

主要优化：
1. 使用原地修改 + 恢复代替 deepcopy
2. 强制垃圾回收
3. 内存监控和限制
4. 数据收集时可选移动到 CPU
"""

import gc
import torch
import torch.nn as nn
import psutil
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

from .layer_pruning import (
    prune_by_global_sparsity,
    compute_sum_pruned_scores,
)
from .calibration import compute_calibration_variables, fit_powerlaw


def force_garbage_collection():
    """
    强制执行垃圾回收。

    在删除大型对象后调用此函数以确保内存被释放。
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class MemoryMonitor:
    """
    内存监控器。

    用于监控和限制内存使用，防止 OOM。
    """

    def __init__(self, memory_limit_gb: float = 30.0):
        """
        初始化内存监控器。

        参数:
            memory_limit_gb: 内存限制（GB）
        """
        self.memory_limit_gb = memory_limit_gb
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024

    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取当前内存使用量。

        返回:
            包含 rss_mb 和 vms_mb 的字典
        """
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
        }

    def check_memory_limit(self):
        """
        检查内存是否超出限制。

        异常:
            MemoryError: 如果内存超出限制
        """
        process = psutil.Process()
        mem_info = process.memory_info()
        if mem_info.rss > self.memory_limit_bytes:
            raise MemoryError(
                f"Memory limit exceeded: {mem_info.rss / (1024**3):.2f} GB > "
                f"{self.memory_limit_gb:.2f} GB"
            )

    def log_memory_usage(self, checkpoint_name: str = ""):
        """
        记录当前内存使用量。

        参数:
            checkpoint_name: 检查点名称（用于日志）
        """
        usage = self.get_memory_usage()
        print(f"[Memory] {checkpoint_name}: RSS={usage['rss_mb']:.1f} MB, "
              f"VMS={usage['vms_mb']:.1f} MB")


class MemoryEfficientDataCollector:
    """
    内存高效的数据收集器。

    收集模型权重、梯度和优化器状态，可选择移动到 CPU 以节省 GPU 内存。
    """

    def collect(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        to_cpu: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        收集模型数据。

        参数:
            model: PyTorch 模型
            optimizer: 优化器
            to_cpu: 是否将数据移动到 CPU

        返回:
            (weights, gradients, exp_avg_sq) 元组
        """
        weights = {}
        gradients = {}
        exp_avg_sq = {}

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if to_cpu:
                    weights[name] = param.data.clone().cpu()
                    gradients[name] = param.grad.clone().cpu()
                else:
                    weights[name] = param.data.clone()
                    gradients[name] = param.grad.clone()

                if param in optimizer.state:
                    state = optimizer.state[param]
                    if 'exp_avg_sq' in state:
                        if to_cpu:
                            exp_avg_sq[name] = state['exp_avg_sq'].clone().cpu()
                        else:
                            exp_avg_sq[name] = state['exp_avg_sq'].clone()
                    else:
                        if to_cpu:
                            exp_avg_sq[name] = torch.zeros_like(param.data).cpu()
                        else:
                            exp_avg_sq[name] = torch.zeros_like(param.data)
                else:
                    if to_cpu:
                        exp_avg_sq[name] = torch.zeros_like(param.data).cpu()
                    else:
                        exp_avg_sq[name] = torch.zeros_like(param.data)

        return weights, gradients, exp_avg_sq


class MemoryEfficientEvaluator:
    """
    内存高效的模型评估器。

    使用原地修改 + 恢复代替 deepcopy，大幅减少内存使用。
    """

    def __init__(self, model: nn.Module):
        """
        初始化评估器。

        参数:
            model: PyTorch 模型
        """
        self.model = model
        self._original_weights: Optional[Dict[str, torch.Tensor]] = None

    def save_original_weights(self):
        """
        保存原始权重。

        在进行任何修改之前调用此方法。
        """
        self._original_weights = {}
        for name, param in self.model.named_parameters():
            self._original_weights[name] = param.data.clone()

    def apply_pruned_weights(self, pruned_weights: Dict[str, torch.Tensor]):
        """
        应用剪枝后的权重。

        参数:
            pruned_weights: 剪枝后的权重字典
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in pruned_weights:
                    param.data.copy_(pruned_weights[name])

    def restore_original_weights(self):
        """
        恢复原始权重。

        异常:
            RuntimeError: 如果没有保存原始权重
        """
        if self._original_weights is None:
            raise RuntimeError("No original weights saved. Call save_original_weights() first.")

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._original_weights:
                    param.data.copy_(self._original_weights[name])

    @contextmanager
    def temporary_weights(self, pruned_weights: Dict[str, torch.Tensor]):
        """
        临时应用剪枝权重的上下文管理器。

        参数:
            pruned_weights: 剪枝后的权重字典

        用法:
            with evaluator.temporary_weights(pruned_weights):
                # 在此处评估剪枝后的模型
                loss = evaluate(model)
            # 退出后权重自动恢复
        """
        # 保存原始权重（如果还没有保存）
        should_cleanup = False
        if self._original_weights is None:
            self.save_original_weights()
            should_cleanup = True

        try:
            self.apply_pruned_weights(pruned_weights)
            yield
        finally:
            self.restore_original_weights()
            if should_cleanup:
                self.cleanup()

    def evaluate_loss(
        self,
        batches: List[Dict[str, torch.Tensor]],
        num_batches: Optional[int] = None
    ) -> float:
        """
        评估模型损失。

        参数:
            batches: 批次数据列表
            num_batches: 使用的批次数量（默认使用全部）

        返回:
            平均损失
        """
        self.model.eval()
        losses = []

        if num_batches is None:
            num_batches = len(batches)

        with torch.no_grad():
            for i, batch in enumerate(batches):
                if i >= num_batches:
                    break

                # 处理不同的输入格式
                if 'input_ids' in batch:
                    # HuggingFace 格式
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        labels=batch.get('labels')
                    )
                else:
                    # 简单格式
                    outputs = self.model(
                        batch['input_ids'],
                        labels=batch.get('labels')
                    )

                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    losses.append(outputs.loss.item())

        self.model.train()

        if not losses:
            return 0.0
        return float(sum(losses) / len(losses))

    def evaluate_loss_increase(
        self,
        pruned_weights: Dict[str, torch.Tensor],
        batches: List[Dict[str, torch.Tensor]],
        num_batches: Optional[int] = None
    ) -> float:
        """
        评估剪枝后的损失增量。

        参数:
            pruned_weights: 剪枝后的权重
            batches: 批次数据列表
            num_batches: 使用的批次数量

        返回:
            损失增量 (pruned_loss - original_loss)
        """
        # 计算原始损失
        original_loss = self.evaluate_loss(batches, num_batches)

        # 使用临时权重计算剪枝后损失
        with self.temporary_weights(pruned_weights):
            pruned_loss = self.evaluate_loss(batches, num_batches)

        return pruned_loss - original_loss

    def cleanup(self):
        """
        清理保存的权重以释放内存。
        """
        if self._original_weights is not None:
            del self._original_weights
            self._original_weights = None
            force_garbage_collection()


def run_single_sparsity_experiment(
    evaluator: MemoryEfficientEvaluator,
    weights: Dict[str, torch.Tensor],
    importance_scores: Dict[str, torch.Tensor],
    cached_batches: List[Dict[str, torch.Tensor]],
    sparsity: float,
    baseline_loss: float
) -> Dict[str, float]:
    """
    运行单个稀疏度的实验。

    参数:
        evaluator: 内存高效评估器
        weights: 模型权重
        importance_scores: 重要性得分
        cached_batches: 缓存的批次数据
        sparsity: 目标稀疏度
        baseline_loss: 基线损失

    返回:
        包含实验结果的字典
    """
    # 剪枝
    pruned_weights, masks = prune_by_global_sparsity(
        weights, importance_scores, sparsity
    )

    # 计算 sum_pruned_scores (p)
    sum_pruned_scores = compute_sum_pruned_scores(importance_scores, masks)

    # 评估损失增量
    actual_loss_increase = evaluator.evaluate_loss_increase(
        pruned_weights, cached_batches
    )

    # 计算校正变量 x 和 y
    calib_vars = compute_calibration_variables(
        sum_pruned_scores=sum_pruned_scores,
        baseline_loss=baseline_loss,
        actual_loss_increase=actual_loss_increase
    )

    # 清理临时数据
    del pruned_weights, masks
    force_garbage_collection()

    return {
        'sparsity': sparsity,
        'x': calib_vars['x'],
        'y': calib_vars['y'],
        'sum_pruned_scores': sum_pruned_scores,
        'actual_loss_increase': actual_loss_increase
    }


def run_sparsity_grid_memory_efficient(
    model: nn.Module,
    weights: Dict[str, torch.Tensor],
    importance_scores: Dict[str, torch.Tensor],
    cached_batches: List[Dict[str, torch.Tensor]],
    sparsities: List[float],
    memory_limit_gb: float = 30.0,
    verbose: bool = True
) -> Dict[str, List]:
    """
    内存高效的稀疏度网格实验。

    使用原地修改 + 恢复代替 deepcopy，大幅减少内存使用。

    参数:
        model: PyTorch 模型
        weights: 模型权重
        importance_scores: 重要性得分
        cached_batches: 缓存的批次数据
        sparsities: 稀疏度网格
        memory_limit_gb: 内存限制（GB）
        verbose: 是否打印详细信息

    返回:
        包含实验数据和拟合结果的字典
    """
    if verbose:
        print(f"Running memory-efficient sparsity grid experiment...")
        print(f"Sparsities: {sparsities}")

    # 初始化内存监控和评估器
    monitor = MemoryMonitor(memory_limit_gb=memory_limit_gb)
    evaluator = MemoryEfficientEvaluator(model)
    evaluator.save_original_weights()

    # 计算基线损失
    baseline_loss = evaluator.evaluate_loss(cached_batches)
    if verbose:
        print(f"Baseline loss L0: {baseline_loss:.4f}")
        monitor.log_memory_usage("After baseline evaluation")

    results = {
        'sparsities': [],
        'xs': [],
        'ys': [],
        'sum_pruned_scores': [],
        'actual_loss_increases': [],
        'baseline_loss': baseline_loss,
    }

    for sparsity in sparsities:
        if verbose:
            print(f"  Sparsity {sparsity*100:.1f}%...")

        # 检查内存限制
        monitor.check_memory_limit()

        # 运行单个稀疏度实验
        result = run_single_sparsity_experiment(
            evaluator=evaluator,
            weights=weights,
            importance_scores=importance_scores,
            cached_batches=cached_batches,
            sparsity=sparsity,
            baseline_loss=baseline_loss
        )

        results['sparsities'].append(result['sparsity'])
        results['xs'].append(result['x'])
        results['ys'].append(result['y'])
        results['sum_pruned_scores'].append(result['sum_pruned_scores'])
        results['actual_loss_increases'].append(result['actual_loss_increase'])

        if verbose:
            print(f"    p={result['sum_pruned_scores']:.4f}, "
                  f"Δloss={result['actual_loss_increase']:.4f}")
            print(f"    x={result['x']:.6f}, y={result['y']:.6f}")
            monitor.log_memory_usage(f"After sparsity {sparsity*100:.1f}%")

    # 拟合幂律模型
    if verbose:
        print("\nFitting power-law model y = a * x^b...")

    # 过滤有效数据点
    valid_indices = [
        i for i in range(len(results['xs']))
        if results['xs'][i] > 0 and results['ys'][i] > 0
    ]

    if len(valid_indices) >= 2:
        xs_valid = [results['xs'][i] for i in valid_indices]
        ys_valid = [results['ys'][i] for i in valid_indices]

        try:
            a, b = fit_powerlaw(xs_valid, ys_valid)
            results['powerlaw_params'] = {'a': a, 'b': b}
            if verbose:
                print(f"Fitted: y = {a:.4f} * x^{b:.4f}")
        except Exception as e:
            if verbose:
                print(f"Power-law fitting failed: {e}")
            results['powerlaw_params'] = None
    else:
        if verbose:
            print("Not enough valid data points for fitting")
        results['powerlaw_params'] = None

    # 清理
    evaluator.cleanup()
    if verbose:
        monitor.log_memory_usage("After cleanup")

    return results
