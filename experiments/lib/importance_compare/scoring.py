"""重要性得分批量计算。

统一接口：一次调用即可计算多种 importance method 的得分。
使用方式:
    scores = compute_scores_by_method(
        model, cached_train, task_type,
        methods=['magnitude', 'first-order', 'second-order-hvp'],
        alpha=0.5, hvp_batches=8,
    )
    # scores['magnitude']  -> {param_name: score_tensor}
    # scores['first-order'] -> {param_name: score_tensor}
"""

from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from dacp.pruning.importance import (
    MagnitudeScorer,
    FirstOrderScorer,
    ResidualMagnitudeScorer,
)
from dacp.pruning.pruner import filter_prunable_params
from experiments.lib.losses import SUPPORTED_TASK_TYPES, make_task_loss


_SUPPORTED_METHODS = frozenset(
    {"magnitude", "first-order", "second-order-hvp", "residual-magnitude"}
)
_SUPPORTED_TASK_TYPES = SUPPORTED_TASK_TYPES
_SUPPORTED_HVP_MODES = frozenset({"full", "block"})


def _validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


# ------------------------------------------------------------------ #
#  核心: 梯度 & HVP 收集
# ------------------------------------------------------------------ #

def _make_loss_fn(task_type: str):
    """Compatibility wrapper for the shared task-loss factory."""
    return make_task_loss(task_type)


def _collect_gradients(
    model: nn.Module,
    cached_train: list,
    task_type: str,
    num_batches: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    """累积梯度并平均。"""
    num_batches = _validate_positive_int("num_batches", num_batches)
    if not cached_train:
        raise ValueError("cached_train must contain at least one batch")

    model.train()
    loss_fn = _make_loss_fn(task_type)
    accumulated = defaultdict(lambda: 0)
    actual = min(num_batches, len(cached_train))

    for i in range(actual):
        batch = {k: v.to(device) for k, v in cached_train[i].items()}
        model.zero_grad()
        loss = loss_fn(model, batch)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                accumulated[name] = accumulated[name] + p.grad.detach().cpu()

    for name in accumulated:
        accumulated[name] /= actual
    return dict(accumulated)


def _compute_hvp_scores(
    model: nn.Module,
    cached_train: list,
    task_type: str,
    alpha: float,
    hvp_batches: int,
    hvp_mode: str,
    chunk_size: int,
    device: str,
    normalize: bool,
    gradients: Dict[str, torch.Tensor],
    model_family: str = 'gpt2',
) -> Dict[str, torch.Tensor]:
    """计算 HVP 二阶得分。

    hvp_mode:
        'block' — 真正的 block-wise HVP（论文 Algorithm 1）。
                  每个 block = 1 Transformer layer，仅对该 block 的参数
                  开启 requires_grad，峰值内存 ≈ 单 block 二阶图 + 前向激活。
        'full'  — 全局 HVP，需要保留整个模型的二阶计算图。
    """
    from dacp.tools.importance import (
        compute_hvp_batched,
    )

    model.train()
    loss_fn = _make_loss_fn(task_type)
    weights = {n: p.data.detach().cpu() for n, p in model.named_parameters()}
    prunable_w = filter_prunable_params(weights)
    prunable_g = {k: gradients.get(k, torch.zeros_like(v)) for k, v in prunable_w.items()}

    # 将 data_batches 移到 device
    gpu_batches = []
    for b in cached_train:  # patched: pass ALL batches; HVP uses num_batches, grad accum uses all
        gpu_batches.append({k: v.to(device) for k, v in b.items()})

    if hvp_mode == 'block':
        # 真正的 block-wise HVP（按 Transformer layer 划分 block）
        from dacp.tools.importance import (
            compute_importance_scores_hvp_blockwise,
        )
        def loss_fn_gpu(m, batch):
            return loss_fn(m, batch)
        scores = compute_importance_scores_hvp_blockwise(
            model, loss_fn_gpu, gpu_batches,
            model_family=model_family, num_batches=hvp_batches,
            alpha=alpha, normalize=normalize,
        )
        # 只保留 prunable
        scores = {k: v.detach().cpu() for k, v in scores.items() if k in prunable_w}
    else:
        # Full HVP
        # 将权重作为 HVP 向量
        vector = {n: w.to(device) for n, w in prunable_w.items()}
        full_vector = {n: torch.zeros_like(p.data) for n, p in model.named_parameters()}
        full_vector.update(vector)

        hvp_result = compute_hvp_batched(model, loss_fn, gpu_batches, full_vector, hvp_batches)
        scores = {}
        _EPS = 1e-12
        for name in prunable_w:
            theta = prunable_w[name]
            grad = prunable_g.get(name, torch.zeros_like(theta))
            hvp = hvp_result.get(name, torch.zeros_like(theta)).cpu()
            first_order = -grad * theta
            second_order = alpha * theta * hvp
            if normalize:
                mu1 = first_order.abs().mean() + _EPS
                mu2 = second_order.abs().mean() + _EPS
                scores[name] = torch.abs(first_order / mu1 + second_order / mu2)
            else:
                scores[name] = torch.abs(first_order + second_order)

    return scores


# ------------------------------------------------------------------ #
#  公开接口
# ------------------------------------------------------------------ #

def compute_scores_by_method(
    model: nn.Module,
    cached_train: list,
    task_type: str,
    methods: List[str],
    alpha: float = 0.5,
    hvp_batches: int = 8,
    hvp_mode: str = 'full',
    chunk_size: int = 10,
    reference_weights: Optional[Dict[str, torch.Tensor]] = None,
    grad_batches_first_order: Optional[int] = None,
    model_family: str = 'gpt2',
    normalize: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """为多种 importance 方法批量计算得分。

    参数:
        model: 模型（已在目标 device 上）
        cached_train: 缓存的训练批次（CPU）
        task_type: 'lm' | 'cls' | 'cv' | 'reg'
        methods: 要计算的方法名列表
        alpha: 二阶项权重
        hvp_batches: HVP 使用的批次数
        hvp_mode: 'full' | 'block'
        chunk_size: block 模式的块大小（仅 full 模式使用，block 模式已废弃此参数）
        reference_weights: 残差剪枝用的参考权重
        grad_batches_first_order: first-order 方法单独的梯度 batch 数
        model_family: 模型族名称（'gpt2', 'pythia', 'vit', 'bert'），用于 block-wise HVP

    返回:
        {method_name: {param_name: score_tensor}} 字典
    """
    if isinstance(methods, str):
        raise ValueError("methods must be a sequence of method names, not a string")
    methods = list(dict.fromkeys(methods))
    unknown_methods = sorted(set(methods) - _SUPPORTED_METHODS)
    if unknown_methods:
        raise ValueError(
            f"Unknown importance method(s): {unknown_methods}. "
            f"Available: {sorted(_SUPPORTED_METHODS)}"
        )
    if not methods:
        return {}
    if task_type not in _SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"Unknown task_type: {task_type}. "
            f"Available: {sorted(_SUPPORTED_TASK_TYPES)}"
        )
    if hvp_mode not in _SUPPORTED_HVP_MODES:
        raise ValueError(
            f"Unknown hvp_mode: {hvp_mode}. "
            f"Available: {sorted(_SUPPORTED_HVP_MODES)}"
        )
    hvp_batches = _validate_positive_int("hvp_batches", hvp_batches)
    if grad_batches_first_order is not None:
        grad_batches_first_order = _validate_positive_int(
            "grad_batches_first_order",
            grad_batches_first_order,
        )

    device = next(model.parameters()).device
    results: Dict[str, Dict[str, torch.Tensor]] = {}

    # 判断是否需要梯度
    need_grad = any(m in ('first-order', 'second-order-hvp') for m in methods)

    # 获取权重
    weights = {n: p.data.detach().cpu() for n, p in model.named_parameters()}
    prunable_w = filter_prunable_params(weights)

    # 收集梯度
    gradients = None
    gradients_fo = None

    if need_grad:
        print("  [scoring] 收集梯度...")
        gradients = _collect_gradients(model, cached_train, task_type, hvp_batches, device)

        # first-order 可能需要不同 batch 数的梯度
        if grad_batches_first_order is not None and 'first-order' in methods:
            gradients_fo = _collect_gradients(
                model, cached_train, task_type, grad_batches_first_order, device)

    # 为每种方法计算得分
    for method in methods:
        print(f"  [scoring] 计算 {method} 得分...")

        if method == 'magnitude':
            scorer = MagnitudeScorer()
            results[method] = scorer.score(prunable_w)

        elif method == 'first-order':
            scorer = FirstOrderScorer()
            g = gradients_fo if gradients_fo is not None else gradients
            prunable_g = {k: g.get(k, torch.zeros_like(v)) for k, v in prunable_w.items()}
            results[method] = scorer.score(prunable_w, prunable_g)

        elif method == 'second-order-hvp':
            results[method] = _compute_hvp_scores(
                model, cached_train, task_type, alpha,
                hvp_batches, hvp_mode, chunk_size, device, normalize, gradients,
                model_family=model_family,
            )

        elif method == 'residual-magnitude':
            scorer = ResidualMagnitudeScorer()
            ref = None
            if reference_weights:
                ref = {k: reference_weights[k] for k in prunable_w if k in reference_weights}
            results[method] = scorer.score(prunable_w, reference_weights=ref)

        else:
            raise ValueError(f"Unknown importance method: {method}")

        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results
