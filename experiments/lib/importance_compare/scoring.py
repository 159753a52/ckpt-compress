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

from typing import Dict, List, Mapping, Optional, Protocol, Tuple, cast

import torch
import torch.nn as nn

from dacp.pruning.importance import (
    FirstOrderScorer,
    ImportanceScorer,
    MagnitudeScorer,
    ResidualMagnitudeScorer,
)
from dacp.pruning.pruner import filter_prunable_params
from experiments.lib.losses import SUPPORTED_TASK_TYPES, make_task_loss

_SUPPORTED_METHODS = frozenset(
    {"magnitude", "first-order", "second-order-hvp", "residual-magnitude"}
)
_SUPPORTED_TASK_TYPES = SUPPORTED_TASK_TYPES
_SUPPORTED_HVP_MODES = frozenset({"full", "block"})


class _LossFn(Protocol):
    def __call__(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor: ...


def _validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


# ------------------------------------------------------------------ #
#  核心: 梯度 & HVP 收集
# ------------------------------------------------------------------ #


def _make_loss_fn(task_type: str) -> _LossFn:
    """Compatibility wrapper for the shared task-loss factory."""
    return cast(_LossFn, make_task_loss(task_type))


_GradientState = Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]


def _snapshot_model_state(
    model: nn.Module,
) -> Tuple[Dict[str, bool], Dict[str, bool], _GradientState]:
    """Capture mutable autograd state before a read-only scoring pass."""
    modes = {name: module.training for name, module in model.named_modules()}
    requires_grad: Dict[str, bool] = {}
    gradients: _GradientState = {}
    for name, parameter in model.named_parameters():
        requires_grad[name] = parameter.requires_grad
        gradients[name] = (
            parameter.grad,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
    return modes, requires_grad, gradients


def _restore_model_state(
    model: nn.Module,
    modes: Dict[str, bool],
    requires_grad: Dict[str, bool],
    gradients: _GradientState,
) -> None:
    """Restore mode, requires-grad flags, and pre-existing gradient buffers."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(requires_grad[name])
        original_gradient, original_values = gradients[name]
        if original_gradient is None:
            parameter.grad = None
        else:
            assert original_values is not None
            with torch.no_grad():
                original_gradient.copy_(original_values)
            parameter.grad = original_gradient
    for name, module in model.named_modules():
        module.training = modes[name]


def _validate_loss(loss: torch.Tensor) -> None:
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise TypeError("Scoring loss must be a scalar tensor")
    if not loss.is_floating_point() or loss.is_complex():
        raise TypeError("Scoring loss must be real floating point")
    if not torch.isfinite(loss).item():
        raise ValueError("Scoring loss must be finite")


def _validate_tensor_like(
    reference: torch.Tensor,
    value: torch.Tensor,
    label: str,
    name: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} for {name!r} must be a tensor")
    if value.shape != reference.shape:
        raise ValueError(
            f"{label} shape for {name!r} must match the parameter: "
            f"{tuple(value.shape)} != {tuple(reference.shape)}"
        )
    if value.device != reference.device:
        raise ValueError(
            f"{label} device for {name!r} must match the parameter: "
            f"{value.device} != {reference.device}"
        )
    if not value.is_floating_point() or value.is_complex():
        raise TypeError(f"{label} for {name!r} must be real floating point")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{label} for {name!r} must contain only finite values")


def _collect_gradients(
    model: nn.Module,
    cached_train: list,
    task_type: str,
    num_batches: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Average per-batch gradients without writing parameter ``.grad`` buffers."""
    num_batches = _validate_positive_int("num_batches", num_batches)
    if not cached_train:
        raise ValueError("cached_train must contain at least one batch")

    named_params = dict(model.named_parameters())
    trainable = [
        (name, parameter) for name, parameter in named_params.items() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("Gradient collection requires at least one trainable parameter")
    prunable_names = set(
        filter_prunable_params(
            {name: parameter.detach() for name, parameter in named_params.items()}
        )
    )
    trainable_names = {name for name, _ in trainable}
    frozen_prunable = sorted(prunable_names.difference(trainable_names))
    if frozen_prunable:
        raise RuntimeError(
            "Gradient collection requires all prunable parameters to have "
            f"requires_grad=True: {frozen_prunable}"
        )

    original_modes, original_requires_grad, original_gradients = _snapshot_model_state(model)
    loss_fn = _make_loss_fn(task_type)
    accumulated: Dict[str, torch.Tensor] = {}
    actual = min(num_batches, len(cached_train))

    model.train()
    try:
        for index in range(actual):
            batch = {key: value.to(device) for key, value in cached_train[index].items()}
            loss = loss_fn(model, batch)
            _validate_loss(loss)
            batch_gradients = torch.autograd.grad(
                loss,
                [parameter for _, parameter in trainable],
                create_graph=False,
                allow_unused=True,
            )
            for (name, parameter), gradient in zip(trainable, batch_gradients):
                if gradient is None:
                    if name in prunable_names:
                        raise RuntimeError(f"Gradient is missing for prunable parameter {name!r}")
                    continue
                _validate_tensor_like(parameter, gradient, "Gradient", name)
                gradient_cpu = gradient.detach().cpu()
                if name in accumulated:
                    accumulated[name].add_(gradient_cpu)
                else:
                    accumulated[name] = gradient_cpu.clone()
    finally:
        _restore_model_state(
            model,
            original_modes,
            original_requires_grad,
            original_gradients,
        )

    for name in accumulated:
        accumulated[name] /= actual
    return dict(accumulated)


def _require_gradients(
    gradients: Optional[Dict[str, torch.Tensor]],
    method: str,
) -> Dict[str, torch.Tensor]:
    if gradients is None:
        raise RuntimeError(f"{method} scoring requires collected gradients")
    return gradients


def _complete_auxiliary_mapping(
    weights: Dict[str, torch.Tensor],
    auxiliary: Dict[str, torch.Tensor],
    label: str,
) -> Dict[str, torch.Tensor]:
    missing = sorted(set(weights).difference(auxiliary))
    if missing:
        raise RuntimeError(f"{label} coverage is missing prunable parameters: {missing}")
    complete = {}
    for name, weight in weights.items():
        _validate_tensor_like(weight, weight, "Weight", name)
        value = auxiliary[name]
        _validate_tensor_like(weight, value, label, name)
        complete[name] = value
    return complete


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
    model_family: str = "gpt2",
) -> Dict[str, torch.Tensor]:
    """计算 HVP 二阶得分。

    hvp_mode:
        'block' — 真正的 block-wise HVP（论文 Algorithm 1）。
                  每个 block = 1 Transformer layer，仅对该 block 的参数
                  开启 requires_grad，峰值内存 ≈ 单 block 二阶图 + 前向激活。
        'full'  — 全局 HVP，需要保留整个模型的二阶计算图。
    """
    from dacp.tools.importance import compute_hvp_batched

    hvp_batches = _validate_positive_int("hvp_batches", hvp_batches)
    if not cached_train:
        raise ValueError("cached_train must contain at least one batch")
    if hvp_mode not in _SUPPORTED_HVP_MODES:
        raise ValueError(f"Unknown hvp_mode: {hvp_mode}")

    original_modes, original_requires_grad, original_gradients = _snapshot_model_state(model)
    try:
        model.train()
        loss_fn = _make_loss_fn(task_type)
        weights = {name: parameter.detach().cpu() for name, parameter in model.named_parameters()}
        prunable_w = filter_prunable_params(weights)
        prunable_g = _complete_auxiliary_mapping(prunable_w, gradients, "Gradient")
        gpu_batches = [
            {key: value.to(device) for key, value in batch.items()} for batch in cached_train
        ]

        if hvp_mode == "block":
            from dacp.tools.importance import (
                build_transformer_blocks,
                compute_hvp_blockwise_batched,
            )

            hvp_result = compute_hvp_blockwise_batched(
                model,
                loss_fn,
                gpu_batches,
                build_transformer_blocks(model, model_family),
                num_batches=hvp_batches,
            )
        else:
            vector = {name: weight.to(device) for name, weight in prunable_w.items()}
            full_vector = {
                name: torch.zeros_like(parameter) for name, parameter in model.named_parameters()
            }
            full_vector.update(vector)
            hvp_result = compute_hvp_batched(
                model,
                loss_fn,
                gpu_batches,
                full_vector,
                hvp_batches,
            )

        prunable_hvp = {
            name: value.detach().cpu() for name, value in hvp_result.items() if name in prunable_w
        }
        prunable_hvp = _complete_auxiliary_mapping(prunable_w, prunable_hvp, "HVP")
        scores: Dict[str, torch.Tensor] = {}
        epsilon = 1e-12
        for name, theta in prunable_w.items():
            first_order = -prunable_g[name] * theta
            second_order = alpha * theta * prunable_hvp[name]
            if normalize:
                first_scale = first_order.abs().mean() + epsilon
                second_scale = second_order.abs().mean() + epsilon
                scores[name] = torch.abs(first_order / first_scale + second_order / second_scale)
            else:
                scores[name] = torch.abs(first_order + second_order)
        return _complete_auxiliary_mapping(prunable_w, scores, "HVP score")
    finally:
        _restore_model_state(
            model,
            original_modes,
            original_requires_grad,
            original_gradients,
        )


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
    hvp_mode: str = "full",
    chunk_size: int = 10,
    reference_weights: Optional[Dict[str, torch.Tensor]] = None,
    grad_batches_first_order: Optional[int] = None,
    model_family: str = "gpt2",
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
            f"Unknown task_type: {task_type}. " f"Available: {sorted(_SUPPORTED_TASK_TYPES)}"
        )
    if hvp_mode not in _SUPPORTED_HVP_MODES:
        raise ValueError(
            f"Unknown hvp_mode: {hvp_mode}. " f"Available: {sorted(_SUPPORTED_HVP_MODES)}"
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
    need_grad = any(m in ("first-order", "second-order-hvp") for m in methods)

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
        if grad_batches_first_order is not None and "first-order" in methods:
            gradients_fo = _collect_gradients(
                model, cached_train, task_type, grad_batches_first_order, device
            )

    # 为每种方法计算得分
    for method in methods:
        print(f"  [scoring] 计算 {method} 得分...")
        scorer: ImportanceScorer

        if method == "magnitude":
            scorer = MagnitudeScorer()
            results[method] = scorer.score(prunable_w)

        elif method == "first-order":
            scorer = FirstOrderScorer()
            g = gradients_fo if gradients_fo is not None else gradients
            g = _require_gradients(g, method)
            prunable_g = _complete_auxiliary_mapping(prunable_w, g, "Gradient")
            results[method] = scorer.score(prunable_w, prunable_g)

        elif method == "second-order-hvp":
            required_gradients = _require_gradients(gradients, method)
            results[method] = _compute_hvp_scores(
                model,
                cached_train,
                task_type,
                alpha,
                hvp_batches,
                hvp_mode,
                chunk_size,
                device,
                normalize,
                required_gradients,
                model_family=model_family,
            )

        elif method == "residual-magnitude":
            scorer = ResidualMagnitudeScorer()
            if reference_weights is None:
                raise ValueError("residual-magnitude scoring requires reference_weights")
            ref = _complete_auxiliary_mapping(
                prunable_w,
                reference_weights,
                "Reference weight",
            )
            results[method] = scorer.score(prunable_w, reference_weights=ref)

        else:
            raise ValueError(f"Unknown importance method: {method}")

        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results
