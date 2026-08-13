"""Block-HVP Taylor scoring for residual checkpoint experiments."""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.lib.residual_masks import TensorDict
from experiments.lib.residual_runtime import empty_device_cache, lm_loss, task_loss

GradientState = Dict[str, Tuple[torch.Tensor | None, torch.Tensor | None]]


def _snapshot_scoring_state(
    model: nn.Module,
) -> tuple[Dict[str, bool], Dict[str, bool], GradientState]:
    modes = {name: module.training for name, module in model.named_modules()}
    requires_grad: Dict[str, bool] = {}
    gradients: GradientState = {}
    for name, parameter in model.named_parameters():
        requires_grad[name] = parameter.requires_grad
        gradients[name] = (
            parameter.grad,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
    return modes, requires_grad, gradients


def _restore_scoring_state(
    model: nn.Module,
    modes: Mapping[str, bool],
    requires_grad: Mapping[str, bool],
    gradients: GradientState,
) -> None:
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


def _validate_real_finite_tensor(
    value: torch.Tensor,
    label: str,
    name: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} for {name!r} must be a tensor")
    if not value.is_floating_point() or value.is_complex():
        raise TypeError(f"{label} for {name!r} must be real floating point")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{label} for {name!r} must contain only finite values")


def _validate_scoring_inputs(
    named_params: Mapping[str, nn.Parameter],
    layers: Sequence[Sequence[str]],
    structural_blocks: Sequence[Sequence[str]],
    delta: Mapping[str, torch.Tensor],
    device: str,
) -> None:
    eligible_names = [name for layer in layers for name in layer]
    if len(eligible_names) != len(set(eligible_names)):
        raise ValueError("Eligible score parameter names must be unique")
    missing_delta = sorted(set(eligible_names).difference(delta))
    if missing_delta:
        raise ValueError(f"Residual probes are missing eligible parameters: {missing_delta}")
    for block_index, block_names in enumerate(structural_blocks):
        if len(block_names) != len(set(block_names)):
            raise ValueError(f"Structural block {block_index} contains duplicate parameters")
        missing = sorted(set(block_names).difference(named_params))
        if missing:
            raise ValueError(f"Structural block parameters are missing from model: {missing}")
    for name in eligible_names:
        parameter = named_params[name]
        probe = delta[name]
        _validate_real_finite_tensor(parameter, "Parameter", name)
        _validate_real_finite_tensor(probe, "Residual probe", name)
        if probe.shape != parameter.shape:
            raise ValueError(
                f"Residual probe shape for {name!r} must match the parameter: "
                f"{tuple(probe.shape)} != {tuple(parameter.shape)}"
            )
        if parameter.device != torch.device(device):
            raise ValueError(
                f"Parameter device for {name!r} must match scoring device: "
                f"{parameter.device} != {torch.device(device)}"
            )


def _validate_output_score(
    parameter: nn.Parameter,
    score: torch.Tensor,
    name: str,
) -> None:
    _validate_real_finite_tensor(score, "Score", name)
    if score.shape != parameter.shape:
        raise ValueError(
            f"Score shape for {name!r} must match the parameter: "
            f"{tuple(score.shape)} != {tuple(parameter.shape)}"
        )
    if score.device.type != "cpu":
        raise ValueError(f"Score for {name!r} must be stored on CPU")


def _validate_derivative(
    parameter: nn.Parameter,
    value: torch.Tensor,
    label: str,
    name: str,
) -> None:
    _validate_real_finite_tensor(value, label, name)
    if value.shape != parameter.shape:
        raise ValueError(
            f"{label} shape for {name!r} must match the parameter: "
            f"{tuple(value.shape)} != {tuple(parameter.shape)}"
        )
    if value.device != parameter.device:
        raise ValueError(
            f"{label} device for {name!r} must match the parameter: "
            f"{value.device} != {parameter.device}"
        )


def transformer_layers(
    model: nn.Module,
    model_family: str = "gpt2",
) -> List[List[str]]:
    """Return complete structural Transformer blocks for a model family."""
    from dacp.tools.importance import build_transformer_blocks

    raw_blocks = build_transformer_blocks(model, model_family)
    if not isinstance(raw_blocks, list) or not all(
        isinstance(block, list) and all(isinstance(name, str) for name in block)
        for block in raw_blocks
    ):
        raise TypeError("Transformer block builder must return lists of parameter names")
    return raw_blocks


def eligible_layers(
    model: nn.Module,
    model_family: str = "gpt2",
) -> List[List[str]]:
    """Return prunable matrix parameters grouped by Transformer block."""

    named_params = dict(model.named_parameters())
    layers = []
    for block in transformer_layers(model, model_family):
        eligible = [
            name
            for name in block
            if named_params[name].dim() >= 2
            and not any(
                pattern in name for pattern in ("wte", "wpe", "lm_head", "bias", "ln_", "LayerNorm")
            )
        ]
        if not eligible:
            raise RuntimeError(f"Transformer block has no eligible tensors: {block[:2]}")
        layers.append(eligible)
    return layers


def model_checksum(model: nn.Module) -> Tuple[float, float]:
    total = torch.zeros((), device=next(model.parameters()).device)
    absolute = torch.zeros_like(total)
    with torch.no_grad():
        for parameter in model.parameters():
            values = parameter.detach().float()
            total += values.sum()
            absolute += values.abs().sum()
    return total.item(), absolute.item()


def _structural_blocks_for_scoring(
    model: nn.Module,
    layers: Sequence[Sequence[str]],
    model_family: str,
    block_parameter_names: Sequence[Sequence[str]] | None,
) -> list[list[str]]:
    named_params = dict(model.named_parameters())
    structural_blocks = (
        [list(block) for block in block_parameter_names]
        if block_parameter_names is not None
        else transformer_layers(model, model_family)
    )
    if len(structural_blocks) != len(layers):
        raise ValueError("Structural blocks must match eligible score layers")
    for index, eligible_names in enumerate(layers):
        missing = [name for name in eligible_names if name not in named_params]
        if missing:
            raise ValueError(f"Eligible score parameters are missing from model: {missing}")
        # Compatibility callers may explicitly score frozen parameters omitted by
        # a model-family block builder. They still belong to the same graph.
        for name in eligible_names:
            if name not in structural_blocks[index]:
                structural_blocks[index].append(name)
    return structural_blocks


def compute_block_first_order_scores(
    model: nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    layers: Sequence[Sequence[str]],
    delta: Mapping[str, torch.Tensor],
    device: str,
    *,
    task_type: str = "lm",
    model_family: str = "gpt2",
    block_parameter_names: Sequence[Sequence[str]] | None = None,
    loss_fn: Callable[[nn.Module, Mapping[str, torch.Tensor], str], torch.Tensor] | None = None,
) -> Tuple[TensorDict, Dict[str, object]]:
    """Compute first-order residual damage without constructing HVP graphs."""
    if not batches:
        raise ValueError("batches must contain at least one scoring batch")
    named_params = dict(model.named_parameters())
    structural_blocks = _structural_blocks_for_scoring(
        model, layers, model_family, block_parameter_names
    )
    if loss_fn is not None:
        loss_function = loss_fn
    elif task_type == "lm":
        loss_function = lm_loss
    else:
        loss_function = lambda current_model, batch, current_device: task_loss(
            current_model, batch, task_type, current_device
        )

    _validate_scoring_inputs(named_params, layers, structural_blocks, delta, device)
    original_modes, original_requires_grad, original_gradients = _snapshot_scoring_state(model)
    versions_before = {name: p._version for name, p in named_params.items()}
    checksum_before = model_checksum(model)
    scores: TensorDict = {}
    layer_seconds = []
    started = time.perf_counter()
    model.eval()
    try:
        for layer_index, (eligible_names, block_names) in enumerate(zip(layers, structural_blocks)):
            if not block_names:
                raise RuntimeError(f"Structural block {layer_index} has no parameters")
            block_set = set(block_names)
            for name, parameter in named_params.items():
                parameter.requires_grad_(name in block_set)
            block_params = [named_params[name] for name in block_names]
            accumulator = {
                name: torch.zeros_like(delta[name], dtype=torch.float32, device="cpu")
                for name in eligible_names
            }
            block_started = time.perf_counter()
            for batch in batches:
                with sdpa_kernel(SDPBackend.MATH):
                    loss = loss_function(model, batch, device)
                    gradients = torch.autograd.grad(
                        loss, block_params, create_graph=False, allow_unused=True
                    )
                gradient_by_name = dict(zip(block_names, gradients))
                for name in eligible_names:
                    gradient = gradient_by_name[name]
                    if gradient is None:
                        raise RuntimeError(f"First-order gradient is missing for {name!r}")
                    _validate_derivative(named_params[name], gradient, "First-order gradient", name)
                    first_order = -gradient.detach() * delta[name].to(device, non_blocking=True)
                    accumulator[name].add_(first_order.float().cpu(), alpha=1.0 / len(batches))
                del loss, gradients, gradient_by_name
            for name, values in accumulator.items():
                score = values.abs()
                _validate_output_score(named_params[name], score, name)
                scores[name] = score
            layer_seconds.append(time.perf_counter() - block_started)
            empty_device_cache(device)
    finally:
        _restore_scoring_state(
            model,
            original_modes,
            original_requires_grad,
            original_gradients,
        )

    versions_after = {name: p._version for name, p in named_params.items()}
    checksum_after = model_checksum(model)
    changed_versions = [
        name for name in versions_before if versions_before[name] != versions_after[name]
    ]
    checksum_delta = [after - before for before, after in zip(checksum_before, checksum_after)]
    if changed_versions or checksum_delta != [0.0, 0.0]:
        raise RuntimeError(
            f"Scoring mutated model weights: versions={changed_versions[:5]}, "
            f"checksum_delta={checksum_delta}"
        )
    return scores, {
        "score_kind": "first_order",
        "total_seconds": time.perf_counter() - started,
        "layer_seconds": layer_seconds,
        "checksum_before": list(checksum_before),
        "checksum_after": list(checksum_after),
        "checksum_delta": checksum_delta,
        "changed_parameter_versions": changed_versions,
        "optimizer_constructed": False,
        "model_mode": "eval",
        "scoring_batches": len(batches),
        "task_type": task_type,
        "model_family": model_family,
    }


def compute_block_taylor_scores(
    model: nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    layers: Sequence[Sequence[str]],
    delta: Mapping[str, torch.Tensor],
    device: str,
    return_components: bool = False,
    *,
    task_type: str = "lm",
    model_family: str = "gpt2",
    block_parameter_names: Sequence[Sequence[str]] | None = None,
    loss_fn: Callable[[nn.Module, Mapping[str, torch.Tensor], str], torch.Tensor] | None = None,
) -> Tuple[TensorDict | Dict[str, TensorDict], Dict[str, object]]:
    """Compute g and H_bb*delta_b from one graph per structural block."""
    if not batches:
        raise ValueError("batches must contain at least one scoring batch")
    named_params = dict(model.named_parameters())
    if block_parameter_names is not None and len(block_parameter_names) != len(layers):
        raise ValueError("Block parameter groups must match eligible score layers")
    if loss_fn is not None:
        loss_function = loss_fn
    elif task_type == "lm":
        loss_function = lm_loss
    else:
        loss_function = lambda current_model, batch, current_device: task_loss(
            current_model,
            batch,
            task_type,
            current_device,
        )
    structural_blocks = _structural_blocks_for_scoring(
        model, layers, model_family, block_parameter_names
    )
    _validate_scoring_inputs(named_params, layers, structural_blocks, delta, device)
    original_modes, original_requires_grad, original_gradients = _snapshot_scoring_state(model)
    versions_before = {name: p._version for name, p in named_params.items()}
    checksum_before = model_checksum(model)
    scores: TensorDict = {}
    first_order_scores: TensorDict = {}
    second_order_scores: TensorDict = {}
    layer_seconds: List[float] = []
    started = time.perf_counter()

    model.eval()
    try:
        for layer_index, eligible_names in enumerate(layers):
            block_names = structural_blocks[layer_index]
            if not block_names:
                raise RuntimeError(f"Structural block {layer_index} has no parameters")
            block_set = set(block_names)
            for name, parameter in named_params.items():
                parameter.requires_grad_(name in block_set)

            block_params = [named_params[name] for name in block_names]
            block_started = time.perf_counter()
            signed_accumulator = {
                name: torch.zeros_like(delta[name], dtype=torch.float32, device="cpu")
                for name in eligible_names
            }
            if return_components:
                first_order_accumulator = {
                    name: torch.zeros_like(delta[name], dtype=torch.float32, device="cpu")
                    for name in eligible_names
                }
                second_order_accumulator = {
                    name: torch.zeros_like(delta[name], dtype=torch.float32, device="cpu")
                    for name in eligible_names
                }
            for batch in batches:
                with sdpa_kernel(SDPBackend.MATH):
                    loss = loss_function(model, batch, device)
                    gradients = torch.autograd.grad(
                        loss,
                        block_params,
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    gradient_by_name = dict(zip(block_names, gradients))
                    gradient_probe = loss.new_zeros(())
                    probes: Dict[str, torch.Tensor] = {}
                    for name in eligible_names:
                        gradient = gradient_by_name[name]
                        if gradient is None:
                            raise RuntimeError(f"Taylor gradient is missing for {name!r}")
                        _validate_derivative(named_params[name], gradient, "Taylor gradient", name)
                        probe = delta[name].to(device, non_blocking=True)
                        probes[name] = probe
                        gradient_probe = gradient_probe + (gradient * probe).sum()
                    hvps = torch.autograd.grad(
                        gradient_probe,
                        block_params,
                        retain_graph=False,
                        allow_unused=True,
                    )
                    hvp_by_name = dict(zip(block_names, hvps))

                for name in eligible_names:
                    gradient = gradient_by_name[name]
                    hvp = hvp_by_name[name]
                    if gradient is None or hvp is None:
                        raise RuntimeError(f"Taylor HVP is missing for {name!r}")
                    _validate_derivative(named_params[name], hvp, "Taylor HVP", name)
                    probe = probes[name]
                    first_order = -gradient.detach() * probe
                    second_order = 0.5 * probe * hvp.detach()
                    signed = first_order + second_order
                    signed_accumulator[name].add_(
                        signed.float().cpu(),
                        alpha=1.0 / len(batches),
                    )
                    if return_components:
                        first_order_accumulator[name].add_(
                            first_order.float().cpu(),
                            alpha=1.0 / len(batches),
                        )
                        second_order_accumulator[name].add_(
                            second_order.float().cpu(),
                            alpha=1.0 / len(batches),
                        )
                del loss, gradients, gradient_by_name, gradient_probe, hvps, hvp_by_name, probes

            for name in eligible_names:
                scores[name] = signed_accumulator[name].abs()
                _validate_output_score(named_params[name], scores[name], name)
                if return_components:
                    first_order_scores[name] = first_order_accumulator[name].abs()
                    second_order_scores[name] = second_order_accumulator[name].abs()
                    _validate_output_score(named_params[name], first_order_scores[name], name)
                    _validate_output_score(named_params[name], second_order_scores[name], name)

            layer_seconds.append(time.perf_counter() - block_started)
            print(
                f"  HVP layer {layer_index + 1:02d}/{len(layers):02d}: "
                f"{layer_seconds[-1]:.2f}s",
                flush=True,
            )
            del signed_accumulator
            if return_components:
                del first_order_accumulator, second_order_accumulator
            empty_device_cache(device)
    finally:
        _restore_scoring_state(
            model,
            original_modes,
            original_requires_grad,
            original_gradients,
        )

    versions_after = {name: p._version for name, p in named_params.items()}
    checksum_after = model_checksum(model)
    changed_versions = [
        name for name in versions_before if versions_before[name] != versions_after[name]
    ]
    checksum_delta = [after - before for before, after in zip(checksum_before, checksum_after)]
    if changed_versions or checksum_delta != [0.0, 0.0]:
        raise RuntimeError(
            f"Scoring mutated model weights: versions={changed_versions[:5]}, "
            f"checksum_delta={checksum_delta}"
        )

    output_scores: TensorDict | Dict[str, TensorDict]
    if return_components:
        output_scores = {
            "first_order": first_order_scores,
            "second_order": second_order_scores,
            "taylor": scores,
        }
    else:
        output_scores = scores

    return output_scores, {
        "score_kind": "taylor_hvp",
        "total_seconds": time.perf_counter() - started,
        "layer_seconds": layer_seconds,
        "checksum_before": list(checksum_before),
        "checksum_after": list(checksum_after),
        "checksum_delta": checksum_delta,
        "changed_parameter_versions": changed_versions,
        "optimizer_constructed": False,
        "model_mode": "eval",
        "hvp_batches": len(batches),
        "task_type": task_type,
        "model_family": model_family,
    }


__all__ = [
    "compute_block_first_order_scores",
    "compute_block_taylor_scores",
    "eligible_layers",
    "model_checksum",
    "transformer_layers",
]
