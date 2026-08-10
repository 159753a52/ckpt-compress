"""Exact residual masks, mask metrics, and model-state application helpers."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


TensorDict = Dict[str, torch.Tensor]
MaskDict = Dict[str, torch.Tensor]


def _validate_prune_count(prune_count: int, score_count: int) -> int:
    if isinstance(prune_count, bool):
        raise ValueError(f"Prune count must be an integer, got {prune_count}")
    try:
        normalized = int(prune_count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Prune count must be an integer, got {prune_count}") from exc
    if normalized != prune_count:
        raise ValueError(f"Prune count must be an integer, got {prune_count}")
    if normalized < 0 or normalized > score_count:
        raise ValueError(
            f"Invalid prune count {prune_count} for {score_count} scores"
        )
    return normalized


def exact_keep_mask(values: torch.Tensor, prune_count: int) -> torch.Tensor:
    """Return a deterministic mask with exactly ``prune_count`` false entries."""
    flat = values.detach().float().flatten().cpu()
    count = flat.numel()
    prune_count = _validate_prune_count(prune_count, count)
    if not torch.isfinite(flat).all().item():
        raise ValueError("Mask scores must be finite")
    if prune_count == 0:
        return torch.ones(count, dtype=torch.bool)
    if prune_count == count:
        return torch.zeros(count, dtype=torch.bool)

    threshold = torch.kthvalue(flat, prune_count).values
    pruned = flat < threshold
    remaining = prune_count - int(pruned.sum().item())
    if remaining:
        tied = torch.nonzero(flat == threshold, as_tuple=False).flatten()
        pruned[tied[:remaining]] = True
    if int(pruned.sum().item()) != prune_count:
        raise RuntimeError("Tie handling failed to produce the requested prune count")
    return ~pruned


def exact_keep_mask_from_order(
    order: torch.Tensor,
    prune_count: int,
) -> torch.Tensor:
    """Build an exact nested mask from one stable ascending score order."""
    if order.ndim != 1:
        raise ValueError("Score order must be one-dimensional")
    count = order.numel()
    prune_count = _validate_prune_count(prune_count, count)
    keep = torch.ones(count, dtype=torch.bool)
    keep[order[:prune_count]] = False
    return keep


def layer_score_orders(
    layers: Sequence[Sequence[str]],
    scores: Mapping[str, torch.Tensor],
    sort_device: str | None = None,
) -> List[torch.Tensor]:
    """Cache stable within-layer score orders for repeated exact masks."""
    orders = []
    for names in layers:
        if not names:
            raise ValueError("Structural layers must be non-empty")
        flat_scores = torch.cat(
            [scores[name].detach().float().flatten().cpu() for name in names]
        )
        if flat_scores.numel() == 0:
            raise ValueError("Structural layers must contain at least one score")
        if not torch.isfinite(flat_scores).all().item():
            raise ValueError("Mask scores must be finite")
        if sort_device is None or sort_device == "cpu":
            order = torch.argsort(flat_scores, stable=True)
        else:
            order = torch.argsort(flat_scores.to(sort_device), stable=True).cpu()
        orders.append(order)
    return orders


def _split_flat_mask(
    names: Sequence[str],
    scores: Mapping[str, torch.Tensor],
    flat_keep: torch.Tensor,
) -> MaskDict:
    masks: MaskDict = {}
    offset = 0
    for name in names:
        size = scores[name].numel()
        masks[name] = flat_keep[offset : offset + size].view_as(scores[name])
        offset += size
    return masks


def layer_mask_at_count(
    names: Sequence[str],
    scores: Mapping[str, torch.Tensor],
    prune_count: int,
    score_order: torch.Tensor | None = None,
) -> MaskDict:
    """Build one structural layer's exact-count mask."""
    if not names:
        raise ValueError("Structural layers must be non-empty")
    if not any(scores[name].numel() for name in names):
        raise ValueError("Structural layers must contain at least one score")
    if score_order is None:
        flat_scores = torch.cat([scores[name].flatten() for name in names])
        flat_keep = exact_keep_mask(flat_scores, prune_count)
    else:
        expected_size = sum(scores[name].numel() for name in names)
        if score_order.numel() != expected_size:
            raise ValueError("The score order has the wrong number of elements")
        flat_keep = exact_keep_mask_from_order(score_order, prune_count)
    return _split_flat_mask(names, scores, flat_keep)


def layer_masks(
    layers: Sequence[Sequence[str]],
    scores: Mapping[str, torch.Tensor],
    counts: Sequence[int],
    score_orders: Sequence[torch.Tensor] | None = None,
) -> MaskDict:
    if len(counts) != len(layers):
        raise ValueError("Prune counts must match the structural layers")
    if score_orders is not None and len(score_orders) != len(layers):
        raise ValueError("Score orders must match the structural layers")
    masks: MaskDict = {}
    for index, (names, prune_count) in enumerate(zip(layers, counts)):
        order = None if score_orders is None else score_orders[index]
        if order is not None:
            expected_size = sum(scores[name].numel() for name in names)
            if order.numel() != expected_size:
                raise ValueError("A score order has the wrong number of elements")
        masks.update(layer_mask_at_count(names, scores, prune_count, order))
    return masks


def global_mask(
    names: Sequence[str],
    scores: Mapping[str, torch.Tensor],
    prune_count: int,
) -> MaskDict:
    return layer_mask_at_count(names, scores, prune_count)


def mask_metrics(
    masks: Mapping[str, torch.Tensor],
    taylor_scores: Mapping[str, torch.Tensor],
) -> Dict[str, float | int]:
    pruned = 0
    proxy_cost = 0.0
    for name, keep in masks.items():
        removed = ~keep
        pruned += removed.sum().item()
        proxy_cost += taylor_scores[name][removed].sum(dtype=torch.float64).item()
    return {"pruned": int(pruned), "proxy_cost": proxy_cost}


def mask_overlap(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> Dict[str, float | int]:
    intersection = 0
    union = 0
    disagreements = 0
    for name in left:
        left_pruned = ~left[name]
        right_pruned = ~right[name]
        intersection += (left_pruned & right_pruned).sum().item()
        union += (left_pruned | right_pruned).sum().item()
        disagreements += (left_pruned != right_pruned).sum().item()
    return {
        "pruned_jaccard": intersection / max(union, 1),
        "mask_disagreements": int(disagreements),
    }


def _apply_mask_from_cpu_states(
    model: nn.Module,
    current_state: Mapping[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    device: str,
) -> None:
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, keep in masks.items():
            reference = reference_state[name].to(device, non_blocking=True)
            current = current_state[name].to(device, non_blocking=True)
            restored = torch.where(
                keep.to(device, non_blocking=True),
                current,
                reference,
            )
            named_params[name].copy_(restored)


def restore_with_mask(
    model: nn.Module,
    current_state: Mapping[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    device: str,
) -> None:
    """Reload the current model state, then restore parameters selected by a mask."""
    model.load_state_dict(current_state, strict=True)
    _apply_mask_from_cpu_states(
        model,
        current_state,
        reference_state,
        masks,
        device,
    )


def cache_mask_states_on_device(
    current_state: Mapping[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor],
    names: Sequence[str],
    device: str,
) -> Tuple[TensorDict, TensorDict]:
    """Transfer mask-selectable states once for repeated whole-model probes."""
    current_device = {
        name: current_state[name].to(device, non_blocking=True, copy=True)
        for name in names
    }
    reference_device = {
        name: reference_state[name].to(device, non_blocking=True, copy=True)
        for name in names
    }
    return current_device, reference_device


def apply_mask_from_device_states(
    model: nn.Module,
    current_device: Mapping[str, torch.Tensor],
    reference_device: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    device: str,
) -> None:
    """Apply a mask using states already resident on the evaluation device."""
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, keep in masks.items():
            torch.where(
                keep.to(device, non_blocking=True),
                current_device[name],
                reference_device[name],
                out=named_params[name],
            )


def apply_layer_mask(
    model: nn.Module,
    current_state: Mapping[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    device: str,
) -> None:
    """Apply a partial mask without reloading parameters outside that layer."""
    _apply_mask_from_cpu_states(
        model,
        current_state,
        reference_state,
        masks,
        device,
    )


def layer_rates(
    layers: Sequence[Sequence[str]],
    masks: Mapping[str, torch.Tensor],
) -> List[float]:
    rates = []
    for names in layers:
        total = sum(masks[name].numel() for name in names)
        pruned = sum((~masks[name]).sum().item() for name in names)
        rates.append(pruned / total)
    return rates


__all__ = [
    "MaskDict",
    "TensorDict",
    "apply_layer_mask",
    "apply_mask_from_device_states",
    "cache_mask_states_on_device",
    "exact_keep_mask",
    "exact_keep_mask_from_order",
    "global_mask",
    "layer_mask_at_count",
    "layer_masks",
    "layer_rates",
    "layer_score_orders",
    "mask_metrics",
    "mask_overlap",
    "restore_with_mask",
]
