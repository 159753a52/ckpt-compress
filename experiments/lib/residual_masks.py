"""Exact residual masks, mask metrics, and model-state application helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from dacp.pruning.masks import exact_keep_mask as _exact_keep_mask
from dacp.pruning.masks import validate_prune_count

TensorDict = Dict[str, torch.Tensor]
MaskDict = Dict[str, torch.Tensor]
_GLOBAL_SELECTION_CHUNK_ELEMENTS = 1 << 20
_FLOAT32_KEY_MASK = (1 << 32) - 1


def exact_keep_mask(values: torch.Tensor, prune_count: int) -> torch.Tensor:
    """Return a deterministic mask with exactly ``prune_count`` false entries."""
    flat = values.detach().float().flatten().cpu()
    return _exact_keep_mask(flat, prune_count)


def exact_keep_mask_from_order(
    order: torch.Tensor,
    prune_count: int,
) -> torch.Tensor:
    """Build an exact nested mask from one stable ascending score order."""
    if order.ndim != 1:
        raise ValueError("Score order must be one-dimensional")
    count = order.numel()
    prune_count = validate_prune_count(prune_count, count)
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
        flat_scores = torch.cat([scores[name].detach().float().flatten().cpu() for name in names])
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
    """Build an exact global mask without concatenating all model scores.

    Float32 scores are mapped to monotonic integer keys. Two histogram passes
    locate the exact 32-bit threshold, and a final pass resolves threshold ties
    in ``names``/flattened-index order. Peak temporary memory is bounded by one
    fixed-size score chunk rather than the total eligible parameter count.
    """
    if not names:
        raise ValueError("Global mask names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError("Global mask names must be unique")
    total = sum(scores[name].numel() for name in names)
    if total == 0:
        raise ValueError("Global mask must contain at least one score")
    prune_count = validate_prune_count(prune_count, total)

    high_histogram = torch.zeros(1 << 16, dtype=torch.int64)
    for name in names:
        for keys in _ordered_score_key_chunks(scores[name]):
            high_histogram += torch.bincount(keys >> 16, minlength=1 << 16)

    if prune_count == 0 or prune_count == total:
        keep_value = prune_count == 0
        return {
            name: torch.full_like(scores[name], keep_value, dtype=torch.bool, device="cpu")
            for name in names
        }

    high_bucket, before_high = _bucket_for_rank(high_histogram, prune_count)
    low_histogram = torch.zeros(1 << 16, dtype=torch.int64)
    for name in names:
        for keys in _ordered_score_key_chunks(scores[name]):
            in_bucket = (keys >> 16) == high_bucket
            if in_bucket.any().item():
                low_histogram += torch.bincount(
                    keys[in_bucket] & 0xFFFF,
                    minlength=1 << 16,
                )
    low_bucket, before_low = _bucket_for_rank(
        low_histogram,
        prune_count - before_high,
    )
    threshold_key = (high_bucket << 16) | low_bucket
    strictly_lower = before_high + before_low
    ties_to_prune = prune_count - strictly_lower

    masks: MaskDict = {}
    remaining_ties = ties_to_prune
    for name in names:
        score = scores[name]
        flat_keep = torch.ones(score.numel(), dtype=torch.bool)
        offset = 0
        for keys in _ordered_score_key_chunks(score):
            remove = keys < threshold_key
            if remaining_ties:
                tied = torch.nonzero(keys == threshold_key, as_tuple=False).flatten()
                take = min(remaining_ties, tied.numel())
                if take:
                    remove[tied[:take]] = True
                    remaining_ties -= take
            flat_keep[offset : offset + keys.numel()] = ~remove
            offset += keys.numel()
        masks[name] = flat_keep.view_as(score)
    if remaining_ties:
        raise RuntimeError("Global threshold tie handling did not meet the prune budget")
    return masks


def _ordered_score_key_chunks(
    values: torch.Tensor,
    chunk_elements: int | None = None,
) -> Iterator[torch.Tensor]:
    """Yield order-preserving float32 bit keys for bounded-size CPU chunks."""
    if chunk_elements is None:
        chunk_elements = _GLOBAL_SELECTION_CHUNK_ELEMENTS
    if (
        isinstance(chunk_elements, bool)
        or not isinstance(chunk_elements, int)
        or chunk_elements < 1
    ):
        raise ValueError("chunk_elements must be a positive integer")
    for value_chunk in _ordered_value_chunks(values.detach(), chunk_elements):
        chunk = value_chunk.to(
            device="cpu",
            dtype=torch.float32,
        ).contiguous()
        if not torch.isfinite(chunk).all().item():
            raise ValueError("Mask scores must be finite")
        bits = chunk.view(torch.int32).to(torch.int64) & _FLOAT32_KEY_MASK
        negative = (bits & (1 << 31)) != 0
        keys = torch.where(
            negative,
            (~bits) & _FLOAT32_KEY_MASK,
            bits ^ (1 << 31),
        )
        # The existing comparator treats signed zeros as one stable tie group.
        keys[chunk == 0] = 1 << 31
        yield keys


def _ordered_value_chunks(
    values: torch.Tensor,
    chunk_elements: int,
) -> Iterator[torch.Tensor]:
    """Yield logical C-order views/copies without flattening a whole strided tensor."""
    if values.numel() == 0:
        return
    if values.numel() <= chunk_elements:
        yield values.reshape(-1)
        return

    # Slice the outermost dimension until any reshape copy is bounded by one
    # chunk. Recursing preserves the same order as ``values.reshape(-1)``.
    trailing_elements = values[0].numel()
    if trailing_elements <= chunk_elements:
        rows_per_chunk = max(1, chunk_elements // trailing_elements)
        for offset in range(0, values.shape[0], rows_per_chunk):
            yield values[offset : offset + rows_per_chunk].reshape(-1)
        return
    for index in range(values.shape[0]):
        yield from _ordered_value_chunks(values[index], chunk_elements)


def _bucket_for_rank(histogram: torch.Tensor, rank: int) -> tuple[int, int]:
    """Return the bucket containing one-based ``rank`` and count before it."""
    cumulative = histogram.cumsum(0)
    bucket = int(torch.searchsorted(cumulative, torch.tensor(rank)).item())
    before = 0 if bucket == 0 else int(cumulative[bucket - 1].item())
    return bucket, before


def _validate_tensor_mapping(
    values: object,
    label: str,
    *,
    require_bool: bool = False,
    require_real_floating: bool = False,
) -> None:
    """Validate a named tensor mapping with at least one materialized element."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if not values:
        raise ValueError(f"{label} must contain at least one tensor")

    invalid_keys = [name for name in values if not isinstance(name, str)]
    if invalid_keys:
        raise TypeError(f"{label} keys must be strings: {invalid_keys}")

    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label}[{name!r}] must be a torch.Tensor")
        if value.layout != torch.strided:
            raise TypeError(f"{label}[{name!r}] must be a dense strided tensor")
        if value.is_quantized:
            raise TypeError(f"{label}[{name!r}] must not be quantized")
        if bool(getattr(value, "is_nested", False)):
            raise TypeError(f"{label}[{name!r}] must not be nested")
        if value.device.type == "meta":
            raise TypeError(f"{label}[{name!r}] must be on a materialized device")
        if require_bool and value.dtype != torch.bool:
            raise TypeError(f"{label}[{name!r}] must have bool dtype")
        if require_real_floating and (not value.is_floating_point() or value.is_complex()):
            raise TypeError(f"{label}[{name!r}] must be real floating point")
        if require_real_floating and not torch.isfinite(value).all().item():
            raise ValueError(f"{label}[{name!r}] must contain only finite values")
    if not any(value.numel() for value in values.values()):
        raise ValueError(f"{label} must contain at least one element")


def _validate_matching_keys(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    label: str,
) -> None:
    """Require two validated named tensor mappings to have identical keys."""
    missing = sorted(set(left).difference(right))
    extra = sorted(set(right).difference(left))
    if missing or extra:
        raise ValueError(f"{label} keys must match: missing={missing}, extra={extra}")


def _validate_matching_tensor_metadata(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    label: str,
) -> None:
    """Require corresponding tensors to have matching shape and device."""
    for name in left:
        left_value = left[name]
        right_value = right[name]
        if left_value.shape != right_value.shape:
            raise ValueError(
                f"{label} shape for {name!r} must match: "
                f"{tuple(left_value.shape)} != {tuple(right_value.shape)}"
            )
        if left_value.device != right_value.device:
            raise ValueError(
                f"{label} device for {name!r} must match: "
                f"{left_value.device} != {right_value.device}"
            )


def mask_metrics(
    masks: Mapping[str, torch.Tensor],
    taylor_scores: Mapping[str, torch.Tensor],
) -> Dict[str, float | int]:
    """Return pruning count and finite Taylor proxy cost for a mask mapping."""
    _validate_tensor_mapping(masks, "masks", require_bool=True)
    _validate_tensor_mapping(
        taylor_scores,
        "taylor_scores",
        require_real_floating=True,
    )
    _validate_matching_keys(masks, taylor_scores, "Mask and Taylor score")
    _validate_matching_tensor_metadata(masks, taylor_scores, "Mask and Taylor score")

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
    """Return exact overlap metrics for two non-empty, compatible mask mappings."""
    _validate_tensor_mapping(left, "left masks", require_bool=True)
    _validate_tensor_mapping(right, "right masks", require_bool=True)
    _validate_matching_keys(left, right, "Mask")
    _validate_matching_tensor_metadata(left, right, "Mask")

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
        "pruned_jaccard": intersection / union if union else 0.0,
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
        name: current_state[name].to(device, non_blocking=True, copy=True) for name in names
    }
    reference_device = {
        name: reference_state[name].to(device, non_blocking=True, copy=True) for name in names
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
