"""Shared exact-count mask construction for pruning workflows."""

from __future__ import annotations

import torch


def validate_prune_count(prune_count: int, score_count: int) -> int:
    """Normalize an integer prune count and validate it against capacity."""
    if isinstance(prune_count, bool):
        raise ValueError(f"prune_count must be an integer, got {prune_count}")
    try:
        normalized = int(prune_count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"prune_count must be an integer, got {prune_count}") from exc
    if normalized != prune_count:
        raise ValueError(f"prune_count must be an integer, got {prune_count}")
    if not 0 <= normalized <= score_count:
        raise ValueError(
            f"prune_count must be in [0, {score_count}], got {prune_count}"
        )
    return normalized


def exact_keep_mask(scores: torch.Tensor, prune_count: int) -> torch.Tensor:
    """Return a boolean mask that keeps all but exactly ``prune_count`` scores."""
    flat_scores = scores.detach().float().flatten()
    prune_count = validate_prune_count(prune_count, flat_scores.numel())
    if not torch.isfinite(flat_scores).all().item():
        raise ValueError("Pruning scores must be finite")
    if prune_count == 0:
        return torch.ones_like(scores, dtype=torch.bool)
    if prune_count == flat_scores.numel():
        return torch.zeros_like(scores, dtype=torch.bool)

    threshold = torch.kthvalue(flat_scores, prune_count).values
    pruned = flat_scores < threshold
    remaining = prune_count - int(pruned.sum().item())
    if remaining:
        tied = torch.nonzero(flat_scores == threshold, as_tuple=False).flatten()
        pruned[tied[:remaining]] = True
    if int(pruned.sum().item()) != prune_count:
        raise RuntimeError("Tie handling failed to produce the requested prune count")
    return (~pruned).reshape_as(scores)


__all__ = ["exact_keep_mask", "validate_prune_count"]
