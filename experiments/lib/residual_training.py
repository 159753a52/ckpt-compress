"""Training lifecycle and deterministic data partitions for residual experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from experiments.lib.residual_runtime import batch_hash, lm_loss, set_seed


@dataclass(frozen=True)
class SeedBatchPartition:
    """Deterministic, disjoint batch slices used by one paired seed run."""

    selected_pool_indices: List[int]
    pre_recovery: List[Mapping[str, torch.Tensor]]
    scoring: List[Mapping[str, torch.Tensor]]
    allocation_probe: List[Mapping[str, torch.Tensor]]
    allocation_selection: List[Mapping[str, torch.Tensor]]
    continuation: List[Mapping[str, torch.Tensor]]

    def data_hashes(self) -> Dict[str, str]:
        return {
            "pre_recovery": batch_hash(self.pre_recovery),
            "scoring": batch_hash(self.scoring),
            "allocation_probe": batch_hash(self.allocation_probe),
            "allocation_selection": batch_hash(self.allocation_selection),
            "continuation": batch_hash(self.continuation),
        }


def clone_model_state_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def seeded_training_batches(
    pool: Sequence[Mapping[str, torch.Tensor]],
    count: int,
    seed: int,
) -> Tuple[List[Mapping[str, torch.Tensor]], List[int]]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"count must be a non-negative integer, got {count}")
    if count > len(pool):
        raise ValueError(
            f"count cannot exceed pool size: count={count}, pool_size={len(pool)}"
        )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(pool), generator=generator)[:count].tolist()
    return [pool[index] for index in indices], indices


def _validate_non_negative_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}")
    return value


def partition_seed_batches(
    pool: Sequence[Mapping[str, torch.Tensor]],
    total_steps: int,
    recovery_step: int,
    hvp_batches: int,
    allocation_probe_batches: int,
    allocation_selection_batches: int,
    seed: int,
) -> SeedBatchPartition:
    total_steps = _validate_non_negative_count("total_steps", total_steps)
    recovery_step = _validate_non_negative_count("recovery_step", recovery_step)
    hvp_batches = _validate_non_negative_count("hvp_batches", hvp_batches)
    allocation_probe_batches = _validate_non_negative_count(
        "allocation_probe_batches", allocation_probe_batches
    )
    allocation_selection_batches = _validate_non_negative_count(
        "allocation_selection_batches", allocation_selection_batches
    )
    if recovery_step > total_steps:
        raise ValueError(
            f"recovery_step cannot exceed total_steps: "
            f"recovery_step={recovery_step}, total_steps={total_steps}"
        )
    selected_count = (
        total_steps
        + hvp_batches
        + allocation_probe_batches
        + allocation_selection_batches
    )
    selected, selected_indices = seeded_training_batches(pool, selected_count, seed)
    score_start = recovery_step
    probe_start = score_start + hvp_batches
    selection_start = probe_start + allocation_probe_batches
    continuation_start = selection_start + allocation_selection_batches
    continuation = selected[continuation_start:]
    if len(continuation) != total_steps - recovery_step:
        raise RuntimeError("Internal batch partition does not match continuation steps")
    return SeedBatchPartition(
        selected_pool_indices=selected_indices,
        pre_recovery=selected[:score_start],
        scoring=selected[score_start:probe_start],
        allocation_probe=selected[probe_start:selection_start],
        allocation_selection=selected[selection_start:continuation_start],
        continuation=continuation,
    )


def build_optimizer(
    model: nn.Module,
    state: Mapping,
    learning_rate: Optional[float] = None,
) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    optimizer.load_state_dict(state)
    if learning_rate is not None:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
            group["initial_lr"] = learning_rate
    return optimizer


def train_segment(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batches: Sequence[Mapping[str, torch.Tensor]],
    seed: int,
    device: str,
) -> Dict[str, object]:
    set_seed(seed)
    model.train()
    losses = []
    learning_rates = []
    started = time.perf_counter()
    for batch in batches:
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.zero_grad(set_to_none=True)
        loss = lm_loss(model, batch, device)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return {
        "steps": len(batches),
        "seconds": time.perf_counter() - started,
        "train_losses": losses,
        "learning_rates": learning_rates,
    }


__all__ = [
    "SeedBatchPartition",
    "build_optimizer",
    "clone_model_state_to_cpu",
    "partition_seed_batches",
    "seeded_training_batches",
    "train_segment",
]
