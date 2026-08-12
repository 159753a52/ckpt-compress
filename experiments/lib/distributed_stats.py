"""Distributed sufficient statistics for parametric score fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ScoreMoments:
    """Sufficient statistics for one non-negative score population."""

    count: int
    total: float
    total_squared: float
    zero_count: int
    maximum: float

    def to_result_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "sum": self.total,
            "sum_squared": self.total_squared,
            "zero_count": self.zero_count,
            "maximum": self.maximum,
        }


def merge_score_moments(moments: Sequence[ScoreMoments]) -> ScoreMoments:
    """Merge independently computed populations without materializing values."""
    if not moments:
        return ScoreMoments(0, 0.0, 0.0, 0, 0.0)
    return ScoreMoments(
        count=sum(moment.count for moment in moments),
        total=sum(moment.total for moment in moments),
        total_squared=sum(moment.total_squared for moment in moments),
        zero_count=sum(moment.zero_count for moment in moments),
        maximum=max(moment.maximum for moment in moments),
    )


def score_moments(values: torch.Tensor) -> ScoreMoments:
    """Compute stable CPU float64 moments for one score tensor."""
    flat = values.detach().flatten().to(device="cpu", dtype=torch.float64)
    if flat.numel() and not torch.isfinite(flat).all().item():
        raise ValueError("Score values must be finite")
    if flat.numel() and (flat < 0).any().item():
        raise ValueError("Score values must be non-negative")
    return ScoreMoments(
        count=flat.numel(),
        total=flat.sum().item(),
        total_squared=flat.square().sum().item(),
        zero_count=int((flat == 0).sum().item()),
        maximum=flat.max().item() if flat.numel() else 0.0,
    )


def layer_score_moments(
    layers: Sequence[Sequence[str]],
    scores: Mapping[str, torch.Tensor],
) -> list[ScoreMoments]:
    """Compute one moment record per structural layer."""
    moments = []
    for names in layers:
        if not names:
            raise ValueError("Structural layers must be non-empty")
        missing = [name for name in names if name not in scores]
        if missing:
            raise ValueError(f"Missing scores for structural layer: {missing}")
        layer_moments = [score_moments(scores[name]) for name in names]
        combined = merge_score_moments(layer_moments)
        if combined.count == 0:
            raise ValueError("Structural layers must contain at least one score")
        moments.append(combined)
    return moments


def _collective_device(process_group=None) -> torch.device:
    backend = str(dist.get_backend(process_group)).lower()
    if "nccl" in backend:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL moment reduction requires a CUDA device")
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def reduce_score_moments(
    moments: Sequence[ScoreMoments],
    process_group=None,
) -> tuple[list[ScoreMoments], dict[str, object]]:
    """All-reduce layer moments when a distributed process group is active."""
    if not dist.is_available() or not dist.is_initialized():
        return list(moments), {
            "distributed": False,
            "world_size": 1,
            "backend": None,
            "communicated_scalars_per_rank": 0,
        }
    if not moments:
        raise ValueError("moments must not be empty")

    device = _collective_device(process_group)
    additive = torch.tensor(
        [
            [m.count, m.total, m.total_squared, m.zero_count]
            for m in moments
        ],
        dtype=torch.float64,
        device=device,
    )
    maxima = torch.tensor(
        [m.maximum for m in moments],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(additive, op=dist.ReduceOp.SUM, group=process_group)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=process_group)
    additive = additive.cpu()
    maxima = maxima.cpu()
    reduced = [
        ScoreMoments(
            count=int(round(row[0].item())),
            total=row[1].item(),
            total_squared=row[2].item(),
            zero_count=int(round(row[3].item())),
            maximum=maxima[index].item(),
        )
        for index, row in enumerate(additive)
    ]
    world_size = dist.get_world_size(process_group)
    return reduced, {
        "distributed": True,
        "world_size": world_size,
        "backend": str(dist.get_backend(process_group)),
        "communicated_scalars_per_rank": 5 * len(moments),
    }


__all__ = [
    "ScoreMoments",
    "layer_score_moments",
    "merge_score_moments",
    "reduce_score_moments",
    "score_moments",
]
