"""Spectral budget-direction math for residual allocation probes."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from experiments.lib.residual_budget import bounded_largest_remainder_counts


def budget_tangent_dct_directions(
    layer_sizes: Sequence[int],
    rank: int,
) -> List[List[float]]:
    """Build low-frequency DCT directions orthogonal to the pruning budget."""
    if not layer_sizes or any(size <= 0 for size in layer_sizes):
        raise ValueError("Layer sizes must be positive")
    if not 1 <= rank < len(layer_sizes):
        raise ValueError("rank must be in [1, number of layers)")

    sizes = np.asarray(layer_sizes, dtype=np.float64)
    layer_indices = np.arange(len(layer_sizes), dtype=np.float64)
    directions = []
    for frequency in range(1, rank + 1):
        direction = np.cos(
            math.pi * (layer_indices + 0.5) * frequency / len(layer_sizes)
        )
        direction -= np.dot(sizes, direction) / sizes.sum()
        maximum = np.max(np.abs(direction))
        if maximum <= 1e-12:
            raise ValueError(f"Degenerate spectral direction at frequency {frequency}")
        directions.append((direction / maximum).tolist())
    return directions


def reconstruct_directional_gradient(
    design: Sequence[Sequence[float]],
    responses: Sequence[float],
) -> Tuple[List[float], Dict[str, float | int]]:
    """Return the minimum-norm gradient matching measured directional derivatives."""
    matrix = np.asarray(design, dtype=np.float64)
    values = np.asarray(responses, dtype=np.float64)
    if matrix.ndim != 2 or values.shape != (matrix.shape[0],):
        raise ValueError("Directional design and responses have incompatible shapes")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(values)):
        raise ValueError("Directional measurements must be finite")

    matrix_rank = int(np.linalg.matrix_rank(matrix))
    if matrix_rank != matrix.shape[0]:
        raise ValueError("Directional design must have full row rank")
    condition_number = float(np.linalg.cond(matrix))
    gradient, _, _, _ = np.linalg.lstsq(matrix, values, rcond=None)
    residual = matrix @ gradient - values
    return gradient.tolist(), {
        "rank": matrix_rank,
        "condition_number": condition_number,
        "response_residual_l2": float(np.linalg.norm(residual)),
    }


def directional_layer_counts(
    direction: Sequence[float],
    layer_sizes: Sequence[int],
    target: int,
    ratio: float,
    probe_radius: float,
    max_layer_ratio: float,
    sign: float,
) -> List[int]:
    if len(direction) != len(layer_sizes):
        raise ValueError("Direction and layer sizes must have the same length")
    lower_bounds = [0] * len(layer_sizes)
    upper_bounds = [int(math.floor(max_layer_ratio * size)) for size in layer_sizes]
    rates = [ratio + sign * probe_radius * value for value in direction]
    if any(rate < 0 or rate > max_layer_ratio for rate in rates):
        raise ValueError("Spectral probe leaves the feasible layer-rate box")
    real_counts = [size * rate for size, rate in zip(layer_sizes, rates)]
    return bounded_largest_remainder_counts(
        real_counts,
        target,
        lower_bounds,
        upper_bounds,
    )


__all__ = [
    "budget_tangent_dct_directions",
    "directional_layer_counts",
    "reconstruct_directional_gradient",
]
