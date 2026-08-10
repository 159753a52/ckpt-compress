"""Exact budget rounding and bounded allocation for residual experiments."""

from __future__ import annotations

import math
from typing import List, Sequence


def largest_remainder_counts(
    real_counts: Sequence[float],
    target: int,
    capacities: Sequence[int],
) -> List[int]:
    counts = [
        min(int(math.floor(value)), capacity)
        for value, capacity in zip(real_counts, capacities)
    ]
    difference = target - sum(counts)
    if difference > 0:
        order = sorted(
            range(len(counts)),
            key=lambda index: (
                real_counts[index] - math.floor(real_counts[index]),
                -index,
            ),
            reverse=True,
        )
        while difference:
            progressed = False
            for index in order:
                if counts[index] < capacities[index]:
                    counts[index] += 1
                    difference -= 1
                    progressed = True
                    if difference == 0:
                        break
            if not progressed:
                raise RuntimeError("Layer capacities cannot meet the global pruning target")
    elif difference < 0:
        order = sorted(
            range(len(counts)),
            key=lambda index: (
                real_counts[index] - math.floor(real_counts[index]),
                -index,
            ),
        )
        while difference:
            progressed = False
            for index in order:
                if counts[index] > 0:
                    counts[index] -= 1
                    difference += 1
                    progressed = True
                    if difference == 0:
                        break
            if not progressed:
                raise RuntimeError("Could not round layer counts to the global target")
    return counts


def uniform_counts(layer_sizes: Sequence[int], target: int, ratio: float) -> List[int]:
    real = [ratio * size for size in layer_sizes]
    return largest_remainder_counts(real, target, list(layer_sizes))


def bounded_largest_remainder_counts(
    real_counts: Sequence[float],
    target: int,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> List[int]:
    if not (
        len(real_counts) == len(lower_bounds) == len(upper_bounds)
        and all(lower <= upper for lower, upper in zip(lower_bounds, upper_bounds))
    ):
        raise ValueError("Invalid bounded-rounding inputs")
    if not sum(lower_bounds) <= target <= sum(upper_bounds):
        raise ValueError("Bounds cannot meet the requested global target")

    shifted_real = [
        min(max(value, lower), upper) - lower
        for value, lower, upper in zip(real_counts, lower_bounds, upper_bounds)
    ]
    shifted_target = target - sum(lower_bounds)
    capacities = [upper - lower for lower, upper in zip(lower_bounds, upper_bounds)]
    shifted = largest_remainder_counts(shifted_real, shifted_target, capacities)
    return [value + lower for value, lower in zip(shifted, lower_bounds)]


def trust_region_counts(
    marginal_losses: Sequence[float],
    layer_sizes: Sequence[int],
    target: int,
    ratio: float,
    trust_radius: float,
    max_layer_ratio: float,
) -> List[int]:
    """Project a loss-decreasing rate step onto an exact, bounded budget."""
    if len(marginal_losses) != len(layer_sizes) or not layer_sizes:
        raise ValueError("Marginals and non-empty layer sizes must have the same length")
    if not 0 < trust_radius <= 1:
        raise ValueError("trust_radius must be in (0, 1]")
    if not all(math.isfinite(value) for value in marginal_losses):
        raise ValueError("Marginal losses must be finite")

    lower_rate = max(0.0, ratio - trust_radius)
    upper_rate = min(1.0, max_layer_ratio, ratio + trust_radius)
    lower_bounds = [int(math.ceil(lower_rate * size)) for size in layer_sizes]
    upper_bounds = [int(math.floor(upper_rate * size)) for size in layer_sizes]
    if any(lower > upper for lower, upper in zip(lower_bounds, upper_bounds)):
        raise ValueError("A layer is too small to represent the requested trust region")
    if not sum(lower_bounds) <= target <= sum(upper_bounds):
        raise ValueError("Trust region cannot meet the requested global target")

    # Convert dL/dp_l into comparable per-parameter marginals.
    per_parameter = [
        marginal / size for marginal, size in zip(marginal_losses, layer_sizes)
    ]
    spread = max(per_parameter) - min(per_parameter)
    if spread <= 1e-30:
        return bounded_largest_remainder_counts(
            [ratio * size for size in layer_sizes],
            target,
            lower_bounds,
            upper_bounds,
        )

    # Closed-form gradient step followed by weighted budget projection.
    step_size = 2.0 * trust_radius / spread

    def rates_at_shift(shift: float) -> List[float]:
        return [
            min(upper_rate, max(lower_rate, ratio - step_size * value + shift))
            for value in per_parameter
        ]

    low = -2.0
    high = 2.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        total = sum(
            size * rate for size, rate in zip(layer_sizes, rates_at_shift(middle))
        )
        if total < target:
            low = middle
        else:
            high = middle
    rates = rates_at_shift(0.5 * (low + high))
    real_counts = [size * rate for size, rate in zip(layer_sizes, rates)]
    return bounded_largest_remainder_counts(
        real_counts,
        target,
        lower_bounds,
        upper_bounds,
    )


__all__ = [
    "bounded_largest_remainder_counts",
    "largest_remainder_counts",
    "trust_region_counts",
    "uniform_counts",
]
