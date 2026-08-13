"""Pure numerical core for empirical quantile residual allocation.

The tensor adapter lives in ``residual_quantile`` and is reexported through
``residual_allocation`` for compatibility. This module only owns the NumPy
optimization after Taylor scores have been materialized in stable order.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]

_BUDGET_BISECTION_STEPS = 70
_STEP_SAFETY_FACTOR = 0.99
_ZERO_TOLERANCE = 1e-30
_INTEGER_ROUNDING_TOLERANCE = 1e-9


@dataclass(frozen=True)
class _EmpiricalAllocationProblem:
    sorted_scores: Sequence[FloatArray]
    layer_sizes: npt.NDArray[np.int64]
    lower_bounds: npt.NDArray[np.int64]
    upper_bounds: npt.NDArray[np.int64]
    target: int
    score_normalizers: FloatArray


def _prox_empirical_count(
    y: float,
    alpha: float,
    sorted_scores: FloatArray,
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Evaluate the scalar prox of an empirical cumulative-score curve."""
    low = 0
    high = len(sorted_scores)
    while low < high:
        middle = (low + high) // 2
        upper_breakpoint = middle + 1 + alpha * float(sorted_scores[middle])
        if upper_breakpoint < y:
            low = middle + 1
        else:
            high = middle

    if low == len(sorted_scores):
        count = float(len(sorted_scores))
    else:
        lower_breakpoint = low + alpha * float(sorted_scores[low])
        count = float(low) if y < lower_breakpoint else y - alpha * float(sorted_scores[low])
    return min(float(upper_bound), max(float(lower_bound), count))


def _budget_proximal_rates(
    rates: FloatArray,
    step_size: float,
    problem: _EmpiricalAllocationProblem,
) -> FloatArray:
    """Apply the exact empirical-cost prox under one weighted budget."""

    def counts_at_multiplier(multiplier: float) -> FloatArray:
        counts = []
        for rate, scores, size, lower, upper, normalizer in zip(
            rates,
            problem.sorted_scores,
            problem.layer_sizes,
            problem.lower_bounds,
            problem.upper_bounds,
            problem.score_normalizers,
        ):
            alpha = step_size * float(size) ** 2 / float(normalizer)
            y = float(size) * rate - step_size * float(size) ** 2 * multiplier
            counts.append(
                _prox_empirical_count(
                    y,
                    alpha,
                    scores,
                    int(lower),
                    int(upper),
                )
            )
        return np.asarray(counts, dtype=np.float64)

    low = -1.0
    high = 1.0
    while counts_at_multiplier(low).sum() < problem.target:
        low *= 2.0
    while counts_at_multiplier(high).sum() > problem.target:
        high *= 2.0
    for _ in range(_BUDGET_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if counts_at_multiplier(middle).sum() > problem.target:
            low = middle
        else:
            high = middle
    counts = counts_at_multiplier(0.5 * (low + high))
    return np.asarray(counts / problem.layer_sizes, dtype=np.float64)


def _empirical_cost(sorted_scores: FloatArray, count: float) -> float:
    integer = min(int(math.floor(count)), len(sorted_scores))
    value = float(np.sum(sorted_scores[:integer], dtype=np.float64))
    if integer < len(sorted_scores):
        value += (count - integer) * float(sorted_scores[integer])
    return value


def _integerize_quantile_smooth_counts(
    continuous_counts: FloatArray,
    smoothness: float,
    problem: _EmpiricalAllocationProblem,
) -> Tuple[List[int], Dict[str, float | int]]:
    """Round a continuous chain solution with an exact-budget two-choice DP."""
    candidates = []
    unary_costs = []
    for value, scores, lower, upper, normalizer in zip(
        continuous_counts,
        problem.sorted_scores,
        problem.lower_bounds,
        problem.upper_bounds,
        problem.score_normalizers,
    ):
        floor_count = min(
            int(upper),
            max(
                int(lower),
                int(math.floor(value + _INTEGER_ROUNDING_TOLERANCE)),
            ),
        )
        ceil_count = min(
            int(upper),
            max(
                int(lower),
                int(math.ceil(value - _INTEGER_ROUNDING_TOLERANCE)),
            ),
        )
        options = sorted(set((floor_count, ceil_count)))
        candidates.append(options)
        unary_costs.append(
            [_empirical_cost(scores, count) / float(normalizer) for count in options]
        )

    base_total = sum(options[0] for options in candidates)
    needed = problem.target - base_total
    optional_layers = sum(len(options) == 2 for options in candidates)
    if not 0 <= needed <= optional_layers:
        raise RuntimeError("Continuous budget residual cannot be rounded by floor/ceil DP")

    states: Dict[Tuple[int, int], Tuple[float, List[int]]] = {}
    for choice, count in enumerate(candidates[0]):
        used = choice if len(candidates[0]) == 2 else 0
        states[(used, choice)] = (unary_costs[0][choice], [count])

    for layer_index in range(1, len(candidates)):
        next_states: Dict[Tuple[int, int], Tuple[float, List[int]]] = {}
        for (used, previous_choice), (cost, path) in states.items():
            previous_count = candidates[layer_index - 1][previous_choice]
            previous_rate = previous_count / float(problem.layer_sizes[layer_index - 1])
            for choice, count in enumerate(candidates[layer_index]):
                increment = choice if len(candidates[layer_index]) == 2 else 0
                new_used = used + increment
                if new_used > needed:
                    continue
                rate = count / float(problem.layer_sizes[layer_index])
                pair_cost = 0.5 * smoothness * (rate - previous_rate) ** 2
                new_cost = cost + unary_costs[layer_index][choice] + pair_cost
                key = (new_used, choice)
                if key not in next_states or new_cost < next_states[key][0]:
                    next_states[key] = (new_cost, path + [count])
        states = next_states

    feasible = [value for (used, _), value in states.items() if used == needed]
    if not feasible:
        raise RuntimeError("Integer rounding DP could not meet the exact budget")
    objective, counts = min(feasible, key=lambda item: item[0])
    if sum(counts) != problem.target:
        raise RuntimeError("Integer rounding DP violated the exact budget")
    return counts, {
        "rounding_objective": float(objective),
        "rounded_up_layers": int(needed),
        "candidate_layers": int(optional_layers),
    }


def solve_quantile_smooth_counts(
    sorted_scores: Sequence[FloatArray],
    layer_sizes: npt.NDArray[np.int64],
    uniform_layer_counts: Sequence[int],
    lower_bounds: npt.NDArray[np.int64],
    upper_bounds: npt.NDArray[np.int64],
    target: int,
    trust_radius: float,
    smoothness_values: Sequence[float],
    normalization: str,
    max_iterations: int,
    tolerance: float,
) -> Tuple[Dict[float, List[int]], Dict[str, object]]:
    """Solve prevalidated empirical curves for one or more smoothness values."""
    target_quantiles = [
        float(values[min(max(count, 1), len(values)) - 1])
        for values, count in zip(sorted_scores, uniform_layer_counts)
    ]
    score_scale = float(np.median(np.abs(target_quantiles)))
    if score_scale <= _ZERO_TOLERANCE:
        score_scale = max(
            max(abs(float(values[0])), abs(float(values[-1]))) for values in sorted_scores
        )
    if score_scale <= _ZERO_TOLERANCE:
        score_scale = max(score_scale, 1.0)

    uniform_proxy_costs = np.asarray(
        [
            _empirical_cost(values, count)
            for values, count in zip(sorted_scores, uniform_layer_counts)
        ],
        dtype=np.float64,
    )
    if normalization == "global":
        score_normalizers: FloatArray = np.full(
            len(layer_sizes),
            float(layer_sizes.sum()) * score_scale,
            dtype=np.float64,
        )
    else:
        if np.any(np.abs(uniform_proxy_costs) <= _ZERO_TOLERANCE):
            raise ValueError("Relative empirical costs require nonzero uniform proxy costs")
        score_normalizers = np.asarray(
            len(layer_sizes) * np.abs(uniform_proxy_costs),
            dtype=np.float64,
        )

    problem = _EmpiricalAllocationProblem(
        sorted_scores=sorted_scores,
        layer_sizes=layer_sizes,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        target=target,
        score_normalizers=score_normalizers,
    )
    difference: FloatArray = np.diff(np.eye(len(layer_sizes), dtype=np.float64), axis=0)
    laplacian = difference.T @ difference
    laplacian_largest_eigenvalue = float(np.linalg.eigvalsh(laplacian)[-1])
    initial_rates = np.asarray(uniform_layer_counts, dtype=np.float64) / layer_sizes

    counts_by_smoothness = {}
    solutions = {}
    solve_started = time.perf_counter()
    for smoothness in smoothness_values:
        if laplacian_largest_eigenvalue <= _ZERO_TOLERANCE:
            # A one-layer chain has no smoothness term; its budget fixes the rate.
            rates = np.asarray([target / float(layer_sizes[0])], dtype=np.float64)
            iteration = 0
            residual = 0.0
            converged = True
        else:
            step_size = _STEP_SAFETY_FACTOR / (smoothness * laplacian_largest_eigenvalue)
            rates = initial_rates.copy()
            extrapolated = rates.copy()
            acceleration = 1.0
            residual = math.inf
            converged = False

            for iteration in range(1, max_iterations + 1):
                gradient = smoothness * (laplacian @ extrapolated)
                candidate = _budget_proximal_rates(
                    extrapolated - step_size * gradient,
                    step_size,
                    problem,
                )
                residual = float(np.max(np.abs(candidate - rates)))
                next_acceleration = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * acceleration**2))
                if np.dot(extrapolated - candidate, candidate - rates) > 0:
                    next_acceleration = 1.0
                    next_extrapolated = candidate.copy()
                else:
                    next_extrapolated = candidate + ((acceleration - 1.0) / next_acceleration) * (
                        candidate - rates
                    )
                rates = candidate
                extrapolated = next_extrapolated
                acceleration = next_acceleration
                if residual <= tolerance:
                    converged = True
                    break

        continuous_counts = rates * layer_sizes
        counts, rounding = _integerize_quantile_smooth_counts(
            continuous_counts,
            smoothness,
            problem,
        )
        final_rates = np.asarray(counts, dtype=np.float64) / layer_sizes
        layer_proxy_costs = [
            _empirical_cost(values, count) for values, count in zip(sorted_scores, counts)
        ]
        proxy_cost = sum(layer_proxy_costs)
        normalized_proxy_cost = sum(
            cost / float(normalizer)
            for cost, normalizer in zip(layer_proxy_costs, score_normalizers)
        )
        smooth_penalty = 0.5 * smoothness * float(np.sum(np.diff(final_rates) ** 2))
        counts_by_smoothness[smoothness] = counts
        solutions[str(smoothness)] = {
            "counts": counts,
            "rates": final_rates.tolist(),
            "continuous_rates": rates.tolist(),
            "iterations": iteration,
            "converged": converged,
            "fixed_point_residual": residual,
            "continuous_budget_residual": float(np.dot(rates, layer_sizes) - target),
            "proxy_cost": float(proxy_cost),
            "normalized_proxy_cost": float(normalized_proxy_cost),
            "smoothness_penalty": smooth_penalty,
            "objective": float(normalized_proxy_cost + smooth_penalty),
            **rounding,
        }

    return counts_by_smoothness, {
        "solve_seconds": time.perf_counter() - solve_started,
        "smoothness_values": list(smoothness_values),
        "trust_radius": trust_radius,
        "difference_order": 1,
        "normalization": normalization,
        "score_scale": score_scale,
        "score_normalizers": score_normalizers.tolist(),
        "uniform_proxy_costs": uniform_proxy_costs.tolist(),
        "target_quantiles": target_quantiles,
        "solutions": solutions,
    }


_solve_quantile_smooth_counts = solve_quantile_smooth_counts

__all__ = ["solve_quantile_smooth_counts"]
