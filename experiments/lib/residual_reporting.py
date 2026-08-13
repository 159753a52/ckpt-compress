"""Statistical reporting contract for paired residual recovery experiments."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Dict, Sequence, TypedDict

from scipy.stats import t as student_t

from experiments.lib.residual_protocol import (
    FIRST_ORDER_UNIFORM_METHOD,
    FIXED_METHODS,
    QUANTILE_METHOD_PREFIX,
    RESIDUAL_MAGNITUDE_UNIFORM_METHOD,
    SECOND_ORDER_UNIFORM_METHOD,
    SPECTRAL_METHOD_PREFIX,
    TAYLOR_EXACT_GLOBAL_METHOD,
    TAYLOR_METHOD_PREFIX,
    TAYLOR_PROBE_TRUST_METHOD,
    TAYLOR_UNIFORM_METHOD,
    TAYLOR_WEIBULL_MOM_METHOD,
)

Summary = Dict[str, object]
MethodSummary = Dict[str, Summary]
ComparisonSummary = Dict[str, Summary]


class AggregateResult(TypedDict):
    current_perplexity: Summary
    no_compression_final_perplexity: Summary
    methods: Dict[str, MethodSummary]
    paired_comparisons: Dict[str, ComparisonSummary]


@dataclass(frozen=True)
class _MethodMetrics:
    immediate_perplexity: float
    final_perplexity: float


@dataclass(frozen=True)
class _SeedMetrics:
    current_perplexity: float
    no_compression_final_perplexity: float
    methods: Dict[str, _MethodMetrics]


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{context} must be finite")
    return normalized


def _mapping(value: object, context: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _field(parent: Mapping[object, object], key: str, context: str) -> object:
    if key not in parent:
        raise ValueError(f"{context} is missing {key!r}")
    return parent[key]


def _perplexity(parent: Mapping[object, object], field: str, context: str) -> float:
    metrics = _mapping(_field(parent, field, context), f"{context}.{field}")
    perplexity = _finite_number(
        _field(metrics, "perplexity", f"{context}.{field}"),
        f"{context}.{field}.perplexity",
    )
    if perplexity <= 0:
        raise ValueError(f"{context}.{field}.perplexity must be positive")
    return perplexity


def _parse_seed(seed_key: str, raw_result: object) -> _SeedMetrics:
    if not isinstance(seed_key, str):
        raise ValueError(f"seed result key {seed_key!r} must be an integer string")
    try:
        seed = int(seed_key)
    except ValueError as exc:
        raise ValueError(f"seed result key {seed_key!r} must be an integer string") from exc
    if str(seed) != seed_key:
        raise ValueError(f"seed result key {seed_key!r} must use canonical integer form")

    context = f"seed_results[{seed_key!r}]"
    result = _mapping(raw_result, context)
    recorded_seed = _field(result, "seed", context)
    if isinstance(recorded_seed, bool) or not isinstance(recorded_seed, int):
        raise ValueError(f"{context}.seed must be an integer")
    if recorded_seed != seed:
        raise ValueError(f"{context}.seed={recorded_seed} does not match mapping key {seed_key!r}")

    raw_methods = _mapping(_field(result, "methods", context), f"{context}.methods")
    methods: Dict[str, _MethodMetrics] = {}
    for method, raw_metrics in raw_methods.items():
        if not isinstance(method, str) or not method:
            raise ValueError(f"{context}.methods keys must be non-empty strings")
        method_context = f"{context}.methods[{method!r}]"
        metrics = _mapping(raw_metrics, method_context)
        methods[method] = _MethodMetrics(
            immediate_perplexity=_perplexity(metrics, "immediate", method_context),
            final_perplexity=_perplexity(metrics, "final", method_context),
        )
    return _SeedMetrics(
        current_perplexity=_perplexity(result, "current", context),
        no_compression_final_perplexity=_perplexity(result, "no_compression_final", context),
        methods=methods,
    )


def summarize_values(values: Sequence[float]) -> Summary:
    if len(values) == 0:
        raise ValueError("values must not be empty")
    normalized = [_finite_number(value, f"values[{index}]") for index, value in enumerate(values)]
    mean = float(statistics.mean(normalized))
    if not math.isfinite(mean):
        raise ValueError("values produce a non-finite mean")
    if len(normalized) < 2:
        return {"values": normalized, "mean": mean, "std": None, "ci95": None}
    std = float(statistics.stdev(normalized))
    half_width = float(student_t.ppf(0.975, len(normalized) - 1)) * std / math.sqrt(len(normalized))
    if not math.isfinite(std) or not math.isfinite(half_width):
        raise ValueError("values produce non-finite summary statistics")
    ci95 = [mean - half_width, mean + half_width]
    if not all(math.isfinite(bound) for bound in ci95):
        raise ValueError("values produce non-finite confidence bounds")
    return {
        "values": normalized,
        "mean": mean,
        "std": std,
        "ci95": ci95,
    }


def aggregate(seed_results: Mapping[str, object]) -> AggregateResult:
    """Aggregate paired seed results without changing method insertion order."""
    if not seed_results:
        raise ValueError("seed_results must not be empty")
    parsed = {seed: _parse_seed(seed, result) for seed, result in seed_results.items()}
    seeds = sorted(parsed, key=int)
    method_names = list(parsed[seeds[0]].methods)
    expected_methods = set(method_names)
    missing_fixed = set(FIXED_METHODS).difference(expected_methods)
    if missing_fixed:
        raise ValueError(
            "Residual aggregate is missing required methods: " + ", ".join(sorted(missing_fixed))
        )
    for seed in seeds[1:]:
        actual_methods = set(parsed[seed].methods)
        if actual_methods != expected_methods:
            missing = sorted(expected_methods.difference(actual_methods))
            extra = sorted(actual_methods.difference(expected_methods))
            raise ValueError(
                f"Seed {seed} method set differs from seed {seeds[0]}: "
                f"missing={missing}, extra={extra}"
            )

    methods_output: Dict[str, MethodSummary] = {}
    comparisons_output: Dict[str, ComparisonSummary] = {}
    output: AggregateResult = {
        "current_perplexity": summarize_values([parsed[seed].current_perplexity for seed in seeds]),
        "no_compression_final_perplexity": summarize_values(
            [parsed[seed].no_compression_final_perplexity for seed in seeds]
        ),
        "methods": methods_output,
        "paired_comparisons": comparisons_output,
    }
    for method in method_names:
        immediate = [parsed[seed].methods[method].immediate_perplexity for seed in seeds]
        final = [parsed[seed].methods[method].final_perplexity for seed in seeds]
        magnitude_immediate = [
            parsed[seed].methods[RESIDUAL_MAGNITUDE_UNIFORM_METHOD].immediate_perplexity
            for seed in seeds
        ]
        magnitude_final = [
            parsed[seed].methods[RESIDUAL_MAGNITUDE_UNIFORM_METHOD].final_perplexity
            for seed in seeds
        ]
        methods_output[method] = {
            "immediate_perplexity": summarize_values(immediate),
            "final_perplexity": summarize_values(final),
            "paired_immediate_delta_vs_magnitude": summarize_values(
                [value - baseline for value, baseline in zip(immediate, magnitude_immediate)]
            ),
            "paired_final_delta_vs_magnitude": summarize_values(
                [value - baseline for value, baseline in zip(final, magnitude_final)]
            ),
        }
    comparisons = {
        "taylor_vs_first_order": (TAYLOR_UNIFORM_METHOD, FIRST_ORDER_UNIFORM_METHOD),
        "second_order_vs_first_order": (
            SECOND_ORDER_UNIFORM_METHOD,
            FIRST_ORDER_UNIFORM_METHOD,
        ),
        "weibull_vs_taylor_uniform": (TAYLOR_WEIBULL_MOM_METHOD, TAYLOR_UNIFORM_METHOD),
        "exact_global_vs_taylor_uniform": (
            TAYLOR_EXACT_GLOBAL_METHOD,
            TAYLOR_UNIFORM_METHOD,
        ),
        "probe_trust_vs_taylor_uniform": (
            TAYLOR_PROBE_TRUST_METHOD,
            TAYLOR_UNIFORM_METHOD,
        ),
        "probe_trust_vs_weibull": (
            TAYLOR_PROBE_TRUST_METHOD,
            TAYLOR_WEIBULL_MOM_METHOD,
        ),
    }
    for method in method_names:
        if not method.startswith((SPECTRAL_METHOD_PREFIX, QUANTILE_METHOD_PREFIX)):
            continue
        label = method.removeprefix(TAYLOR_METHOD_PREFIX)
        comparisons[f"{label}_vs_taylor_uniform"] = (method, TAYLOR_UNIFORM_METHOD)
        comparisons[f"{label}_vs_weibull"] = (method, TAYLOR_WEIBULL_MOM_METHOD)
        comparisons[f"{label}_vs_probe_trust"] = (method, TAYLOR_PROBE_TRUST_METHOD)
    for label, (left, right) in comparisons.items():
        comparison = {}
        for stage in ("immediate", "final"):
            deltas = [
                (
                    parsed[seed].methods[left].immediate_perplexity
                    - parsed[seed].methods[right].immediate_perplexity
                    if stage == "immediate"
                    else parsed[seed].methods[left].final_perplexity
                    - parsed[seed].methods[right].final_perplexity
                )
                for seed in seeds
            ]
            comparison[f"{stage}_perplexity_delta"] = summarize_values(deltas)
        comparisons_output[label] = comparison
    return output


__all__ = ["aggregate", "summarize_values"]
