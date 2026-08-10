"""Statistical reporting contract for paired residual recovery experiments."""

from __future__ import annotations

import math
import statistics
from typing import Dict, Mapping, Sequence

from scipy.stats import t as student_t

from experiments.lib.residual_protocol import (
    FIRST_ORDER_UNIFORM_METHOD,
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


def summarize_values(values: Sequence[float]) -> Dict[str, object]:
    if len(values) == 0:
        raise ValueError("values must not be empty")
    mean = statistics.mean(values)
    if len(values) < 2:
        return {"values": list(values), "mean": mean, "std": None, "ci95": None}
    std = statistics.stdev(values)
    half_width = float(student_t.ppf(0.975, len(values) - 1)) * std / math.sqrt(len(values))
    return {
        "values": list(values),
        "mean": mean,
        "std": std,
        "ci95": [mean - half_width, mean + half_width],
    }


def aggregate(seed_results: Mapping[str, Mapping]) -> Dict[str, object]:
    """Aggregate paired seed results without changing method insertion order."""
    if not seed_results:
        raise ValueError("seed_results must not be empty")
    seeds = sorted(seed_results, key=int)
    method_names = list(seed_results[seeds[0]]["methods"])
    output: Dict[str, object] = {
        "current_perplexity": summarize_values(
            [seed_results[seed]["current"]["perplexity"] for seed in seeds]
        ),
        "no_compression_final_perplexity": summarize_values(
            [seed_results[seed]["no_compression_final"]["perplexity"] for seed in seeds]
        ),
        "methods": {},
    }
    for method in method_names:
        immediate = [
            seed_results[seed]["methods"][method]["immediate"]["perplexity"]
            for seed in seeds
        ]
        final = [
            seed_results[seed]["methods"][method]["final"]["perplexity"]
            for seed in seeds
        ]
        magnitude_immediate = [
            seed_results[seed]["methods"][RESIDUAL_MAGNITUDE_UNIFORM_METHOD][
                "immediate"
            ]["perplexity"]
            for seed in seeds
        ]
        magnitude_final = [
            seed_results[seed]["methods"][RESIDUAL_MAGNITUDE_UNIFORM_METHOD]["final"][
                "perplexity"
            ]
            for seed in seeds
        ]
        output["methods"][method] = {
            "immediate_perplexity": summarize_values(immediate),
            "final_perplexity": summarize_values(final),
            "paired_immediate_delta_vs_magnitude": summarize_values(
                [value - baseline for value, baseline in zip(immediate, magnitude_immediate)]
            ),
            "paired_final_delta_vs_magnitude": summarize_values(
                [value - baseline for value, baseline in zip(final, magnitude_final)]
            ),
        }
    output["paired_comparisons"] = {}
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
                seed_results[seed]["methods"][left][stage]["perplexity"]
                - seed_results[seed]["methods"][right][stage]["perplexity"]
                for seed in seeds
            ]
            comparison[f"{stage}_perplexity_delta"] = summarize_values(deltas)
        output["paired_comparisons"][label] = comparison
    return output


__all__ = ["aggregate", "summarize_values"]
