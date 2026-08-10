"""Statistical reporting contract for paired residual recovery experiments."""

from __future__ import annotations

import math
import statistics
from typing import Dict, Mapping, Sequence

from scipy.stats import t as student_t


def summarize_values(values: Sequence[float]) -> Dict[str, object]:
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
            seed_results[seed]["methods"]["residual_magnitude_uniform"]["immediate"][
                "perplexity"
            ]
            for seed in seeds
        ]
        magnitude_final = [
            seed_results[seed]["methods"]["residual_magnitude_uniform"]["final"][
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
        "taylor_vs_first_order": ("taylor_uniform", "first_order_uniform"),
        "second_order_vs_first_order": ("second_order_uniform", "first_order_uniform"),
        "weibull_vs_taylor_uniform": ("taylor_weibull_mom", "taylor_uniform"),
        "exact_global_vs_taylor_uniform": ("taylor_exact_global", "taylor_uniform"),
        "probe_trust_vs_taylor_uniform": ("taylor_probe_trust", "taylor_uniform"),
        "probe_trust_vs_weibull": ("taylor_probe_trust", "taylor_weibull_mom"),
    }
    for method in method_names:
        if not method.startswith(("taylor_spectral_k", "taylor_quantile_")):
            continue
        label = method.removeprefix("taylor_")
        comparisons[f"{label}_vs_taylor_uniform"] = (method, "taylor_uniform")
        comparisons[f"{label}_vs_weibull"] = (method, "taylor_weibull_mom")
        comparisons[f"{label}_vs_probe_trust"] = (method, "taylor_probe_trust")
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
