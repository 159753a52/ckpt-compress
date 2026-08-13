"""CPU-only: Compare distribution fits for real checkpoint weight magnitudes.

Loads a checkpoint, computes magnitude scores per layer, fits Weibull/Gamma/LogNormal,
and reports KS-D statistics. This is a magnitude-score diagnostic; it does not by
itself validate the paper's separate claim about Taylor/HVP damage scores.

Run on server:
    python scripts/validate_distribution.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats

# Setup path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"

import torch


def _fit_failure(layer_name, distribution, error):
    raise RuntimeError(f"{layer_name}: {distribution} fit failed: {error}") from error


def fit_and_compare(data, layer_name, max_samples=50000):
    """Fit Weibull, Gamma, LogNormal to positive data and compare KS-D."""
    data_pos = data[data > 0]
    if len(data_pos) < 100:
        return None

    if len(data_pos) > max_samples:
        rng = np.random.RandomState(42)
        data_pos = rng.choice(data_pos, max_samples, replace=False)

    mean_val = np.mean(data_pos)
    var_val = np.var(data_pos)
    if not np.isfinite(data_pos).all() or not np.isfinite(mean_val) or not np.isfinite(var_val):
        raise ValueError(f"{layer_name}: positive samples must be finite")
    if mean_val <= 0 or var_val <= 0:
        raise ValueError(f"{layer_name}: positive samples must have positive mean and variance")

    results = {}

    # Weibull
    try:
        wb_params = stats.weibull_min.fit(data_pos, floc=0)
        ks_d, ks_p = stats.kstest(data_pos, "weibull_min", args=wb_params)
        results["weibull"] = {
            "ks_d": ks_d,
            "ks_p": ks_p,
            "shape": wb_params[0],
            "scale": wb_params[2],
        }
    except Exception as error:
        _fit_failure(layer_name, "Weibull MLE", error)

    # Gamma (MoM)
    try:
        beta = var_val / mean_val
        alpha = mean_val / beta
        ks_d, ks_p = stats.kstest(data_pos, "gamma", args=(alpha, 0, beta))
        results["gamma_mom"] = {"ks_d": ks_d, "ks_p": ks_p, "alpha": alpha, "beta": beta}
    except Exception as error:
        _fit_failure(layer_name, "Gamma MoM", error)

    # Gamma (MLE)
    try:
        gm_params = stats.gamma.fit(data_pos, floc=0)
        ks_d, ks_p = stats.kstest(data_pos, "gamma", args=gm_params)
        results["gamma_mle"] = {"ks_d": ks_d, "ks_p": ks_p}
    except Exception as error:
        _fit_failure(layer_name, "Gamma MLE", error)

    # LogNormal
    try:
        ln_params = stats.lognorm.fit(data_pos, floc=0)
        ks_d, ks_p = stats.kstest(data_pos, "lognorm", args=ln_params)
        results["lognormal"] = {"ks_d": ks_d, "ks_p": ks_p}
    except Exception as error:
        _fit_failure(layer_name, "LogNormal MLE", error)

    return results


def analyze_checkpoint(ckpt_path, model_name):
    """Analyze distribution of magnitude scores for one checkpoint."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Filter prunable params (skip bias, LayerNorm, etc.)
    prunable = {}
    for name, tensor in state_dict.items():
        if tensor.dim() >= 2 and tensor.numel() > 100:
            if not any(
                skip in name for skip in ["bias", "LayerNorm", "ln_", "embed", "wte", "wpe"]
            ):
                prunable[name] = tensor

    print(f"Prunable layers: {len(prunable)}")
    total_params = sum(t.numel() for t in prunable.values())
    print(f"Total prunable params: {total_params:,}")

    all_results = {}
    summary = {"weibull_wins": 0, "gamma_wins": 0, "lognorm_wins": 0, "total": 0}

    for name, tensor in sorted(prunable.items()):
        data = tensor.abs().flatten().numpy().astype(np.float64)
        result = fit_and_compare(data, name)
        if result is None:
            continue

        all_results[name] = result
        summary["total"] += 1

        # Determine winner
        ks_values = {}
        for dist_name in ["weibull", "gamma_mom", "lognormal"]:
            if dist_name in result:
                ks_values[dist_name] = result[dist_name]["ks_d"]

        if ks_values:
            winner = min(ks_values, key=lambda name: ks_values[name])
            if winner == "weibull":
                summary["weibull_wins"] += 1
            elif winner == "gamma_mom":
                summary["gamma_wins"] += 1
            elif winner == "lognormal":
                summary["lognorm_wins"] += 1

        # Print layer details
        wb_d = result.get("weibull", {}).get("ks_d", float("inf"))
        gm_d = result.get("gamma_mom", {}).get("ks_d", float("inf"))
        ln_d = result.get("lognormal", {}).get("ks_d", float("inf"))
        best = min(wb_d, gm_d, ln_d)
        marker = lambda d: " *" if d == best else ""
        short_name = name.split(".")[-2] + "." + name.split(".")[-1] if "." in name else name
        print(
            f"  {short_name:40s}  Weibull={wb_d:.4f}{marker(wb_d)}  Gamma={gm_d:.4f}{marker(gm_d)}  LogN={ln_d:.4f}{marker(ln_d)}"
        )

    print(
        f"\nSummary: Weibull wins {summary['weibull_wins']}/{summary['total']}, "
        f"Gamma wins {summary['gamma_wins']}/{summary['total']}, "
        f"LogNormal wins {summary['lognorm_wins']}/{summary['total']}"
    )

    # KS-D ranges
    wb_ds = [r["weibull"]["ks_d"] for r in all_results.values() if "weibull" in r]
    gm_ds = [r["gamma_mom"]["ks_d"] for r in all_results.values() if "gamma_mom" in r]
    ln_ds = [r["lognormal"]["ks_d"] for r in all_results.values() if "lognormal" in r]

    if wb_ds:
        print(
            f"Weibull KS-D range: [{min(wb_ds):.4f}, {max(wb_ds):.4f}], mean={np.mean(wb_ds):.4f}"
        )
    if gm_ds:
        print(
            f"Gamma   KS-D range: [{min(gm_ds):.4f}, {max(gm_ds):.4f}], mean={np.mean(gm_ds):.4f}"
        )
    if ln_ds:
        print(
            f"LogNorm KS-D range: [{min(ln_ds):.4f}, {max(ln_ds):.4f}], mean={np.mean(ln_ds):.4f}"
        )

    return all_results, summary


def _build_parser() -> argparse.ArgumentParser:
    """Build the fixed-checkpoint diagnostic parser without loading inputs."""
    return argparse.ArgumentParser(description=__doc__)


def main(argv: Sequence[str] | None = None) -> None:
    _build_parser().parse_args(argv)
    checkpoints = [
        ("checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt", "GPT-2 Medium"),
        (
            "checkpoints/bert_large_sst2_1000steps/checkpoint_step_1000_final.pt",
            "BERT-Large (SST-2)",
        ),
        (
            "checkpoints/bert_large_mnli_1000steps/checkpoint_step_1000_final.pt",
            "BERT-Large (MNLI)",
        ),
    ]

    all_summaries = {}
    for ckpt_path, name in checkpoints:
        full_path = ROOT / ckpt_path
        if full_path.exists():
            results, summary = analyze_checkpoint(str(full_path), name)
            all_summaries[name] = summary
        else:
            print(f"Checkpoint not found: {full_path}")

    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    for name, s in all_summaries.items():
        print(
            f"{name}: Weibull={s['weibull_wins']}/{s['total']}, Gamma={s['gamma_wins']}/{s['total']}, LogN={s['lognorm_wins']}/{s['total']}"
        )


if __name__ == "__main__":
    main()
