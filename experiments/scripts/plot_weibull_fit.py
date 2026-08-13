"""Fit representative damage-score layers and render empirical/theoretical CDFs."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt
import torch
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_DISK_SPACE_CHECK", "1")

from experiments.lib.data import cache_batches, get_data_loaders
from experiments.lib.importance_compare.scoring import compute_scores_by_method
from experiments.lib.models import load_model

DEFAULT_OUTPUT_PATH = (
    ROOT / "results" / "paper_results" / "figures" / "weibull_fit_visualization.png"
)
PREFERRED_LAYER_TOKENS = (
    "h.0.attn.c_attn",
    "h.0.mlp.c_fc",
    "h.5.attn.c_attn",
    "h.10.mlp.c_proj",
)


def select_representative_layers(
    scores: Mapping[str, torch.Tensor],
    *,
    limit: int = 4,
) -> list[str]:
    """Select preferred GPT-2 layers, filling gaps in stable input order."""
    if limit <= 0:
        raise ValueError("layer limit must be positive")
    if not scores:
        raise ValueError("damage scores must include at least one layer")

    names = list(scores)
    selected: list[str] = []
    for token in PREFERRED_LAYER_TOKENS:
        match = next((name for name in names if token in name and name not in selected), None)
        if match is not None:
            selected.append(match)
        if len(selected) == limit:
            return selected
    selected.extend(name for name in names if name not in selected)
    return selected[:limit]


def positive_finite_samples(
    scores: torch.Tensor,
    layer_name: str,
    *,
    max_samples: int = 50_000,
    seed: int = 42,
) -> npt.NDArray[np.float64]:
    """Return deterministic positive samples, rejecting invalid fit inputs."""
    if max_samples <= 1:
        raise ValueError("max_samples must be greater than one")
    values = scores.detach().flatten().to(dtype=torch.float64, device="cpu").numpy()
    if values.size == 0:
        raise ValueError(f"{layer_name}: damage scores are empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{layer_name}: damage scores must be finite")
    positive = values[values > 0]
    if positive.size < 2:
        raise ValueError(f"{layer_name}: at least two positive damage scores are required")
    if float(np.var(positive)) <= 0:
        raise ValueError(f"{layer_name}: positive damage scores must have non-zero variance")
    if positive.size > max_samples:
        rng = np.random.default_rng(seed)
        positive = rng.choice(positive, max_samples, replace=False)
    return np.asarray(positive, dtype=np.float64)


def _fit_distributions(data: np.ndarray) -> dict[str, tuple[tuple[float, ...], float]]:
    weibull = tuple(float(value) for value in stats.weibull_min.fit(data, floc=0))
    gamma_mean = float(data.mean())
    gamma_variance = float(data.var())
    gamma_shape = gamma_mean**2 / gamma_variance
    gamma_scale = gamma_variance / gamma_mean
    gamma = (gamma_shape, 0.0, gamma_scale)
    lognormal = tuple(float(value) for value in stats.lognorm.fit(data, floc=0))
    exponential = tuple(float(value) for value in stats.expon.fit(data, floc=0))
    return {
        "weibull": (weibull, float(stats.kstest(data, "weibull_min", args=weibull).statistic)),
        "gamma": (gamma, float(stats.kstest(data, "gamma", args=gamma).statistic)),
        "lognormal": (
            lognormal,
            float(stats.kstest(data, "lognorm", args=lognormal).statistic),
        ),
        "exponential": (
            exponential,
            float(stats.kstest(data, "expon", args=exponential).statistic),
        ),
    }


def render_weibull_fit(
    scores: Mapping[str, torch.Tensor],
    output_path: Path,
    *,
    model_name: str,
    max_samples: int = 50_000,
    seed: int = 42,
) -> Path:
    """Render distribution fits for selected layers and return *output_path*."""
    selected = select_representative_layers(scores)
    columns = 2
    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(14, 5 * rows), squeeze=False)
    figure.suptitle(
        f"Weibull Distribution Fitting on Damage Scores ({model_name})",
        fontsize=14,
        fontweight="bold",
    )
    try:
        for axis, name in zip(axes.flat, selected):
            data = positive_finite_samples(scores[name], name, max_samples=max_samples, seed=seed)
            fitted = _fit_distributions(data)
            sorted_data = np.sort(data)
            empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            step = max(1, len(sorted_data) // 2_000)
            axis.plot(
                sorted_data[::step],
                empirical_cdf[::step],
                color="steelblue",
                linewidth=2,
                label="Empirical CDF",
            )
            x_values = np.linspace(sorted_data[0], sorted_data[-1], 1_000)
            curves = (
                ("weibull", stats.weibull_min, "Weibull", "red", "-", 2.5),
                ("gamma", stats.gamma, "Gamma", "orange", "--", 2.0),
                ("lognormal", stats.lognorm, "LogNormal", "green", ":", 1.5),
                ("exponential", stats.expon, "Exponential", "purple", "-.", 1.5),
            )
            for key, distribution, label, color, linestyle, width in curves:
                parameters, ks_statistic = fitted[key]
                axis.plot(
                    x_values,
                    distribution.cdf(x_values, *parameters),
                    color=color,
                    linewidth=width,
                    linestyle=linestyle,
                    label=f"{label} (KS-D={ks_statistic:.4f})",
                )
            short_name = name.replace("transformer.", "").replace(".weight", "")
            axis.set_title(short_name, fontsize=11)
            axis.set_xlabel("Damage Score")
            axis.set_ylabel("CDF")
            axis.legend(fontsize=8, loc="lower right")
            axis.set_ylim(0, 1.05)
        for unused_axis in list(axes.flat)[len(selected) :]:
            unused_axis.set_visible(False)

        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output_path


def plot_weibull_fit(
    *,
    model_name: str = "gpt2-small",
    dataset_name: str = "wikitext2",
    device: str | None = None,
    batch_size: int = 2,
    sequence_length: int = 128,
    train_batches: int = 10,
    hvp_batches: int = 2,
    data_dir: Path = ROOT / "data",
    output_path: Path = DEFAULT_OUTPUT_PATH,
    max_samples: int = 50_000,
    seed: int = 42,
) -> Path:
    """Load inputs, compute HVP scores, and render the diagnostic figure."""
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {selected_device}")
    print("Loading model...")
    model, model_family = load_model(model_name, device=selected_device)

    print("Loading data...")
    train_loader, _, task_type = get_data_loaders(
        model_name,
        dataset_name,
        batch_size=batch_size,
        seq_length=sequence_length,
        data_dir=str(data_dir),
    )
    cached_train = cache_batches(train_loader, train_batches, task_type)
    print("Computing damage scores (block-wise HVP)...")
    score_cache = compute_scores_by_method(
        model,
        cached_train,
        task_type,
        methods=["second-order-hvp"],
        alpha=0.5,
        hvp_batches=hvp_batches,
        hvp_mode="block",
        model_family=model_family,
    )
    output = render_weibull_fit(
        score_cache["second-order-hvp"],
        output_path,
        model_name=model_name,
        max_samples=max_samples,
        seed=seed,
    )
    print(f"Figure saved to: {output}")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-batches", type=int, default=10)
    parser.add_argument("--hvp-batches", type=int, default=2)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plot_weibull_fit(
        model_name=args.model,
        dataset_name=args.dataset,
        device=args.device,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        train_batches=args.train_batches,
        hvp_batches=args.hvp_batches,
        data_dir=args.data_dir.resolve(),
        output_path=args.output.resolve(),
        max_samples=args.max_samples,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
