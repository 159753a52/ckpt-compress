"""Generate the SST-2 sparsity/quality Pareto figure from the recorded sweep."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "paper_results" / "figures"

ORACLE_ACCURACY = 92.78
SST2_SWEEP: dict[str, dict[str, object]] = {
    "Magnitude + Uniform": {
        "sparsity": [0.10, 0.20, 0.30, 0.40],
        "accuracy": [92.43, 92.20, 91.86, 91.97],
        "color": "#1976D2",
        "marker": "s",
        "linestyle": "--",
    },
    "First-order + Uniform": {
        "sparsity": [0.10, 0.20, 0.30, 0.40],
        "accuracy": [92.55, 91.97, 92.09, 90.48],
        "color": "#388E3C",
        "marker": "^",
        "linestyle": ":",
    },
    "Magnitude + Weibull (Ours)": {
        "sparsity": [0.10, 0.20, 0.30, 0.40],
        "accuracy": [92.20, 92.43, 91.63, 91.97],
        "color": "#D32F2F",
        "marker": "o",
        "linestyle": "-",
    },
}


def _numeric_series(method: str, values: object, field: str) -> npt.NDArray[np.float64]:
    try:
        series = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{method}: {field} must be numeric") from error
    if series.ndim != 1 or series.size == 0:
        raise ValueError(f"{method}: {field} must be a non-empty one-dimensional series")
    if not np.isfinite(series).all():
        raise ValueError(f"{method}: {field} must contain only finite values")
    return np.asarray(series, dtype=np.float64)


def plot_sst2_pareto(
    output_dir: Path,
    *,
    sweep: Mapping[str, Mapping[str, object]] = SST2_SWEEP,
    oracle_accuracy: float = ORACLE_ACCURACY,
) -> tuple[Path, Path]:
    """Render the Pareto curve and return its PDF and PNG paths."""
    if not sweep:
        raise ValueError("SST-2 sweep must include at least one method")
    if not np.isfinite(oracle_accuracy):
        raise ValueError("oracle accuracy must be finite")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "figure.dpi": 300,
        }
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    try:
        axis.axhline(
            y=oracle_accuracy,
            color="gray",
            linestyle="-",
            linewidth=1,
            alpha=0.5,
            label=f"No compression ({oracle_accuracy}%)",
        )
        for name, record in sweep.items():
            sparsity = _numeric_series(name, record.get("sparsity"), "sparsity")
            accuracy = _numeric_series(name, record.get("accuracy"), "accuracy")
            if sparsity.size != accuracy.size:
                raise ValueError(f"{name}: sparsity and accuracy lengths differ")
            axis.plot(
                sparsity * 100,
                accuracy,
                color=str(record.get("color", "black")),
                marker=str(record.get("marker", "o")),
                markersize=8,
                linestyle=str(record.get("linestyle", "-")),
                linewidth=2,
                label=name,
            )

        axis.set_xlabel("Sparsity Ratio (%)")
        axis.set_ylabel("SST-2 Accuracy (%)")
        axis.set_title(
            "BERT-Large SST-2: Accuracy vs Sparsity (K=5 FT cycles)",
            fontweight="bold",
        )
        axis.legend(loc="lower left", framealpha=0.9)
        axis.set_xlim(5, 45)
        axis.set_ylim(89.5, 93.5)
        axis.grid(True, alpha=0.2)
        axis.annotate(
            "First-order\ncollapses",
            xy=(40, 90.48),
            xytext=(33, 89.8),
            fontsize=9,
            color="#388E3C",
            arrowprops={"arrowstyle": "->", "color": "#388E3C"},
        )
        figure.tight_layout()

        pdf_path = output_dir / "sst2_pareto_curve.pdf"
        png_path = output_dir / "sst2_pareto_curve.png"
        figure.savefig(pdf_path, bbox_inches="tight")
        figure.savefig(png_path, bbox_inches="tight")
    finally:
        plt.close(figure)
    return pdf_path, png_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"figure directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    pdf_path, png_path = plot_sst2_pareto(args.output_dir.resolve())
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
