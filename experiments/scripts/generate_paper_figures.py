"""Generate paper figures exclusively from completed experiment JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT.parent.parent / "paper/figs"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "figure.dpi": 300,
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_completed_suite(path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    """Load a complete suite and verify every registered job digest."""
    path = path.resolve()
    payload = _load_json(path)
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError("Suite manifest must be a completed JSON object")
    completed = payload.get("completed_jobs")
    if not isinstance(completed, dict) or not completed:
        raise ValueError("Suite manifest contains no completed jobs")
    paths = {}
    for job_id, entry in completed.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"Malformed completed job entry: {job_id}")
        job_path = Path(str(entry.get("path")))
        if not job_path.is_absolute():
            job_path = path.parent / job_path
        if not job_path.is_file():
            raise FileNotFoundError(f"Missing paper job JSON: {job_path}")
        expected = entry.get("sha256")
        actual = sha256_file(job_path)
        if actual != expected:
            raise ValueError(f"Paper job digest mismatch: {job_id}")
        job_payload = _load_json(job_path)
        if not isinstance(job_payload, dict) or job_payload.get("status") != "complete":
            raise ValueError(f"Paper job is not complete: {job_id}")
        paths[str(job_id)] = job_path.resolve()
    return payload, paths


def load_fit_records(path: Path) -> list[dict[str, object]]:
    """Load real empirical-CDF grids emitted by distribution validation."""
    payload = _load_json(path)
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError("Fit source must be a completed JSON object")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) < 4:
        raise ValueError("Fit source must contain at least four layer records")
    required_curves = {"x", "empirical", "weibull", "gamma", "lognormal"}
    validated = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("cdf"), dict):
            continue
        cdf = record["cdf"]
        missing = required_curves - set(cdf)
        if missing:
            continue
        lengths = {len(cdf[key]) for key in required_curves}
        if len(lengths) != 1 or 0 in lengths:
            raise ValueError("CDF curves must be non-empty and aligned")
        for key in required_curves:
            if not all(math.isfinite(float(value)) for value in cdf[key]):
                raise ValueError(f"CDF curve contains non-finite values: {key}")
        validated.append(record)
    if len(validated) < 4:
        raise ValueError("Fit source contains fewer than four complete CDF records")
    return validated


def _write_provenance(output: Path, payload: Mapping[str, object]) -> None:
    provenance_path = output.with_suffix(".provenance.json")
    with provenance_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def generate_distribution_fit_figure(
    fit_source: Path,
    output_dir: Path,
) -> Path:
    records = load_fit_records(fit_source)
    selected = sorted(
        records,
        key=lambda record: (-int(record.get("n_params", 0)), str(record.get("layer"))),
    )[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    styles = {
        "empirical": ("#2667a8", "-", 2.0, "Empirical"),
        "weibull": ("#c53b32", "-", 1.8, "Weibull"),
        "gamma": ("#d28700", "--", 1.6, "Gamma"),
        "lognormal": ("#2a7f4f", ":", 1.6, "Log-Normal"),
    }
    for axis, record in zip(axes.flat, selected):
        cdf = record["cdf"]
        x = np.asarray(cdf["x"], dtype=float)
        for key, (color, linestyle, width, label) in styles.items():
            ks = record.get(f"{key}_ks")
            curve_label = label if ks is None else f"{label} ($D$={float(ks):.3f})"
            axis.plot(
                x,
                np.asarray(cdf[key], dtype=float),
                color=color,
                linestyle=linestyle,
                linewidth=width,
                label=curve_label,
            )
        axis.set_title(str(record.get("layer", "unnamed layer")))
        axis.set_xlabel("Damage score")
        axis.set_ylabel("CDF")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True, alpha=0.2)
        axis.legend(loc="lower right")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "distribution_fit_quality.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    _write_provenance(
        output,
        {
            "kind": "distribution_fit_quality",
            "source": str(fit_source.resolve()),
            "source_sha256": sha256_file(fit_source),
            "layers": [record.get("layer") for record in selected],
        },
    )
    return output


def _select_allocation_record(
    job_paths: Mapping[str, Path],
    requested_job: str | None,
) -> tuple[str, Path, dict[str, object], dict[str, object]]:
    candidates = []
    for job_id, path in job_paths.items():
        if requested_job and job_id != requested_job:
            continue
        payload = _load_json(path)
        config = payload.get("config", {})
        for record in payload.get("results", []):
            if record.get("method") != "dacp" or not record.get("cycles"):
                continue
            rates = record["cycles"][0].get("compression", {}).get("mask", {}).get(
                "layer_rates"
            )
            if rates:
                preference = (
                    config.get("name") != "bert_large_mnli",
                    abs(float(config.get("prune_ratio", 0.0)) - 0.3),
                    int(record.get("seed", 0)),
                )
                candidates.append((preference, job_id, path, payload, record))
    if not candidates:
        raise ValueError("No completed DACP cycle with layer_rates was found")
    _, job_id, path, payload, record = min(candidates, key=lambda item: item[0])
    return job_id, path, payload, record


def generate_allocation_figure(
    suite_path: Path,
    output_dir: Path,
    requested_job: str | None = None,
) -> Path:
    _, job_paths = load_completed_suite(suite_path)
    job_id, job_path, payload, record = _select_allocation_record(
        job_paths, requested_job
    )
    config = payload["config"]
    ratio = float(config["prune_ratio"])
    rates = np.asarray(
        record["cycles"][0]["compression"]["mask"]["layer_rates"], dtype=float
    )
    if rates.ndim != 1 or not rates.size or not np.isfinite(rates).all():
        raise ValueError("DACP layer rates must be a finite one-dimensional array")

    x = np.arange(rates.size)
    width = 0.38
    fig, axis = plt.subplots(figsize=(max(8.0, rates.size * 0.42), 4.2))
    axis.bar(
        x - width / 2,
        np.full(rates.size, ratio),
        width,
        label="Uniform",
        color="#79a8d8",
        edgecolor="#275d91",
        linewidth=0.5,
    )
    axis.bar(
        x + width / 2,
        rates,
        width,
        label="DACP",
        color="#df6b5f",
        edgecolor="#8f2d27",
        linewidth=0.5,
    )
    axis.set_xlabel("Transformer block index")
    axis.set_ylabel("Block pruning rate")
    axis.set_xticks(x)
    axis.set_xticklabels([str(index) for index in x], fontsize=7)
    axis.set_ylim(0, max(float(rates.max()), ratio) * 1.18)
    axis.grid(True, axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "blockwise_sparsity_allocation.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    _write_provenance(
        output,
        {
            "kind": "blockwise_sparsity_allocation",
            "suite": str(suite_path.resolve()),
            "suite_sha256": sha256_file(suite_path),
            "job": job_id,
            "job_source": str(job_path),
            "job_source_sha256": sha256_file(job_path),
            "method": record["method"],
            "seed": record["seed"],
            "cycle": 1,
        },
    )
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--fit-results", type=Path)
    parser.add_argument("--allocation-job")
    parser.add_argument(
        "--figure",
        choices=("all", "distribution", "allocation"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.figure in {"all", "distribution"}:
        if args.fit_results is None:
            raise ValueError("--fit-results is required for the distribution figure")
        print(generate_distribution_fit_figure(args.fit_results, args.output_dir))
    if args.figure in {"all", "allocation"}:
        if args.suite is None:
            raise ValueError("--suite is required for the allocation figure")
        print(
            generate_allocation_figure(
                args.suite, args.output_dir, requested_job=args.allocation_job
            )
        )


if __name__ == "__main__":
    main()
