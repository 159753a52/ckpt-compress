"""Generate paper figures exclusively from completed experiment JSON."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence, TypedDict

import matplotlib

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.lib.paper_results import SuiteResultStore  # noqa: E402
from experiments.lib.residual_runtime import sha256_file, write_json  # noqa: E402

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


class FitRecord(TypedDict, total=False):
    layer: str
    n_params: int
    cdf: dict[str, list[float]]
    empirical_ks: float
    weibull_ks: float
    gamma_ks: float
    lognormal_ks: float


class AllocationCandidate(TypedDict):
    job_id: str
    path: Path
    payload: dict[str, object]
    record: dict[str, object]
    ratio: float
    rates: list[float]


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_completed_suite(path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    """Load a complete suite and validate every schema-v5 job semantically."""
    path = path.resolve()
    payload = _load_json(path)
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError("Suite manifest must be a completed JSON object")
    SuiteResultStore.open(path, payload, resume=True)
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
        paths[str(job_id)] = job_path.resolve()
    return payload, paths


def load_fit_records(path: Path) -> list[FitRecord]:
    """Load real empirical-CDF grids emitted by distribution validation."""
    payload = _load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("status") != "complete"
    ):
        raise ValueError("Fit source must be a completed JSON object")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) < 4:
        raise ValueError("Fit source must contain at least four layer records")
    required_curves = {"x", "empirical", "weibull", "gamma", "lognormal"}
    validated: list[FitRecord] = []
    seen_layers: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("cdf"), dict):
            raise ValueError("Fit source contains a malformed layer record")
        cdf = record["cdf"]
        missing = required_curves - set(cdf)
        if missing:
            raise ValueError(f"Fit record is missing CDF competitors: {sorted(missing)}")
        if (
            not isinstance(record.get("layer"), str)
            or not record["layer"]
            or isinstance(record.get("n_params"), bool)
            or not isinstance(record.get("n_params"), int)
            or record["n_params"] < 1
        ):
            raise ValueError("Fit records must identify a non-empty layer with parameters")
        if record["layer"] in seen_layers:
            raise ValueError(f"Fit source contains duplicate layer: {record['layer']}")
        seen_layers.add(record["layer"])
        if not all(isinstance(cdf[key], list) for key in required_curves):
            raise ValueError("CDF curves must be JSON arrays")
        lengths = {len(cdf[key]) for key in required_curves}
        if len(lengths) != 1 or 0 in lengths:
            raise ValueError("CDF curves must be non-empty and aligned")
        numeric_curves = {}
        for key in required_curves:
            try:
                values = [float(value) for value in cdf[key]]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"CDF curve contains non-numeric values: {key}") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"CDF curve contains non-finite values: {key}")
            numeric_curves[key] = values
        x_values = numeric_curves["x"]
        if (
            len(x_values) < 2
            or any(value < 0.0 for value in x_values)
            or any(right <= left for left, right in zip(x_values, x_values[1:]))
        ):
            raise ValueError("CDF x values must be non-negative and strictly increasing")
        for key in required_curves - {"x"}:
            values = numeric_curves[key]
            if any(not 0.0 <= value <= 1.0 for value in values) or any(
                right < left for left, right in zip(values, values[1:])
            ):
                raise ValueError(f"CDF values must be in [0, 1] and non-decreasing: {key}")
        normalized: FitRecord = {
            "layer": record["layer"],
            "n_params": record["n_params"],
            "cdf": numeric_curves,
        }
        for key in ("weibull_ks", "gamma_ks", "lognormal_ks"):
            raw = record.get(key)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"Fit statistic {key} must be numeric")
            numeric = float(raw)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError(f"Fit statistic {key} must be finite and in [0, 1]")
            curve_name = key.removesuffix("_ks")
            grid_difference = max(
                abs(empirical - fitted)
                for empirical, fitted in zip(
                    numeric_curves["empirical"],
                    numeric_curves[curve_name],
                )
            )
            if numeric + 1e-12 < grid_difference:
                raise ValueError(f"Fit statistic {key} is smaller than its CDF grid difference")
            if key == "weibull_ks":
                normalized["weibull_ks"] = numeric
            elif key == "gamma_ks":
                normalized["gamma_ks"] = numeric
            else:
                normalized["lognormal_ks"] = numeric
        validated.append(normalized)
    if len(validated) < 4:
        raise ValueError("Fit source contains fewer than four complete CDF records")
    return validated


def _write_provenance(output: Path, payload: Mapping[str, object]) -> None:
    provenance_path = output.with_suffix(".provenance.json")
    write_json(provenance_path, payload)


def generate_distribution_fit_figure(
    fit_source: Path,
    output_dir: Path,
) -> Path:
    records = load_fit_records(fit_source)
    selected = sorted(
        records,
        key=lambda record: (-record["n_params"], record["layer"]),
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
            if ks is not None and (isinstance(ks, bool) or not isinstance(ks, (int, float))):
                raise TypeError(f"Validated fit statistic has an invalid type: {key}_ks")
            curve_label = label if ks is None else f"{label} ($D$={ks:.3f})"
            axis.plot(
                x,
                np.asarray(cdf[key], dtype=float),
                color=color,
                linestyle=linestyle,
                linewidth=width,
                label=curve_label,
            )
        axis.set_title(record["layer"])
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
            "layers": [record["layer"] for record in selected],
        },
    )
    return output


def _select_allocation_record(
    job_paths: Mapping[str, Path],
    requested_job: str | None,
) -> AllocationCandidate:
    candidates: list[tuple[tuple[bool, float, int], AllocationCandidate]] = []
    for job_id, path in job_paths.items():
        if requested_job and job_id != requested_job:
            continue
        payload = _load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Paper job must be a JSON object: {path}")
        config = payload.get("config")
        records = payload.get("results")
        if not isinstance(config, dict) or not isinstance(records, list):
            raise ValueError(f"Paper job has invalid config/results: {path}")
        ratio_raw = config.get("prune_ratio")
        if isinstance(ratio_raw, bool) or not isinstance(ratio_raw, (int, float)):
            raise ValueError(f"Paper job has invalid prune ratio: {path}")
        ratio = float(ratio_raw)
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Paper job contains a malformed result: {path}")
            if record.get("method") != "dacp" or not record.get("cycles"):
                continue
            cycles = record["cycles"]
            if not isinstance(cycles, list) or not cycles or not isinstance(cycles[0], dict):
                continue
            compression = cycles[0].get("compression")
            mask = compression.get("mask") if isinstance(compression, dict) else None
            raw_rates = mask.get("layer_rates") if isinstance(mask, dict) else None
            seed = record.get("seed")
            if (
                not isinstance(raw_rates, list)
                or isinstance(seed, bool)
                or not isinstance(seed, int)
            ):
                continue
            rates = [float(value) for value in raw_rates]
            preference = (
                config.get("name") != "bert_large_mnli",
                abs(ratio - 0.3),
                seed,
            )
            candidates.append(
                (
                    preference,
                    {
                        "job_id": job_id,
                        "path": path,
                        "payload": payload,
                        "record": record,
                        "ratio": ratio,
                        "rates": rates,
                    },
                )
            )
    if not candidates:
        raise ValueError("No completed DACP cycle with layer_rates was found")
    return min(candidates, key=lambda item: item[0])[1]


def generate_allocation_figure(
    suite_path: Path,
    output_dir: Path,
    requested_job: str | None = None,
) -> Path:
    _, job_paths = load_completed_suite(suite_path)
    selected = _select_allocation_record(job_paths, requested_job)
    job_id = selected["job_id"]
    job_path = selected["path"]
    record = selected["record"]
    ratio = selected["ratio"]
    rates = np.asarray(selected["rates"], dtype=float)
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
