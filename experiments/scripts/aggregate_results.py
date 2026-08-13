"""Aggregate validated paper-result JSON files into comparable tables."""

import argparse
import math
import os
import sys
from collections import defaultdict
from numbers import Real
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple, TypedDict, Union, cast

import yaml

ROOT = Path(__file__).parent.parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from experiments.lib.result_schema import (
    ResultBundle,
    load_result_bundle,
    normalize_result_payload,
    primary_metric,
)

PathLike = Union[str, os.PathLike[str]]


class ResultScanError(ValueError):
    """Raised when a candidate paper result cannot be trusted."""


class MetricEntry(TypedDict):
    """One validated final metric used by the summary renderer."""

    model: str
    dataset: str
    prune_ratio: float
    K: Union[int, str]
    lr: Any
    method: str
    metric: str
    value: float


def _nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResultScanError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ResultScanError(f"{name} must be a finite number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ResultScanError(f"{name} must be a finite number, got {value!r}")
    return numeric


def _prune_ratio(record: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    value = record.get(
        "prune_ratio",
        record.get("target_ratio", config.get("prune_ratio", config.get("target_ratio"))),
    )
    ratio = _finite_number("prune_ratio", value)
    if not 0.0 <= ratio <= 1.0:
        raise ResultScanError(f"prune_ratio must be within [0, 1], got {ratio}")
    return ratio


def _recoveries(config: Mapping[str, Any]) -> Union[int, str]:
    value = config.get("num_recoveries")
    if value is None:
        return "?"
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ResultScanError(f"num_recoveries must be a positive integer, got {value!r}")
    return cast(int, value)


def _metric(record: Mapping[str, Any]) -> Tuple[str, float]:
    metric_name, value = primary_metric(record)
    if metric_name is None:
        method = record.get("method", "?")
        raise ResultScanError(f"result for method {method!r} has no supported metric")
    numeric = _finite_number(f"metric {metric_name!r}", value)
    if metric_name in ("accuracy", "top1_accuracy") and not 0.0 <= numeric <= 1.0:
        raise ResultScanError(f"{metric_name} must be within [0, 1], got {numeric}")
    if metric_name == "pearson" and not -1.0 <= numeric <= 1.0:
        raise ResultScanError(f"pearson must be within [-1, 1], got {numeric}")
    if metric_name == "perplexity" and numeric <= 0.0:
        raise ResultScanError(f"perplexity must be positive, got {numeric}")
    if metric_name in ("loss", "final_loss", "val_loss") and numeric < 0.0:
        raise ResultScanError(f"{metric_name} must be non-negative, got {numeric}")
    return metric_name, numeric


def _result_candidates(results_dir: PathLike) -> List[Path]:
    root = Path(results_dir)
    if not root.is_dir():
        raise ResultScanError(f"results directory not found: {root}")

    def is_result_file(path: Path) -> bool:
        if path.name == "results.json":
            return True
        if path.with_name(f"{path.stem}_config.json").is_file():
            return True
        return "_table1_" in path.stem or "_ft_" in path.stem

    return sorted(
        path
        for path in root.rglob("*.json")
        if not path.name.endswith("_config.json")
        and path.name != "metadata.json"
        and is_result_file(path)
    )


def load_all_results(results_dir: PathLike, *, strict: bool = True) -> List[ResultBundle]:
    """Load result files, rejecting bad candidates unless lenient mode is explicit."""
    results: List[ResultBundle] = []
    for path in _result_candidates(results_dir):
        try:
            bundle = load_result_bundle(path)
            if not any(isinstance(record.get("method"), str) for record in bundle.records):
                raise ResultScanError("result contains no method records")
        except (OSError, TypeError, ValueError, UnicodeError, yaml.YAMLError) as exc:
            if strict:
                raise ResultScanError(f"invalid result file {path}: {exc}") from exc
            continue
        results.append(bundle)
    return results


def scan_results(results_dir: PathLike, *, strict: bool = True) -> List[dict[str, Any]]:
    """Return normalized records from recognized and validated result files."""
    records: List[dict[str, Any]] = []
    for bundle in load_all_results(results_dir, strict=strict):
        try:
            config = bundle.config
            model = _nonempty_string("model", config.get("model"))
            dataset = _nonempty_string("dataset", config.get("dataset"))
            bundle_records: List[dict[str, Any]] = []
            for record in bundle.records:
                method = _nonempty_string("method", record.get("method"))
                _metric(record)
                bundle_records.append(
                    {
                        "model": _nonempty_string("model", record.get("model", model)),
                        "dataset": _nonempty_string("dataset", record.get("dataset", dataset)),
                        "method": method,
                        "prune_ratio": _prune_ratio(record, config),
                        "source_file": str(bundle.source_path),
                        **{
                            key: value
                            for key, value in record.items()
                            if key not in {"model", "dataset", "method", "prune_ratio"}
                        },
                    }
                )
        except ResultScanError as exc:
            if strict:
                raise ResultScanError(f"invalid result file {bundle.source_path}: {exc}") from exc
            continue
        records.extend(bundle_records)
    return records


def _final_records(bundle: ResultBundle) -> List[Mapping[str, Any]]:
    configured_finals = bundle.config.get("final")
    if configured_finals is not None:
        if not isinstance(configured_finals, Mapping):
            raise ResultScanError("config final results must be an object")
        records: List[Mapping[str, Any]] = []
        for method, metrics in configured_finals.items():
            if not isinstance(metrics, Mapping):
                raise ResultScanError(f"final metrics for {method!r} must be an object")
            records.append({**metrics, "method": method})
        return records

    return [cast(Mapping[str, Any], record) for record in bundle.records]


def extract_final_metrics(result: Union[ResultBundle, Any]) -> List[MetricEntry]:
    """Extract and validate the last metric for every method in one result."""
    bundle = result if isinstance(result, ResultBundle) else normalize_result_payload(result)
    config = bundle.config
    model = _nonempty_string("model", config.get("model"))
    dataset = _nonempty_string("dataset", config.get("dataset"))
    recoveries = _recoveries(config)

    entries: List[MetricEntry] = []
    for record in _final_records(bundle):
        method = _nonempty_string("method", record.get("method"))
        metric_name, value = _metric(record)
        entries.append(
            {
                "model": model,
                "dataset": dataset,
                "prune_ratio": _prune_ratio(record, config),
                "K": recoveries,
                "lr": config.get("lr", "?"),
                "method": method,
                "metric": metric_name,
                "value": value,
            }
        )
    return entries


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "paper_results",
        help="Directory containing paper-result JSON files",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Skip malformed and unrecognized JSON files during exploratory scans",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        all_results = load_all_results(args.results_dir, strict=not args.lenient)
        entries: List[MetricEntry] = []
        valid_file_count = 0
        for bundle in all_results:
            try:
                bundle_entries = extract_final_metrics(bundle)
            except ResultScanError as exc:
                if not args.lenient:
                    raise ResultScanError(
                        f"invalid result file {bundle.source_path}: {exc}"
                    ) from exc
                continue
            entries.extend(bundle_entries)
            valid_file_count += 1

        print(f"Found {valid_file_count} valid result files\n")
        if not entries:
            raise ResultScanError("no valid results found")

        groups: dict[Tuple[str, str, str, str], List[MetricEntry]] = defaultdict(list)
        for entry in entries:
            key = (entry["model"], entry["dataset"], str(entry["K"]), entry["metric"])
            groups[key].append(entry)

        for (model, dataset, recoveries, metric_name), group in sorted(groups.items()):
            higher_better = metric_name in ("accuracy", "top1_accuracy", "pearson")
            print(f"{'=' * 60}")
            print(f"{model} / {dataset} (K={recoveries})")
            print(
                f"Metric: {metric_name} "
                f"{'(higher=better)' if higher_better else '(lower=better)'}"
            )
            print(f"{'=' * 60}")

            by_ratio: dict[float, dict[str, float]] = defaultdict(dict)
            for entry in group:
                ratio = entry["prune_ratio"]
                method = entry["method"]
                if method in by_ratio[ratio]:
                    raise ResultScanError(
                        f"duplicate result for {model}/{dataset}/{metric_name}, "
                        f"K={recoveries}, ratio={ratio}, method={method}"
                    )
                by_ratio[ratio][method] = entry["value"]

            all_methods = sorted({entry["method"] for entry in group})
            ratios = sorted(by_ratio)
            header = f"{'Method':<30}" + "".join(f" | {ratio:>8.0%}" for ratio in ratios)
            print(header)
            print("-" * len(header))

            for method in all_methods:
                row = f"{method:<30}"
                for ratio in ratios:
                    value = by_ratio[ratio].get(method)
                    if value is None:
                        row += f" | {'-':>8}"
                    elif metric_name == "perplexity":
                        row += f" | {value:>8.2f}"
                    elif metric_name in ("accuracy", "top1_accuracy", "pearson"):
                        row += f" | {value:>7.1%}"
                    else:
                        row += f" | {value:>8.4f}"
                print(row)

            print()
            best_row = f"{'BEST':<30}"
            for ratio in ratios:
                values = by_ratio[ratio]
                select = max if higher_better else min
                best_method = select(values, key=lambda method: values[method])
                best_row += f" | {best_method[:8]:>8}"
            print(best_row)
            print()
        return 0
    except ResultScanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
