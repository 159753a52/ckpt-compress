"""Result management utilities for experiments.

Standardized storage, loading, and comparison of experiment results.

Usage:
    from experiments.lib.results import ResultManager

    manager = ResultManager(output_dir='experiments/results')
    manager.save(results, experiment_name='table1', run_name='run_20260313')
    results = manager.load(experiment_name='table1', run_name='run_20260313')
    manager.compare(['run1', 'run2', 'run3'])
"""

import json
import math
import numbers
import shutil
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from .result_schema import load_result_bundle, result_metrics


def _validate_path_component(value: str, *, field: str) -> str:
    """Validate a caller-provided directory or filename component."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if Path(value).is_absolute() or value in {".", ".."}:
        raise ValueError(f"{field} must be a relative path component")
    if "/" in value or "\\" in value:
        raise ValueError(f"{field} must not contain path separators")
    return value


def _validate_json_numbers(value: object, *, path: str = "result") -> None:
    """Reject non-finite numbers before any evidence file is opened."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} must not contain non-finite numbers")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_json_numbers(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_numbers(child, path=f"{path}[{index}]")


def _json_text(value: object) -> str:
    """Serialize validated evidence as standards-compliant JSON."""
    _validate_json_numbers(value)

    def json_default(item: object) -> object:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, numbers.Integral) and not isinstance(item, bool):
            return int(item)
        if isinstance(item, numbers.Real) and not isinstance(item, bool):
            return float(item)
        raise TypeError(f"unsupported result value type: {type(item).__name__}")

    return json.dumps(value, indent=2, default=json_default, allow_nan=False)


@lru_cache(maxsize=1)
def _get_pd():
    """Import pandas only when table output is requested."""
    import pandas

    return pandas


class ResultManager:
    """Experiment result manager."""

    def __init__(self, output_dir: str = "experiments/results"):
        """Initialize the result manager.

        Args:
            output_dir: output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        results: List[Dict[str, Any]],
        experiment_name: str,
        run_name: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Save experiment results.

        Args:
            results: list of result dicts
            experiment_name: experiment identifier (e.g. 'table1')
            run_name: run identifier (usually includes a timestamp)
            config: optional experiment config
        """
        experiment_name = _validate_path_component(
            experiment_name, field="experiment_name"
        )
        run_name = _validate_path_component(run_name, field="run_name")
        experiment_path = self.output_dir / experiment_name
        output_path = experiment_path / run_name

        # Serialize before opening any destination so invalid evidence cannot
        # truncate a previous successful run.
        results_text = _json_text(results)
        metadata = {
            "experiment_name": experiment_name,
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "num_results": len(results),
        }
        metadata_text = _json_text(metadata)
        config_text = None
        if config:
            _validate_json_numbers(config, path="config")
            config_text = yaml.safe_dump(
                config,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=True,
            )
        if output_path.exists():
            raise FileExistsError(f"Result run already exists: {output_path}")
        experiment_path.mkdir(parents=True, exist_ok=True)
        staging_path = Path(
            tempfile.mkdtemp(prefix=f".{run_name}.tmp-", dir=experiment_path)
        )
        try:
            (staging_path / "results.json").write_text(
                results_text, encoding="utf-8"
            )
            self._save_csv(results, staging_path / "results.csv")
            if config_text is not None:
                (staging_path / "config.yaml").write_text(
                    config_text, encoding="utf-8"
                )
            (staging_path / "metadata.json").write_text(
                metadata_text, encoding="utf-8"
            )
            staging_path.rename(output_path)
        except BaseException:
            shutil.rmtree(staging_path, ignore_errors=True)
            raise

        print(f"Results saved to: {output_path}")

    def load(
        self,
        experiment_name: str,
        run_name: str,
    ) -> List[Dict[str, Any]]:
        """Load experiment results.

        Args:
            experiment_name: experiment identifier
            run_name: run identifier

        Returns:
            list of result dicts
        """
        experiment_name = _validate_path_component(
            experiment_name, field="experiment_name"
        )
        run_name = _validate_path_component(run_name, field="run_name")
        results_file = self.output_dir / experiment_name / run_name / "results.json"

        if not results_file.exists():
            raise FileNotFoundError(f"Results not found: {results_file}")

        return [dict(record) for record in load_result_bundle(results_file).records]

    def list_runs(self, experiment_name: str) -> List[str]:
        """List all runs for an experiment."""
        experiment_name = _validate_path_component(
            experiment_name, field="experiment_name"
        )
        experiment_path = self.output_dir / experiment_name
        if not experiment_path.exists():
            return []

        return [
            d.name
            for d in experiment_path.iterdir()
            if d.is_dir() and (d / "results.json").exists()
        ]

    def compare(
        self,
        run_names: List[str],
        experiment_name: Optional[str] = None,
    ):
        """Compare multiple runs.

        Args:
            run_names: list of run identifiers
            experiment_name: experiment id (if None, infer from run_names)
        """
        if not run_names:
            print("No runs to compare")
            return

        all_results = []
        for run_name in run_names:
            try:
                results = self.load(experiment_name or self._infer_experiment(run_name), run_name)
                for r in results:
                    r["run_name"] = run_name
                all_results.extend(results)
            except FileNotFoundError as e:
                print(f"Warning: {e}")

        if not all_results:
            print("No results loaded")
            return

        self._print_comparison_table(all_results)

    def print_table(self, results: List[Dict[str, Any]]):
        """Print a results table."""
        df = self._results_to_dataframe(results)

        if df.empty:
            print("No data to display")
            return

        _get_pd().set_option("display.max_columns", None)
        _get_pd().set_option("display.width", None)
        _get_pd().set_option("display.max_colwidth", None)

        print("\n" + "=" * 100)
        print("Experiment Results Summary")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)

    def _save_csv(self, results: List[Dict[str, Any]], csv_file: Path):
        """Save results as CSV."""
        df = self._results_to_dataframe(results)
        df.to_csv(csv_file, index=False)

    def _results_to_dataframe(self, results: List[Dict[str, Any]]):
        """Convert results to a pandas DataFrame."""
        if not results:
            return _get_pd().DataFrame()

        rows = []
        for r in results:
            row = {
                "method": r.get("method"),
                "importance": r.get("importance"),
                "allocation": r.get("allocation"),
                "prune_ratio": r.get("prune_ratio"),
                "actual_ratio": r.get("actual_ratio"),
            }

            for key, value in result_metrics(r).items():
                row[f"metric_{key}"] = value

            baseline = r.get("baseline", {})
            if isinstance(baseline, dict):
                for key, value in baseline.items():
                    row[f"baseline_{key}"] = value

            rows.append(row)

        return _get_pd().DataFrame(rows)

    def _print_comparison_table(self, all_results: List[Dict[str, Any]]):
        """Print a comparison table across runs."""
        df = self._results_to_dataframe(all_results)

        if df.empty:
            print("No data to compare")
            return

        pivot_cols = ["method", "prune_ratio"]
        value_cols = [c for c in df.columns if c.startswith("metric_")]

        if not value_cols:
            print("No metric columns found")
            return

        print("\n" + "=" * 120)
        print("Experiment Comparison")
        print("=" * 120)

        for metric_col in value_cols:
            print(f"\n{metric_col}:")
            pivot_df = df.pivot_table(
                index="method", columns="prune_ratio", values=metric_col, aggfunc="first"
            )
            print(pivot_df.to_string())

        print("=" * 120)

    def _infer_experiment(self, run_name: str) -> str:
        """Infer experiment name from run name (simplified)."""
        return "experiment"


# ============================================================
# Standalone function interface (for scripts to import directly)
# ============================================================


def save_results(
    results: list,
    output_dir: str,
    experiment_name: str,
    config: Optional[dict] = None,
):
    """Save experiment results to JSON + CSV.

    Args:
        results: list of result dicts
        output_dir: output directory
        experiment_name: experiment name (used for filename)
        config: experiment config
    """
    experiment_name = _validate_path_component(
        experiment_name, field="experiment_name"
    )
    output_path = Path(output_dir)

    payload_text = _json_text({"results": results, "config": config})
    config_text = _json_text(config) if config else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{experiment_name}"
    base_name = stem
    suffix = 1
    while (output_path / base_name).exists():
        base_name = f"{stem}_{suffix}"
        suffix += 1
    output_path.mkdir(parents=True, exist_ok=True)
    final_path = output_path / base_name
    staging_path = Path(
        tempfile.mkdtemp(prefix=f".{base_name}.tmp-", dir=output_path)
    )
    try:
        (staging_path / "results.json").write_text(payload_text, encoding="utf-8")
        _get_pd().DataFrame(results).to_csv(staging_path / "results.csv", index=False)
        if config_text is not None:
            (staging_path / "config.json").write_text(config_text, encoding="utf-8")
        staging_path.rename(final_path)
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise

    print(f"Results saved: {final_path}")


def print_results_table(results: list, columns: Optional[list] = None):
    """Print a results table.

    Args:
        results: list of result dicts
        columns: column names to show (None = all)
    """
    if not results:
        print("No results")
        return

    df = _get_pd().DataFrame(results)
    if columns:
        cols = [c for c in columns if c in df.columns]
        if cols:
            df = df[cols]

    _get_pd().set_option("display.max_columns", None)
    _get_pd().set_option("display.width", None)
    _get_pd().set_option("display.max_colwidth", 40)

    print("\n" + "=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
