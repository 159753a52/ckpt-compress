"""Result management utilities for experiments.

Standardized storage, loading, and comparison of experiment results.

Usage:
    from experiments.lib.results import ResultManager

    manager = ResultManager(output_dir='experiments/results')
    manager.save(results, experiment_name='table1', run_name='run_20260313')
    results = manager.load(experiment_name='table1', run_name='run_20260313')
    manager.compare(['run1', 'run2', 'run3'])
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import yaml

pd = None  # lazy import of pandas
def _get_pd():
    global pd
    if pd is None:
        import pandas as _pd
        pd = _pd
    return pd


class ResultManager:
    """Experiment result manager."""

    def __init__(self, output_dir: str = 'experiments/results'):
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
        output_path = self.output_dir / experiment_name / run_name
        output_path.mkdir(parents=True, exist_ok=True)

        # full results as JSON
        results_file = output_path / 'results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        # CSV
        csv_file = output_path / 'results.csv'
        self._save_csv(results, csv_file)

        # config
        if config:
            config_file = output_path / 'config.yaml'
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        # metadata
        metadata = {
            'experiment_name': experiment_name,
            'run_name': run_name,
            'timestamp': datetime.now().isoformat(),
            'num_results': len(results),
        }
        metadata_file = output_path / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

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
        results_file = self.output_dir / experiment_name / run_name / 'results.json'

        if not results_file.exists():
            raise FileNotFoundError(f"Results not found: {results_file}")

        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_runs(self, experiment_name: str) -> List[str]:
        """List all runs for an experiment."""
        experiment_path = self.output_dir / experiment_name
        if not experiment_path.exists():
            return []

        return [
            d.name for d in experiment_path.iterdir()
            if d.is_dir() and (d / 'results.json').exists()
        ]

    def compare(
        self,
        run_names: List[str],
        experiment_name: str = None,
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
                    r['run_name'] = run_name
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

        _get_pd().set_option('display.max_columns', None)
        _get_pd().set_option('display.width', None)
        _get_pd().set_option('display.max_colwidth', None)

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
                'method': r.get('method', 'unknown'),
                'importance': r.get('importance', 'unknown'),
                'allocation': r.get('allocation', 'unknown'),
                'prune_ratio': r.get('prune_ratio', 0),
                'actual_ratio': r.get('actual_ratio', 0),
            }

            metrics = r.get('metrics', {})
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    row[f'metric_{key}'] = value

            baseline = r.get('baseline', {})
            if isinstance(baseline, dict):
                for key, value in baseline.items():
                    row[f'baseline_{key}'] = value

            rows.append(row)

        return _get_pd().DataFrame(rows)

    def _print_comparison_table(self, all_results: List[Dict[str, Any]]):
        """Print a comparison table across runs."""
        df = self._results_to_dataframe(all_results)

        if df.empty:
            print("No data to compare")
            return

        pivot_cols = ['method', 'prune_ratio']
        value_cols = [c for c in df.columns if c.startswith('metric_')]

        if not value_cols:
            print("No metric columns found")
            return

        print("\n" + "=" * 120)
        print("Experiment Comparison")
        print("=" * 120)

        for metric_col in value_cols:
            print(f"\n{metric_col}:")
            pivot_df = df.pivot_table(
                index='method',
                columns='prune_ratio',
                values=metric_col,
                aggfunc='first'
            )
            print(pivot_df.to_string())

        print("=" * 120)

    def _infer_experiment(self, run_name: str) -> str:
        """Infer experiment name from run name (simplified)."""
        return 'experiment'


# ============================================================
# Standalone function interface (for scripts to import directly)
# ============================================================

def save_results(
    results: list,
    output_dir: str,
    experiment_name: str,
    config: dict = None,
):
    """Save experiment results to JSON + CSV.

    Args:
        results: list of result dicts
        output_dir: output directory
        experiment_name: experiment name (used for filename)
        config: experiment config
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"{timestamp}_{experiment_name}"

    # JSON
    json_file = output_path / f'{base_name}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'config': config}, f, indent=2, default=str)

    # CSV
    csv_file = output_path / f'{base_name}.csv'
    df = _get_pd().DataFrame(results)
    df.to_csv(csv_file, index=False)

    # config
    if config:
        cfg_file = output_path / f'{base_name}_config.json'
        with open(cfg_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, default=str)

    print(f"Results saved: {output_path / base_name}.*")


def print_results_table(results: list, columns: list = None):
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

    _get_pd().set_option('display.max_columns', None)
    _get_pd().set_option('display.width', None)
    _get_pd().set_option('display.max_colwidth', 40)

    print("\n" + "=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
