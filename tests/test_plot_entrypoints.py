import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.scripts.plot_sst2_pareto import plot_sst2_pareto
from experiments.scripts.plot_weibull_fit import (
    positive_finite_samples,
    select_representative_layers,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "experiments/scripts/plot_sst2_pareto.py",
        "experiments/scripts/plot_weibull_fit.py",
        "experiments/scripts/run_pruning_heatmap.py",
    ],
)
def test_plot_cli_help_has_no_execution_side_effects(script: str, tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert list(tmp_path.iterdir()) == []


def test_sst2_plot_creates_explicit_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "figures"

    pdf_path, png_path = plot_sst2_pareto(output_dir)

    assert pdf_path.is_file()
    assert png_path.is_file()
    assert pdf_path.parent == output_dir


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        (torch.tensor([]), "empty"),
        (torch.tensor([0.0, -1.0]), "at least two positive"),
        (torch.tensor([1.0, float("nan")]), "finite"),
        (torch.tensor([1.0, float("inf")]), "finite"),
        (torch.tensor([1.0, 1.0]), "non-zero variance"),
    ],
)
def test_weibull_samples_fail_closed(scores: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        positive_finite_samples(scores, "layer")


def test_weibull_selection_and_sampling_are_deterministic() -> None:
    with pytest.raises(ValueError, match="at least one layer"):
        select_representative_layers({})

    values = torch.arange(1.0, 101.0)
    first = positive_finite_samples(values, "layer", max_samples=10, seed=7)
    second = positive_finite_samples(values, "layer", max_samples=10, seed=7)
    assert np.array_equal(first, second)
