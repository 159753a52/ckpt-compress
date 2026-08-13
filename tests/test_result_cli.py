import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.scripts import aggregate_results
from experiments.scripts.finetune import train_nlp


def _write_result(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "results": [{"method": "uniform", "target_ratio": 0.3, "perplexity": 12.0}],
                "config": {"model": "gpt2", "dataset": "wiki", "num_recoveries": 2},
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_cli_is_strict_by_default_and_lenient_only_when_requested(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_result(tmp_path / "valid_table1_run.json")
    (tmp_path / "broken_table1_run.json").write_text("{", encoding="utf-8")

    assert aggregate_results.main(["--results-dir", str(tmp_path)]) == 1
    assert "broken_table1_run.json" in capsys.readouterr().err

    assert aggregate_results.main(["--results-dir", str(tmp_path), "--lenient"]) == 0
    output = capsys.readouterr()
    assert "Found 1 valid result files" in output.out
    assert "uniform" in output.out


def test_read_table1_import_has_no_output_or_file_access(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "read_table1.py"
    process = subprocess.run(
        [sys.executable, "-c", f"import runpy; runpy.run_path({str(script)!r})"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert process.returncode == 0
    assert process.stdout == ""
    assert process.stderr == ""


def test_train_nlp_module_import_is_side_effect_free() -> None:
    assert importlib.reload(train_nlp) is train_nlp


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--epochs", "0"),
        ("--batch_size", "-1"),
        ("--lr", "0"),
        ("--lr", "nan"),
        ("--weight_decay", "inf"),
        ("--num_workers", "-1"),
    ),
)
def test_train_nlp_parser_rejects_invalid_numeric_boundaries(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        train_nlp._build_parser().parse_args([option, value])
