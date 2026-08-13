from pathlib import Path

import pytest

import experiments.scripts.run_checkpoint_size_demo as checkpoint_size
from experiments.lib.residual_runtime import write_json


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--prune_ratios", "0"),
        ("--prune_ratios", "0.5,nan"),
        ("--methods", "unknown"),
        ("--hvp_batches", "0"),
        ("--eval_batches", "-1"),
        ("--batch_size", "0"),
        ("--seq_length", "0"),
        ("--alpha", "inf"),
    ],
)
def test_checkpoint_size_cli_rejects_invalid_boundaries(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        checkpoint_size._build_parser().parse_args(
            ["--model", "gpt2-small", "--dataset", "wikitext2", option, value]
        )


@pytest.mark.parametrize(
    ("task_type", "metrics", "expected"),
    [
        ("lm", {"perplexity": 2.0}, ("perplexity", 2.0, False)),
        ("cls", {"accuracy": 0.8}, ("accuracy", 0.8, True)),
        ("cv", {"accuracy": 0.7}, ("accuracy", 0.7, True)),
        ("reg", {"pearson": -0.2}, ("pearson", -0.2, True)),
    ],
)
def test_checkpoint_size_uses_task_metric_contract(task_type, metrics, expected) -> None:
    assert checkpoint_size._quality_metric(metrics, task_type) == expected


def test_checkpoint_size_metric_contract_fails_closed() -> None:
    with pytest.raises(KeyError, match="accuracy"):
        checkpoint_size._quality_metric({"loss": 1.0}, "cls")
    with pytest.raises(ValueError, match="finite"):
        checkpoint_size._quality_metric({"perplexity": float("nan")}, "lm")
    with pytest.raises(ValueError, match="Unsupported task type"):
        checkpoint_size._quality_metric({}, "unknown")


def test_strict_atomic_writer_does_not_replace_previous_result(tmp_path: Path) -> None:
    output = tmp_path / "diagnostic.json"
    output.write_text("previous\n", encoding="utf-8")

    with pytest.raises(ValueError):
        write_json(output, {"metric": float("nan")})

    assert output.read_text(encoding="utf-8") == "previous\n"
