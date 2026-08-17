import json
from unittest import mock

import pytest
import torch

import experiments.scripts.run_blockwise_hvp_analysis as analysis


def _args(tmp_path, *extra):
    return analysis.parse_args(
        [
            "--model",
            "gpt2-small",
            "--dataset",
            "wikitext2",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
            *extra,
        ]
    )


def test_parse_and_validate_plan_includes_default_seed_and_normalized_values(tmp_path) -> None:
    args = _args(
        tmp_path,
        "--seq_lengths",
        "8,16",
        "--alpha_sweep",
        "0.25,0.5",
        "--prune_ratios",
        "0.2,0.4",
    )

    assert args.seed == 42
    plan = analysis.validate_args(args, check_output=False)

    assert plan["seq_lengths"] == [8, 16]
    assert plan["alphas"] == [0.25, 0.5]
    assert plan["prune_ratios"] == [0.2, 0.4]
    assert "gpt2-small_seed42" in plan["output_file"]


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--seq_lengths", "0", "seq_lengths"),
        ("--alpha_sweep", "-0.1", "alpha values"),
        ("--prune_ratios", "-0.1", "prune_ratios"),
    ],
)
def test_validation_rejects_invalid_experiment_values(tmp_path, option, value, message) -> None:
    args = _args(tmp_path, option, value)
    with pytest.raises(ValueError, match=message):
        analysis.validate_args(args, check_output=False)


def test_alpha_sweep_preserves_values_above_one(tmp_path) -> None:
    args = _args(tmp_path, "--alpha_sweep", "1.1,2.5")

    plan = analysis.validate_args(args, check_output=False)

    assert plan["alphas"] == [1.1, 2.5]


def test_negative_seed_is_rejected_by_argument_parser(tmp_path) -> None:
    with pytest.raises(SystemExit):
        _args(tmp_path, "--seed", "-1")


def test_dry_run_does_not_load_model_data_or_cuda(monkeypatch, tmp_path, capsys) -> None:
    load_model = mock.Mock(side_effect=AssertionError("load_model called"))
    get_data_loaders = mock.Mock(side_effect=AssertionError("get_data_loaders called"))
    cuda_available = mock.Mock(side_effect=AssertionError("CUDA queried"))
    monkeypatch.setattr(analysis, "load_model", load_model)
    monkeypatch.setattr(analysis, "get_data_loaders", get_data_loaders)
    monkeypatch.setattr(analysis.torch.cuda, "is_available", cuda_available)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blockwise_hvp_analysis.py",
            "--model",
            "gpt2-small",
            "--dataset",
            "wikitext2",
            "--seq_lengths",
            "8,16",
            "--alpha",
            "0.25",
            "--prune_ratios",
            "0.2,0.4",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
    )

    analysis.main()

    output = capsys.readouterr().out
    assert "gpt2-small_seed42" in output
    load_model.assert_not_called()
    get_data_loaders.assert_not_called()
    cuda_available.assert_not_called()


def test_measure_hvp_reports_zero_cpu_peak_and_uses_runtime_helpers(monkeypatch) -> None:
    reset = mock.Mock()
    peak = mock.Mock(return_value=1234)
    synchronize = mock.Mock()
    monkeypatch.setattr(analysis, "reset_peak_memory", reset)
    monkeypatch.setattr(analysis, "peak_memory_bytes", peak)
    monkeypatch.setattr(analysis, "synchronize_device", synchronize)

    result, measurement = analysis._measure_hvp("cpu", lambda: {"weight": torch.ones(1)})

    assert result["weight"].item() == 1.0
    assert measurement["wall_seconds"] >= 0
    assert measurement["peak_gpu_memory_bytes"] == 0
    reset.assert_called_once_with("cpu")
    peak.assert_called_once_with("cpu")
    assert synchronize.call_count == 2


def test_measure_hvp_uses_cuda_peak_helper_when_device_is_cuda(monkeypatch) -> None:
    reset = mock.Mock()
    peak = mock.Mock(return_value=9876)
    synchronize = mock.Mock()
    monkeypatch.setattr(analysis, "reset_peak_memory", reset)
    monkeypatch.setattr(analysis, "peak_memory_bytes", peak)
    monkeypatch.setattr(analysis, "synchronize_device", synchronize)

    _, measurement = analysis._measure_hvp("cuda:0", lambda: {})

    assert measurement["peak_gpu_memory_bytes"] == 9876
    reset.assert_called_once_with("cuda:0")
    peak.assert_called_once_with("cuda:0")
    assert synchronize.call_count == 2


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_comparison_seed_order_cleanup_and_schema(monkeypatch) -> None:
    model = _TinyModel()
    batch = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "labels": torch.tensor([[1, 2]], dtype=torch.long),
    }
    events = []
    seed_mock = mock.Mock(side_effect=lambda seed: events.append(("seed", seed)))
    monkeypatch.setattr(analysis, "set_seed", seed_mock)
    monkeypatch.setattr(analysis, "get_data_loaders", lambda *args: ([batch], [], "lm"))
    monkeypatch.setattr(analysis, "cache_batches", lambda loader, count, task: [batch])
    monkeypatch.setattr(
        analysis,
        "_compute_shared_gradient",
        lambda work_model, loss_fn, batches, count: {
            "weight": torch.full_like(work_model.weight, 0.5)
        },
    )
    monkeypatch.setattr(
        analysis,
        "_compute_full_hvp",
        lambda work_model, loss_fn, batches, names, count: (
            events.append("global"),
            {"weight": torch.full_like(work_model.weight, 2.0)},
        )[1],
    )
    monkeypatch.setattr(
        analysis,
        "_compute_block_hvp",
        lambda work_model, loss_fn, batches, family, names, count: (
            events.append("block"),
            {"weight": torch.full_like(work_model.weight, 3.0)},
        )[1],
    )
    monkeypatch.setattr(
        analysis,
        "_cleanup_device",
        lambda device: events.append("cleanup"),
    )

    result = analysis.run_single_comparison(
        model=model,
        model_name="gpt2-small",
        model_family="gpt2",
        dataset_name="wikitext2",
        seq_length=8,
        batch_size=1,
        alpha=0.5,
        hvp_batches=1,
        prune_ratios=[0.2],
        device="cpu",
        seed=7,
        checkpoint=None,
        created_at="2026-08-17T00:00:00Z",
        device_name="cpu",
        dtype="torch.float32",
        source_git={"git_commit": "abc", "git_dirty": False},
    )

    assert [item for item in events if isinstance(item, tuple)] == [
        ("seed", 7),
        ("seed", 7),
        ("seed", 7),
    ]
    assert events.index("global") < events.index("block")
    assert events.index("cleanup", events.index("global")) < events.index("block")
    assert result["schema_version"] == analysis.SCHEMA_VERSION
    assert result["created_at"] == "2026-08-17T00:00:00Z"
    assert result["seed"] == 7
    assert result["batch_hash"]
    assert result["paths"]["global"]["wall_seconds"] >= 0
    assert result["paths"]["global"]["peak_gpu_memory_bytes"] == 0
    assert result["paths"]["block"]["peak_gpu_memory_bytes"] == 0
    assert result["measurement_order"] == ["global", "block"]
    assert "global_l2_error" in result
    assert "global_spearman" in result
    assert "global_pearson" in result
    assert "mask_ious" in result


def test_incremental_results_survive_later_combination_failure(monkeypatch, tmp_path) -> None:
    args = _args(tmp_path, "--seq_lengths", "8,16")
    monkeypatch.setattr(analysis, "parse_args", lambda: args)
    load_model = mock.Mock(return_value=(_TinyModel(), "gpt2"))
    monkeypatch.setattr(analysis, "load_model", load_model)
    monkeypatch.setattr(
        analysis,
        "_source_git_state",
        lambda: {"git_commit": "a" * 40, "git_dirty": False},
    )
    calls = []

    def fake_comparison(**kwargs):
        calls.append(kwargs["seq_length"])
        if len(calls) == 2:
            raise RuntimeError("second combination failed")
        return {
            "seq_length": kwargs["seq_length"],
            "batch_hash": "first-batch-hash",
        }

    monkeypatch.setattr(analysis, "run_single_comparison", fake_comparison)

    with pytest.raises(RuntimeError, match="second combination failed"):
        analysis.main()

    output_file = analysis._result_path(args)
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert calls == [8, 16]
    load_model.assert_called_once()
    assert payload["schema_version"] == analysis.SCHEMA_VERSION
    assert payload["source"] == {"git_commit": "a" * 40, "git_dirty": False}
    assert payload["batch_hashes"] == ["first-batch-hash"]
    assert len(payload["results"]) == 1
    assert payload["results"][0]["seq_length"] == 8


def test_existing_output_is_rejected_before_model_loading(tmp_path, monkeypatch) -> None:
    args = _args(tmp_path)
    output_file = analysis._result_path(args)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("existing", encoding="utf-8")
    load_model = mock.Mock(side_effect=AssertionError("model must not load"))
    monkeypatch.setattr(analysis, "load_model", load_model)
    monkeypatch.setattr(analysis, "parse_args", lambda: args)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        analysis.main()

    load_model.assert_not_called()


def test_explicit_output_file_still_requires_model_and_seed_tokens(tmp_path) -> None:
    args = _args(tmp_path, "--output-file", str(tmp_path / "result.json"))
    with pytest.raises(ValueError, match="model and seed"):
        analysis.validate_args(args, check_output=False)
