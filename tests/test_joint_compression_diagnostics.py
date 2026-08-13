from unittest import mock

import pytest
import torch
import torch.nn as nn

from experiments.scripts.run_joint_compression import (
    _validate_reconstructed_parameters,
    run_excp_first_checkpoint_diagnostic,
)


def test_excp_diagnostic_reports_measured_sparsity_without_fake_target_ratio() -> None:
    model = nn.Linear(3, 2)
    cached_eval = [{"input_ids": torch.ones(1, 3), "labels": torch.zeros(1, dtype=torch.long)}]

    with mock.patch(
        "experiments.scripts.run_joint_compression.evaluate",
        return_value={"loss": 1.25},
    ):
        result = run_excp_first_checkpoint_diagnostic(
            model,
            cached_eval,
            "cls",
            "cpu",
        )

    assert result["method"] == "ExCP (first-checkpoint diagnostic)"
    assert result["fidelity"] == "style"
    assert result["payload_scope"] == "weights_and_exp_avg"
    assert result["optimizer_state_source"].startswith("synthetic_")
    assert result["optimizer_state_restored"] is False
    assert "prune_ratio" not in result
    assert 0.0 <= result["actual_prune_ratio"] <= 1.0
    assert result["loss"] == 1.25


def test_adapter_reconstruction_must_cover_every_parameter_with_matching_shape() -> None:
    expected = {"weight": torch.ones(2, 2), "bias": torch.ones(2)}
    with pytest.raises(RuntimeError, match="keys must match"):
        _validate_reconstructed_parameters(
            expected,
            {"weight": torch.ones(2, 2)},
            "adapter",
        )
    with pytest.raises(ValueError, match="shape"):
        _validate_reconstructed_parameters(
            expected,
            {"weight": torch.ones(4), "bias": torch.ones(2)},
            "adapter",
        )
