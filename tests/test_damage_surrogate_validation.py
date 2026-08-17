import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn as nn

import experiments.scripts.run_damage_surrogate_validation as damage_script
from experiments.lib.damage_surrogate_validation import MaskCandidate, validate_damage_surrogate


class QuadraticToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.bias = nn.Parameter(torch.tensor([7.0]))
        self.dropout = nn.Dropout(p=0.5)


def _loss(model: nn.Module, batch, _device: str) -> torch.Tensor:
    return ((model.weight - batch["target"]) ** 2).mean()


def _toy_inputs():
    layers = [["weight"]]
    delta = {"weight": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    previous = {"weight": torch.zeros(4)}
    batches = [{"target": torch.tensor([1.0, 2.0, 3.0, 4.0])}]
    masks = [
        MaskCandidate(
            id="low_pair",
            source="magnitude",
            mask={"weight": torch.tensor([False, False, True, True])},
            score_family="quadratic",
        ),
        MaskCandidate(
            id="middle_pair",
            source="random",
            mask={"weight": torch.tensor([True, False, False, True])},
            score_family="quadratic",
        ),
        MaskCandidate(
            id="high_pair",
            source="taylor",
            mask={"weight": torch.tensor([True, True, False, False])},
            score_family="quadratic",
        ),
    ]
    scores = {"quadratic": {"weight": torch.tensor([1.0, 4.0, 9.0, 16.0])}}
    return layers, delta, previous, batches, masks, scores


class TestDamageSurrogateValidation(unittest.TestCase):
    def test_predicted_and_actual_losses_and_rank_statistics(self) -> None:
        layers, delta, previous, batches, masks, scores = _toy_inputs()
        model = QuadraticToy()
        result = validate_damage_surrogate(
            model,
            delta,
            layers,
            batches,
            masks,
            scores,
            previous_reconstructed=previous,
            device="cpu",
            loss_fn=_loss,
        )

        self.assertEqual(result["mask_count"], 3)
        self.assertEqual(result["pruned_count"], 2)
        self.assertEqual(result["correlations"]["quadratic"]["spearman"], 1.0)
        self.assertEqual(result["correlations"]["quadratic"]["kendall"], 1.0)
        records = {record["id"]: record for record in result["records"]}
        self.assertEqual(records["low_pair"]["predicted_cost"], 5.0)
        self.assertAlmostEqual(records["low_pair"]["actual_delta_loss"], 1.25)
        self.assertAlmostEqual(records["high_pair"]["masked_loss"], 6.25)
        for record in records.values():
            self.assertEqual(record["base_loss"], 0.0)

    def test_deterministic_result_and_exact_budget(self) -> None:
        layers, delta, previous, batches, masks, scores = _toy_inputs()
        first = validate_damage_surrogate(
            QuadraticToy(),
            delta,
            layers,
            batches,
            masks,
            scores,
            previous_reconstructed=previous,
            device="cpu",
            loss_fn=_loss,
        )
        second = validate_damage_surrogate(
            QuadraticToy(),
            delta,
            layers,
            batches,
            masks,
            scores,
            previous_reconstructed=previous,
            device="cpu",
            loss_fn=_loss,
        )
        self.assertEqual(first, second)
        self.assertTrue(all(record["pruned_count"] == 2 for record in first["records"]))

    def test_restores_state_modes_requires_grad_and_grad_identity(self) -> None:
        layers, delta, previous, batches, masks, scores = _toy_inputs()
        model = QuadraticToy()
        model.train()
        model.dropout.eval()
        model.bias.requires_grad_(False)
        weight_grad = torch.full_like(model.weight, 3.0)
        bias_grad = torch.full_like(model.bias, 4.0)
        model.weight.grad = weight_grad
        model.bias.grad = bias_grad
        before_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        before_modes = {name: module.training for name, module in model.named_modules()}

        validate_damage_surrogate(
            model,
            delta,
            layers,
            batches,
            masks,
            scores,
            previous_reconstructed=previous,
            device="cpu",
            loss_fn=_loss,
        )

        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before_state[name])
        self.assertEqual(
            before_modes,
            {name: module.training for name, module in model.named_modules()},
        )
        self.assertTrue(model.bias.requires_grad is False)
        self.assertIs(model.weight.grad, weight_grad)
        self.assertIs(model.bias.grad, bias_grad)
        torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 3.0))
        torch.testing.assert_close(model.bias.grad, torch.full_like(model.bias, 4.0))

    def test_rejects_invalid_inputs(self) -> None:
        for mutation, message in (
            ("nonfinite_score", "finite"),
            ("negative_score", "non-negative"),
            ("different_count", "one prune count"),
            ("duplicate_mask", "Duplicate candidate mask"),
        ):
            with self.subTest(mutation=mutation):
                layers, delta, previous, batches, masks, scores = _toy_inputs()
                if mutation == "nonfinite_score":
                    scores["quadratic"]["weight"][0] = float("nan")
                elif mutation == "negative_score":
                    scores["quadratic"]["weight"][0] = -1.0
                elif mutation == "different_count":
                    masks.append(
                        MaskCandidate(
                            id="bad_budget",
                            source="random",
                            mask={"weight": torch.tensor([False, True, True, True])},
                        )
                    )
                elif mutation == "duplicate_mask":
                    masks.append(copy.copy(masks[0]))
                    masks[-1] = MaskCandidate(
                        id="duplicate",
                        source="random",
                        mask=masks[-1].mask,
                    )
                with pytest.raises(ValueError, match=message):
                    validate_damage_surrogate(
                        QuadraticToy(),
                        delta,
                        layers,
                        batches,
                        masks,
                        scores,
                        previous_reconstructed=previous,
                        device="cpu",
                        loss_fn=_loss,
                    )

    def test_rejects_nonfinite_loss(self) -> None:
        layers, delta, previous, batches, masks, scores = _toy_inputs()

        def bad_loss(_model, _batch, _device):
            return torch.tensor(float("inf"))

        with pytest.raises(ValueError, match="loss.*finite"):
            validate_damage_surrogate(
                QuadraticToy(),
                delta,
                layers,
                batches,
                masks,
                scores,
                previous_reconstructed=previous,
                device="cpu",
                loss_fn=bad_loss,
            )


class TestDamageSurrogateScript(unittest.TestCase):
    def test_default_dry_run_does_not_load_model_or_data(self) -> None:
        with (
            mock.patch.object(damage_script, "load_model") as load_model,
            mock.patch.object(damage_script, "get_data_loaders") as get_data_loaders,
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                damage_script.main(["--dry-run"])

        plan = json.loads(output.getvalue())
        self.assertEqual(plan["status"], "dry_run")
        self.assertEqual(plan["workload"], "gpt2_medium_wikitext103")
        self.assertEqual(plan["seed"], 42)
        self.assertEqual(plan["prune_ratio"], 0.5)
        self.assertEqual(plan["recovery_count"], 1)
        self.assertEqual(plan["mask_count"], 32)
        self.assertFalse(plan["loads_model"])
        self.assertFalse(plan["loads_gpu"])
        load_model.assert_not_called()
        get_data_loaders.assert_not_called()

    def test_existing_output_blocks_non_dry_run_before_runtime_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "existing.json"
            original = "preserve this evidence\n"
            output_path.write_text(original, encoding="utf-8")
            with (
                mock.patch.object(damage_script, "load_training_checkpoint") as load_checkpoint,
                mock.patch.object(damage_script, "get_data_loaders") as get_data_loaders,
                mock.patch.object(damage_script, "load_model") as load_model,
                mock.patch.object(damage_script.torch.cuda, "is_available") as cuda_available,
            ):
                with self.assertRaises(FileExistsError):
                    damage_script.main(["--output", str(output_path), "--device", "cuda"])

            self.assertEqual(output_path.read_text(encoding="utf-8"), original)
            load_checkpoint.assert_not_called()
            get_data_loaders.assert_not_called()
            load_model.assert_not_called()
            cuda_available.assert_not_called()

    def test_dry_run_reports_existing_output_without_loading_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "existing.json"
            original = "preserve this evidence\n"
            output_path.write_text(original, encoding="utf-8")
            with (
                mock.patch.object(damage_script, "load_training_checkpoint") as load_checkpoint,
                mock.patch.object(damage_script, "get_data_loaders") as get_data_loaders,
                mock.patch.object(damage_script, "load_model") as load_model,
                mock.patch.object(damage_script.torch.cuda, "is_available") as cuda_available,
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    damage_script.main(
                        ["--dry-run", "--output", str(output_path), "--device", "cuda"]
                    )

            plan = json.loads(output.getvalue())
            self.assertTrue(plan["output_exists"])
            self.assertEqual(output_path.read_text(encoding="utf-8"), original)
            load_checkpoint.assert_not_called()
            get_data_loaders.assert_not_called()
            load_model.assert_not_called()
            cuda_available.assert_not_called()


if __name__ == "__main__":
    unittest.main()
