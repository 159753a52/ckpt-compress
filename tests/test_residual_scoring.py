import io
import math
import unittest
from contextlib import redirect_stdout
from typing import Mapping
from unittest import mock

import torch

import experiments.lib.residual_scoring as residual_scoring
from experiments.lib.residual_scoring import (
    compute_block_first_order_scores,
    compute_block_taylor_scores,
    eligible_layers,
    model_checksum,
)


class ToyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))
        self.right = torch.nn.Parameter(
            torch.tensor([[-1.0, 0.25], [2.0, -0.5]]),
            requires_grad=False,
        )


class ToyTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.h = torch.nn.ModuleList([ToyBlock()])


class ToyScoringModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = ToyTransformer()


class TestResidualScoring(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ToyScoringModel()
        self.layers = [["transformer.h.0.left", "transformer.h.0.right"]]
        self.delta = {
            "transformer.h.0.left": torch.tensor([[0.5, -0.25], [1.0, -0.5]]),
            "transformer.h.0.right": torch.tensor([[-0.5, 1.0], [0.25, -1.0]]),
        }
        self.batches = [
            {"scale": torch.tensor(1.0)},
            {"scale": torch.tensor(3.0)},
        ]

    @staticmethod
    def quadratic_loss(
        model: ToyScoringModel,
        batch: Mapping[str, torch.Tensor],
        device: str,
    ) -> torch.Tensor:
        del device
        block = model.transformer.h[0]
        return batch["scale"] * (0.5 * block.left.square().sum() + block.right.square().sum())

    def test_eligible_layers_and_checksum(self) -> None:
        self.assertEqual(model_checksum(self.model), (3.25, 10.25))
        self.assertEqual(eligible_layers(self.model), [["transformer.h.0.left"]])

    def test_paper_model_family_parameter_names_form_structural_layers(self) -> None:
        cases = {
            "gpt2": ["transformer.h.0.attn.weight", "transformer.h.1.mlp.weight"],
            "bert": [
                "bert.encoder.layer.0.attention.weight",
                "bert.encoder.layer.1.intermediate.weight",
            ],
            "pythia": [
                "gpt_neox.layers.0.attention.weight",
                "gpt_neox.layers.1.mlp.weight",
            ],
        }
        for family, names in cases.items():
            with self.subTest(family=family):
                model = torch.nn.Module()
                for index, name in enumerate(names):
                    owner = model
                    parts = name.split(".")
                    for part in parts[:-1]:
                        if part not in owner._modules:
                            owner.add_module(part, torch.nn.Module())
                        owner = owner._modules[part]
                    owner.register_parameter(parts[-1], torch.nn.Parameter(torch.ones(2, 2)))
                self.assertEqual(eligible_layers(model, family), [[names[0]], [names[1]]])

    def test_block_taylor_components_match_quadratic_loss(self) -> None:
        named_params = dict(self.model.named_parameters())
        weights_before = {
            name: parameter.detach().clone() for name, parameter in named_params.items()
        }
        versions_before = {name: parameter._version for name, parameter in named_params.items()}
        requires_grad_before = {
            name: parameter.requires_grad for name, parameter in named_params.items()
        }

        with (
            mock.patch.object(
                residual_scoring,
                "lm_loss",
                side_effect=self.quadratic_loss,
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            components, metadata = compute_block_taylor_scores(
                self.model,
                self.batches,
                self.layers,
                self.delta,
                device="cpu",
                return_components=True,
            )

        left = weights_before["transformer.h.0.left"]
        right = weights_before["transformer.h.0.right"]
        left_delta = self.delta["transformer.h.0.left"]
        right_delta = self.delta["transformer.h.0.right"]
        expected = {
            "first_order": {
                "transformer.h.0.left": (-2.0 * left * left_delta).abs(),
                "transformer.h.0.right": (-4.0 * right * right_delta).abs(),
            },
            "second_order": {
                "transformer.h.0.left": left_delta.square(),
                "transformer.h.0.right": 2.0 * right_delta.square(),
            },
            "taylor": {
                "transformer.h.0.left": (-2.0 * left * left_delta + left_delta.square()).abs(),
                "transformer.h.0.right": (
                    -4.0 * right * right_delta + 2.0 * right_delta.square()
                ).abs(),
            },
        }
        self.assertEqual(set(components), set(expected))
        for component_name, expected_scores in expected.items():
            self.assertEqual(set(components[component_name]), set(expected_scores))
            for name, expected_score in expected_scores.items():
                torch.testing.assert_close(
                    components[component_name][name],
                    expected_score,
                )
                self.assertEqual(components[component_name][name].device.type, "cpu")
                self.assertEqual(components[component_name][name].dtype, torch.float32)

        self.assertIn("HVP layer 01/01", output.getvalue())
        self.assertEqual(
            set(metadata),
            {
                "total_seconds",
                "score_kind",
                "layer_seconds",
                "checksum_before",
                "checksum_after",
                "checksum_delta",
                "changed_parameter_versions",
                "optimizer_constructed",
                "model_mode",
                "hvp_batches",
                "task_type",
                "model_family",
                "aggregation",
            },
        )
        self.assertEqual(metadata["checksum_before"], [3.25, 10.25])
        self.assertEqual(metadata["checksum_after"], [3.25, 10.25])
        self.assertEqual(metadata["checksum_delta"], [0.0, 0.0])
        self.assertEqual(metadata["changed_parameter_versions"], [])
        self.assertFalse(metadata["optimizer_constructed"])
        self.assertEqual(metadata["model_mode"], "eval")
        self.assertEqual(metadata["hvp_batches"], 2)
        self.assertEqual(metadata["task_type"], "lm")
        self.assertEqual(metadata["model_family"], "gpt2")
        self.assertEqual(metadata["score_kind"], "taylor_hvp")
        self.assertEqual(metadata["aggregation"], "abs_mean")
        self.assertEqual(len(metadata["layer_seconds"]), 1)
        for duration in [metadata["total_seconds"], *metadata["layer_seconds"]]:
            self.assertTrue(math.isfinite(duration))
            self.assertGreaterEqual(duration, 0.0)

        for name, parameter in named_params.items():
            self.assertTrue(torch.equal(parameter, weights_before[name]))
            self.assertEqual(parameter._version, versions_before[name])
            self.assertEqual(parameter.requires_grad, requires_grad_before[name])
            self.assertIsNone(parameter.grad)
        self.assertTrue(self.model.training)

    def test_taylor_aggregation_modes(self) -> None:
        """signed_mean keeps sign; mean_abs avoids cross-batch cancellation."""
        batches = [
            {"scale": torch.tensor(1.0)},
            {"scale": torch.tensor(-1.0)},
        ]
        left_name = "transformer.h.0.left"
        results = {}
        for mode in ("abs_mean", "mean_abs", "signed_mean"):
            with (
                mock.patch.object(
                    residual_scoring,
                    "lm_loss",
                    side_effect=self.quadratic_loss,
                ),
                redirect_stdout(io.StringIO()),
            ):
                scores, metadata = compute_block_taylor_scores(
                    self.model,
                    batches,
                    self.layers,
                    self.delta,
                    device="cpu",
                    aggregation=mode,
                )
            results[mode] = (scores, metadata)

        # Contributions at scale +1 and -1 cancel exactly under signed averaging.
        torch.testing.assert_close(
            results["signed_mean"][0][left_name],
            torch.zeros_like(self.delta[left_name]),
        )
        torch.testing.assert_close(
            results["abs_mean"][0][left_name],
            torch.zeros_like(self.delta[left_name]),
        )
        # mean_abs keeps the magnitude of non-cancelling contributions.
        self.assertGreater(results["mean_abs"][0][left_name][1, 1].item(), 0.0)
        for mode, (_, metadata) in results.items():
            self.assertEqual(metadata["aggregation"], mode)

    def test_taylor_aggregation_rejects_unknown_mode(self) -> None:
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                compute_block_taylor_scores(
                    self.model,
                    self.batches,
                    self.layers,
                    self.delta,
                    device="cpu",
                    aggregation="bogus",
                )

    def test_explicit_block_groups_support_non_gpt_parameter_names(self) -> None:
        with redirect_stdout(io.StringIO()):
            scores, metadata = compute_block_taylor_scores(
                self.model,
                self.batches,
                [["transformer.h.0.left"]],
                {"transformer.h.0.left": self.delta["transformer.h.0.left"]},
                device="cpu",
                task_type="cls",
                loss_fn=self.quadratic_loss,
                block_parameter_names=[["transformer.h.0.left", "transformer.h.0.right"]],
            )

        self.assertEqual(set(scores), {"transformer.h.0.left"})
        self.assertEqual(metadata["task_type"], "cls")

    def test_scoring_restores_mode_and_rejects_disconnected_eligible_parameters(self) -> None:
        self.model.eval()
        left_grad = torch.full_like(self.model.transformer.h[0].left, 7.0)
        right_grad = torch.full_like(self.model.transformer.h[0].right, 9.0)
        self.model.transformer.h[0].left.grad = left_grad
        self.model.transformer.h[0].right.grad = right_grad

        def disconnected_loss(
            model: ToyScoringModel,
            batch: Mapping[str, torch.Tensor],
            device: str,
        ) -> torch.Tensor:
            del batch, device
            return model.transformer.h[0].left.square().sum()

        with self.assertRaisesRegex(RuntimeError, "missing"):
            compute_block_taylor_scores(
                self.model,
                self.batches,
                self.layers,
                self.delta,
                device="cpu",
                loss_fn=disconnected_loss,
                block_parameter_names=[["transformer.h.0.left", "transformer.h.0.right"]],
            )

        self.assertFalse(self.model.training)
        self.assertTrue(self.model.transformer.h[0].left.requires_grad)
        self.assertFalse(self.model.transformer.h[0].right.requires_grad)
        self.assertIs(self.model.transformer.h[0].left.grad, left_grad)
        self.assertIs(self.model.transformer.h[0].right.grad, right_grad)
        torch.testing.assert_close(self.model.transformer.h[0].left.grad, torch.full((2, 2), 7.0))
        torch.testing.assert_close(self.model.transformer.h[0].right.grad, torch.full((2, 2), 9.0))

    def test_first_order_scoring_preserves_existing_gradients_without_zero_grad(self) -> None:
        self.model.eval()
        left_grad = torch.full_like(self.model.transformer.h[0].left, 4.0)
        right_grad = torch.full_like(self.model.transformer.h[0].right, 6.0)
        self.model.transformer.h[0].left.grad = left_grad
        self.model.transformer.h[0].right.grad = right_grad

        with (
            mock.patch.object(
                self.model,
                "zero_grad",
                side_effect=AssertionError("zero_grad called"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            scores, _ = compute_block_first_order_scores(
                self.model,
                self.batches,
                self.layers,
                self.delta,
                device="cpu",
                loss_fn=self.quadratic_loss,
                block_parameter_names=self.layers,
            )

        self.assertEqual(set(scores), set(self.delta))
        self.assertFalse(self.model.training)
        self.assertIs(self.model.transformer.h[0].left.grad, left_grad)
        self.assertIs(self.model.transformer.h[0].right.grad, right_grad)
        torch.testing.assert_close(self.model.transformer.h[0].left.grad, torch.full((2, 2), 4.0))
        torch.testing.assert_close(self.model.transformer.h[0].right.grad, torch.full((2, 2), 6.0))

    def test_scoring_rejects_invalid_probe_before_changing_model_state(self) -> None:
        self.model.eval()
        invalid_delta = dict(self.delta)
        invalid_delta["transformer.h.0.left"] = torch.ones(4)

        with self.assertRaisesRegex(ValueError, "probe shape"):
            compute_block_taylor_scores(
                self.model,
                self.batches,
                self.layers,
                invalid_delta,
                device="cpu",
                loss_fn=self.quadratic_loss,
                block_parameter_names=self.layers,
            )

        self.assertFalse(self.model.training)
        self.assertTrue(self.model.transformer.h[0].left.requires_grad)
        self.assertFalse(self.model.transformer.h[0].right.requires_grad)


if __name__ == "__main__":
    unittest.main()
