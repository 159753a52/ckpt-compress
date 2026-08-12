import io
import math
import unittest
from contextlib import redirect_stdout
from typing import Mapping
from unittest import mock

import torch

import experiments.lib.residual_scoring as residual_scoring
from experiments.lib.residual_scoring import (
    compute_block_taylor_scores,
    eligible_layers,
    model_checksum,
)


class ToyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(
            torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        )
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
            "transformer.h.0.left": torch.tensor(
                [[0.5, -0.25], [1.0, -0.5]]
            ),
            "transformer.h.0.right": torch.tensor(
                [[-0.5, 1.0], [0.25, -1.0]]
            ),
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
        return batch["scale"] * (
            0.5 * block.left.square().sum() + block.right.square().sum()
        )

    def test_eligible_layers_and_checksum(self) -> None:
        self.assertEqual(model_checksum(self.model), (3.25, 10.25))
        self.assertEqual(eligible_layers(self.model), [["transformer.h.0.left"]])

    def test_block_taylor_components_match_quadratic_loss(self) -> None:
        named_params = dict(self.model.named_parameters())
        weights_before = {
            name: parameter.detach().clone()
            for name, parameter in named_params.items()
        }
        versions_before = {name: parameter._version for name, parameter in named_params.items()}
        requires_grad_before = {
            name: parameter.requires_grad for name, parameter in named_params.items()
        }

        with mock.patch.object(
            residual_scoring,
            "lm_loss",
            side_effect=self.quadratic_loss,
        ), redirect_stdout(io.StringIO()) as output:
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
                "transformer.h.0.left": (
                    -2.0 * left * left_delta + left_delta.square()
                ).abs(),
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
        self.assertEqual(len(metadata["layer_seconds"]), 1)
        for duration in [metadata["total_seconds"], *metadata["layer_seconds"]]:
            self.assertTrue(math.isfinite(duration))
            self.assertGreaterEqual(duration, 0.0)

        for name, parameter in named_params.items():
            self.assertTrue(torch.equal(parameter, weights_before[name]))
            self.assertEqual(parameter._version, versions_before[name])
            self.assertEqual(parameter.requires_grad, requires_grad_before[name])
            self.assertIsNone(parameter.grad)
        self.assertFalse(self.model.training)

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
                block_parameter_names=[
                    ["transformer.h.0.left", "transformer.h.0.right"]
                ],
            )

        self.assertEqual(set(scores), {"transformer.h.0.left"})
        self.assertEqual(metadata["task_type"], "cls")


if __name__ == "__main__":
    unittest.main()
