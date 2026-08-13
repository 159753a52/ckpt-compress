import unittest
from unittest import mock

import torch

from experiments.lib.importance_compare.scoring import (
    _collect_gradients,
    _complete_auxiliary_mapping,
    _compute_hvp_scores,
    _make_loss_fn,
    _require_gradients,
    compute_scores_by_method,
)


class ToyRegressionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[2.0]]))
        self.unused = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        return input_ids @ self.weight


class TestImportanceScoringContract(unittest.TestCase):
    def test_gradient_dependent_methods_fail_closed_without_gradients(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires collected gradients"):
            _require_gradients(None, "second-order-hvp")
        with self.assertRaisesRegex(RuntimeError, "missing prunable parameters"):
            _complete_auxiliary_mapping(
                {"left": torch.ones(1), "right": torch.ones(1)},
                {"left": torch.ones(1)},
                "Gradient",
            )

    def test_loss_factory_rejects_unknown_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown task_type"):
            _make_loss_fn("unknown")

    def test_empty_methods_return_without_touching_model(self) -> None:
        self.assertEqual(
            compute_scores_by_method(
                None,
                [],
                "lm",
                methods=[],
            ),
            {},
        )

    def test_method_and_hvp_arguments_are_validated_before_model_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown importance method"):
            compute_scores_by_method(None, [], "lm", methods=["invalid"])

        with self.assertRaisesRegex(ValueError, "Unknown task_type"):
            compute_scores_by_method(None, [], "unknown", methods=["magnitude"])

        with self.assertRaisesRegex(ValueError, "Unknown hvp_mode"):
            compute_scores_by_method(
                None,
                [],
                "lm",
                methods=["magnitude"],
                hvp_mode="invalid",
            )

        with self.assertRaisesRegex(ValueError, "hvp_batches"):
            compute_scores_by_method(
                None,
                [],
                "lm",
                methods=["magnitude"],
                hvp_batches=0,
            )

        with self.assertRaisesRegex(ValueError, "grad_batches_first_order"):
            compute_scores_by_method(
                None,
                [],
                "lm",
                methods=["magnitude"],
                grad_batches_first_order=True,
            )

    def test_gradient_collection_rejects_empty_batches_and_non_positive_count(self) -> None:
        model = torch.nn.Linear(1, 1)

        with self.assertRaisesRegex(ValueError, "cached_train"):
            _collect_gradients(model, [], "lm", 1, "cpu")

        with self.assertRaisesRegex(ValueError, "num_batches"):
            _collect_gradients(model, [{}], "lm", 0, "cpu")

    def test_gradient_collection_uses_autograd_grad_and_restores_model_state(self) -> None:
        model = ToyRegressionModel()
        model.eval()
        original_grad = torch.tensor([[7.0]])
        model.weight.grad = original_grad
        batches = [
            {
                "input_ids": torch.tensor([[1.0]]),
                "labels": torch.tensor([[0.0]]),
            },
            {
                "input_ids": torch.tensor([[2.0]]),
                "labels": torch.tensor([[0.0]]),
            },
        ]

        with mock.patch.object(model, "zero_grad", side_effect=AssertionError("zero_grad called")):
            gradients = _collect_gradients(model, batches, "reg", 2, "cpu")

        torch.testing.assert_close(gradients["weight"], torch.tensor([[10.0]]))
        self.assertFalse(model.training)
        self.assertIs(model.weight.grad, original_grad)
        torch.testing.assert_close(model.weight.grad, torch.tensor([[7.0]]))
        self.assertIsNone(model.unused.grad)

    def test_gradient_collection_restores_state_when_loss_is_non_finite(self) -> None:
        model = ToyRegressionModel()
        model.eval()
        original_grad = torch.tensor([[3.0]])
        model.weight.grad = original_grad
        batch = {
            "input_ids": torch.tensor([[float("nan")]]),
            "labels": torch.tensor([[0.0]]),
        }

        with self.assertRaisesRegex(ValueError, "finite"):
            _collect_gradients(model, [batch], "reg", 1, "cpu")

        self.assertFalse(model.training)
        self.assertIs(model.weight.grad, original_grad)
        torch.testing.assert_close(model.weight.grad, torch.tensor([[3.0]]))

    def test_auxiliary_mapping_validates_shape_device_dtype_and_finite_values(self) -> None:
        weights = {"weight": torch.ones(2, 2)}
        bad_values = (
            torch.ones(4),
            torch.ones(2, 2, dtype=torch.int64),
            torch.full((2, 2), float("inf")),
            torch.ones(2, 2, device="meta"),
        )
        for bad_value in bad_values:
            with self.subTest(value=bad_value):
                with self.assertRaises((TypeError, ValueError)):
                    _complete_auxiliary_mapping(weights, {"weight": bad_value}, "Gradient")

    def test_hvp_scoring_restores_state_and_rejects_incomplete_hvp(self) -> None:
        model = ToyRegressionModel()
        model.eval()
        original_grad = torch.tensor([[5.0]])
        model.weight.grad = original_grad
        batches = [
            {
                "input_ids": torch.tensor([[1.0]]),
                "labels": torch.tensor([[0.0]]),
            }
        ]

        with mock.patch(
            "dacp.tools.importance.compute_hvp_batched",
            return_value={},
        ):
            with self.assertRaisesRegex(RuntimeError, "HVP coverage"):
                _compute_hvp_scores(
                    model,
                    batches,
                    "reg",
                    0.5,
                    1,
                    "full",
                    1,
                    "cpu",
                    False,
                    {"weight": torch.ones(1, 1)},
                )

        self.assertFalse(model.training)
        self.assertTrue(model.weight.requires_grad)
        self.assertIs(model.weight.grad, original_grad)
        torch.testing.assert_close(model.weight.grad, torch.tensor([[5.0]]))


if __name__ == "__main__":
    unittest.main()
