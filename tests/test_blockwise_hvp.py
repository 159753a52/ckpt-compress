import unittest

import torch

from dacp.tools.importance import (
    compute_hvp_batched,
    compute_hvp_blockwise,
    compute_hvp_blockwise_batched,
    compute_importance_scores_hvp,
    compute_importance_scores_hvp_abs,
    compute_importance_scores_hvp_blockwise,
)


class _QuadraticModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.right = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)


def _quadratic_loss(model, _batch):
    return 0.5 * model.left.square().sum() + model.right.square().sum()


class TestBlockwiseHvp(unittest.TestCase):
    def test_batched_hvp_rejects_empty_data_or_non_positive_budget(self) -> None:
        model = _QuadraticModel()
        vector = {"left": model.left.detach()}

        with self.assertRaisesRegex(ValueError, "data_batches"):
            compute_hvp_batched(model, _quadratic_loss, [], vector)
        with self.assertRaisesRegex(ValueError, "num_batches"):
            compute_hvp_blockwise_batched(model, _quadratic_loss, [None], [["left"]], num_batches=0)

    def test_blockwise_score_entrypoint_rejects_empty_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "data_batches"):
            compute_importance_scores_hvp_blockwise(
                _QuadraticModel(), _quadratic_loss, [], model_family="gpt2"
            )

    def test_blockwise_score_rejects_non_positive_gradient_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_batches"):
            compute_importance_scores_hvp_blockwise(
                _QuadraticModel(),
                _quadratic_loss,
                [None],
                model_family="gpt2",
                grad_accumulation_batches=0,
            )

    def test_signed_and_absolute_hvp_scores_share_the_same_taylor_terms(self) -> None:
        model = _QuadraticModel()

        signed = compute_importance_scores_hvp(
            model,
            _quadratic_loss,
            [None],
        )
        absolute = compute_importance_scores_hvp_abs(
            model,
            _quadratic_loss,
            [None],
        )

        torch.testing.assert_close(
            signed["left"],
            torch.tensor([-0.5, -2.0]),
        )
        torch.testing.assert_close(absolute["left"], signed["left"].abs())

    def test_matches_quadratic_hessian_vector_product(self) -> None:
        model = _QuadraticModel()

        hvp = compute_hvp_blockwise(
            model,
            _quadratic_loss,
            None,
            [["left"], ["right"]],
        )

        self.assertTrue(torch.equal(hvp["left"], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(hvp["right"], torch.tensor([6.0])))
        self.assertTrue(model.left.requires_grad)
        self.assertFalse(model.right.requires_grad)

    def test_restores_requires_grad_when_a_later_block_fails(self) -> None:
        model = _QuadraticModel()
        calls = 0

        def failing_loss(current_model, batch):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic loss failure")
            return _quadratic_loss(current_model, batch)

        with self.assertRaisesRegex(RuntimeError, "synthetic loss failure"):
            compute_hvp_blockwise(
                model,
                failing_loss,
                None,
                [["left"], ["right"]],
            )

        self.assertTrue(model.left.requires_grad)
        self.assertFalse(model.right.requires_grad)
        self.assertIsNone(model.left.grad)
        self.assertIsNone(model.right.grad)


class _ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(2.0))


def _scalar_quadratic_loss(model: _ScalarModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return 0.5 * batch["scale"] * (model.w - batch["target"]).square()


class TestImportanceScoresHvp(unittest.TestCase):
    def _batches(self) -> list[dict[str, torch.Tensor]]:
        return [
            {"scale": torch.tensor(1.0), "target": torch.tensor(0.0)},
            {"scale": torch.tensor(3.0), "target": torch.tensor(1.0)},
        ]

    def test_scalar_scores_average_gradient_and_hvp(self) -> None:
        model = _ScalarModel()
        batches = self._batches()

        two = compute_importance_scores_hvp(model, _scalar_quadratic_loss, batches, 2)
        one = compute_importance_scores_hvp(model, _scalar_quadratic_loss, batches, 1)
        capped = compute_importance_scores_hvp(model, _scalar_quadratic_loss, batches, 9)
        absolute = compute_importance_scores_hvp_abs(model, _scalar_quadratic_loss, batches, 2)

        torch.testing.assert_close(two["w"], torch.tensor(-1.0))
        torch.testing.assert_close(one["w"], torch.tensor(-2.0))
        torch.testing.assert_close(capped["w"], two["w"])
        torch.testing.assert_close(absolute["w"], torch.tensor(1.0))

    def test_success_restores_training_requires_grad_and_original_gradient(self) -> None:
        model = _ScalarModel()
        model.eval()
        model.w.grad = torch.tensor(7.0)
        original_gradient = model.w.grad
        original_value = original_gradient.detach().clone()

        def stateful_loss(
            current_model: _ScalarModel, batch: dict[str, torch.Tensor]
        ) -> torch.Tensor:
            current_model.train()
            return _scalar_quadratic_loss(current_model, batch)

        compute_importance_scores_hvp(model, stateful_loss, self._batches(), 2)

        self.assertFalse(model.training)
        self.assertTrue(model.w.requires_grad)
        self.assertIs(model.w.grad, original_gradient)
        torch.testing.assert_close(model.w.grad, original_value)

    def test_exception_restores_training_requires_grad_and_original_gradient(self) -> None:
        model = _ScalarModel()
        model.eval()
        model.w.grad = torch.tensor(7.0)
        original_gradient = model.w.grad
        original_value = original_gradient.detach().clone()

        def failing_loss(
            current_model: _ScalarModel, _batch: dict[str, torch.Tensor]
        ) -> torch.Tensor:
            current_model.train()
            current_model.w.requires_grad_(False)
            raise RuntimeError("synthetic HVP failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic HVP failure"):
            compute_importance_scores_hvp(model, failing_loss, self._batches(), 1)

        self.assertFalse(model.training)
        self.assertTrue(model.w.requires_grad)
        self.assertIs(model.w.grad, original_gradient)
        torch.testing.assert_close(model.w.grad, original_value)


if __name__ == "__main__":
    unittest.main()
