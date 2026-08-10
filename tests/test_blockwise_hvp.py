import unittest

import torch

from dacp.tools.importance import compute_hvp_blockwise


class _QuadraticModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.right = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)


def _quadratic_loss(model, _batch):
    return 0.5 * model.left.square().sum() + model.right.square().sum()


class TestBlockwiseHvp(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
