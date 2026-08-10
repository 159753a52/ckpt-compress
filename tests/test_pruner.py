import unittest

import torch

from dacp.pruning.pruner import apply_pruning, exact_pruning_mask


class _TwoParameterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.right = torch.nn.Parameter(torch.tensor([5.0]))


class TestExactPruningMask(unittest.TestCase):
    def test_ties_and_boundaries_keep_an_exact_budget(self) -> None:
        scores = torch.tensor([0.0, 0.0, 0.0, 1.0])

        self.assertTrue(
            torch.equal(exact_pruning_mask(scores, 0), torch.ones(4))
        )
        self.assertTrue(
            torch.equal(
                exact_pruning_mask(scores, 2),
                torch.tensor([0.0, 0.0, 1.0, 1.0]),
            )
        )
        self.assertTrue(
            torch.equal(exact_pruning_mask(scores, 4), torch.zeros(4))
        )

        with self.assertRaisesRegex(ValueError, "prune_count"):
            exact_pruning_mask(scores, 5)

    def test_apply_pruning_counts_zero_budget_layers(self) -> None:
        model = _TwoParameterModel()
        scores = {
            "left": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            "right": torch.tensor([1.0]),
        }

        returned_model, masks, actual_ratio = apply_pruning(
            model,
            scores,
            {"left": 0.5, "right": 0.2},
        )

        self.assertIs(returned_model, model)
        self.assertEqual(masks["left"].dtype, torch.float32)
        self.assertTrue(
            torch.equal(masks["left"], torch.tensor([0.0, 0.0, 1.0, 1.0]))
        )
        self.assertTrue(torch.equal(masks["right"], torch.ones(1)))
        self.assertTrue(torch.equal(model.left, torch.tensor([0.0, 0.0, 3.0, 4.0])))
        self.assertTrue(torch.equal(model.right, torch.tensor([5.0])))
        self.assertAlmostEqual(actual_ratio, 2 / 5)

    def test_apply_pruning_validates_ratio_and_shape(self) -> None:
        model = _TwoParameterModel()

        with self.assertRaisesRegex(ValueError, "must be in"):
            apply_pruning(model, {"left": torch.ones(4)}, {"left": 1.1})
        with self.assertRaisesRegex(ValueError, "must match"):
            apply_pruning(model, {"left": torch.ones(2)}, {"left": 0.5})


if __name__ == "__main__":
    unittest.main()
