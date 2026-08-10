import unittest

import torch

from dacp.pruning import Pruner
from dacp.pruning.importance import (
    FirstOrderScorer,
    IMPORTANCE_REGISTRY,
    ImportanceScorer,
    MagnitudeScorer,
    ResidualMagnitudeScorer,
    apply_magnitude_protection,
    combine_scores_2d_with_protection,
    get_importance_scorer,
)
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
        with self.assertRaisesRegex(ValueError, "integer"):
            exact_pruning_mask(scores, 1.5)
        with self.assertRaisesRegex(ValueError, "integer"):
            exact_pruning_mask(scores, True)

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

        with self.assertRaisesRegex(ValueError, "Pruning ratio"):
            apply_pruning(model, {"left": torch.ones(4)}, {"left": 1.1})
        with self.assertRaisesRegex(ValueError, "Pruning ratio"):
            apply_pruning(model, {"left": torch.ones(4)}, {"left": True})
        with self.assertRaisesRegex(ValueError, "must match"):
            apply_pruning(model, {"left": torch.ones(2)}, {"left": 0.5})


class TestPrunerContracts(unittest.TestCase):
    def test_component_kwargs_are_forwarded_to_the_selected_component(self) -> None:
        class _ScaledMagnitude(ImportanceScorer):
            def __init__(self, scale: float = 1.0) -> None:
                super().__init__()
                self.scale = scale

            @property
            def name(self) -> str:
                return "scaled-magnitude"

            def score(self, weights, gradients=None, reference_weights=None):
                del gradients, reference_weights
                return {name: self.scale * weight.abs() for name, weight in weights.items()}

        previous = IMPORTANCE_REGISTRY.get("scaled-magnitude")
        IMPORTANCE_REGISTRY["scaled-magnitude"] = _ScaledMagnitude
        try:
            pruner = Pruner(
                importance="scaled-magnitude",
                allocation="weibull-adaptive",
                importance_kwargs={"scale": 3.0},
                allocation_kwargs={"max_layer_ratio": 0.25},
            )
            self.assertEqual(pruner.scorer.scale, 3.0)
            self.assertEqual(pruner.allocator.max_layer_ratio, 0.25)
        finally:
            if previous is None:
                IMPORTANCE_REGISTRY.pop("scaled-magnitude", None)
            else:
                IMPORTANCE_REGISTRY["scaled-magnitude"] = previous

    def test_missing_required_score_inputs_fail_at_pruner_boundary(self) -> None:
        weights = {"layer": torch.ones(2, 2)}

        with self.assertRaisesRegex(ValueError, "requires gradients"):
            Pruner(importance="first-order").compute_scores(weights)
        with self.assertRaisesRegex(ValueError, "requires reference_weights"):
            Pruner(importance="residual-magnitude").compute_scores(weights)

    def test_direct_scorers_reject_invalid_auxiliary_tensors(self) -> None:
        weights = {"layer": torch.ones(2, 2)}

        with self.assertRaisesRegex(ValueError, "requires gradients"):
            FirstOrderScorer().score(weights)
        with self.assertRaisesRegex(ValueError, "Gradient and weight shapes"):
            FirstOrderScorer().score(weights, {"layer": torch.ones(2)})
        with self.assertRaisesRegex(ValueError, "Reference weight and weight shapes"):
            ResidualMagnitudeScorer().score(
                weights,
                reference_weights={"layer": torch.ones(2)},
            )

    def test_registered_scorer_instances_are_not_called_as_factories(self) -> None:
        previous = IMPORTANCE_REGISTRY.get("instance-magnitude")
        IMPORTANCE_REGISTRY["instance-magnitude"] = MagnitudeScorer()
        try:
            scorer = get_importance_scorer("instance-magnitude")
            self.assertIsInstance(scorer, MagnitudeScorer)
            with self.assertRaisesRegex(TypeError, "does not accept"):
                get_importance_scorer("instance-magnitude", scale=2.0)
        finally:
            if previous is None:
                IMPORTANCE_REGISTRY.pop("instance-magnitude", None)
            else:
                IMPORTANCE_REGISTRY["instance-magnitude"] = previous

    def test_legacy_top_level_kwargs_are_not_silently_ignored(self) -> None:
        with self.assertRaises(TypeError):
            Pruner(importance="magnitude", alpha=0.5)

    def test_magnitude_protection_uses_exact_top_k_and_handles_zero_scores(self) -> None:
        weights = {"layer": torch.tensor([1.0, 2.0, 3.0, 4.0])}
        scores = {"layer": torch.zeros(4)}

        protected = apply_magnitude_protection(scores, weights, protection_ratio=0.1)

        self.assertEqual(int((protected["layer"] == torch.finfo(torch.float32).max).sum()), 1)
        self.assertEqual(protected["layer"][-1], torch.finfo(torch.float32).max)

    def test_score_combination_validates_shape_and_hyperparameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "shapes"):
            combine_scores_2d_with_protection(
                {"layer": torch.ones(2)},
                {"layer": torch.ones(3)},
            )
        with self.assertRaisesRegex(ValueError, "alpha"):
            combine_scores_2d_with_protection(
                {"layer": torch.ones(2)},
                {"layer": torch.ones(2)},
                alpha=1.1,
            )


if __name__ == "__main__":
    unittest.main()
