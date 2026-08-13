import unittest
from unittest.mock import patch

import torch

from dacp.pruning.allocation import (
    GammaAdaptiveAllocation,
    GlobalTopKAllocation,
    UniformAllocation,
    WeibullAdaptiveAllocation,
)


class TestGlobalTopKAllocation(unittest.TestCase):
    def test_empty_scores_return_an_empty_allocation(self) -> None:
        self.assertEqual(GlobalTopKAllocation().allocate({}, 0.5), {})

    def test_ties_still_receive_the_exact_global_budget(self) -> None:
        scores = {
            "left": torch.ones(3),
            "right": torch.ones(2),
        }

        rates = GlobalTopKAllocation().allocate(scores, 0.4)

        self.assertEqual(
            sum(int(scores[name].numel() * rates[name]) for name in scores),
            2,
        )
        self.assertEqual(rates, {"left": 2 / 3, "right": 0.0})

    def test_boundary_ratios_do_not_use_the_approximate_path(self) -> None:
        scores = {
            "empty": torch.empty(0),
            "active": torch.ones(3),
        }

        self.assertEqual(
            GlobalTopKAllocation().allocate(scores, 0.0),
            {"empty": 0.0, "active": 0.0},
        )
        self.assertEqual(
            GlobalTopKAllocation().allocate(scores, 1.0),
            {"empty": 0.0, "active": 1.0},
        )

    def test_empty_layers_do_not_divide_by_zero(self) -> None:
        scores = {
            "empty": torch.empty(0),
            "active": torch.tensor([1.0, 2.0]),
        }

        rates = GlobalTopKAllocation().allocate(scores, 0.5)

        self.assertEqual(rates["empty"], 0.0)
        self.assertEqual(rates["active"], 0.5)

    def test_large_path_keeps_exact_budget_with_tied_scores(self) -> None:
        scores = {
            "left": torch.zeros(3),
            "right": torch.tensor([0.0, 1.0]),
        }

        with patch("dacp.pruning.allocation._GLOBAL_SORT_MAX_PARAMS", 4):
            rates = GlobalTopKAllocation().allocate(scores, 0.8)

        self.assertEqual(rates, {"left": 1.0, "right": 0.5})
        self.assertEqual(
            sum(round(scores[name].numel() * rates[name]) for name in scores),
            4,
        )

    def test_nonfinite_scores_are_rejected_by_data_dependent_allocators(self) -> None:
        scores = {"layer": torch.tensor([0.0, float("nan")])}

        for allocator in (
            GlobalTopKAllocation(),
            GammaAdaptiveAllocation(),
            WeibullAdaptiveAllocation(),
        ):
            with self.subTest(allocator=allocator.name):
                with self.assertRaisesRegex(ValueError, "finite"):
                    allocator.allocate(scores, 0.5)

    def test_score_preparation_detaches_autograd_history(self) -> None:
        scores = {
            "layer": torch.arange(1.0, 13.0, requires_grad=True),
        }

        for allocator in (
            GlobalTopKAllocation(),
            GammaAdaptiveAllocation(),
            WeibullAdaptiveAllocation(max_layer_ratio=1.0),
        ):
            with self.subTest(allocator=allocator.name):
                rates = allocator.allocate(scores, 0.5)

                self.assertEqual(set(rates), {"layer"})

    def test_ratio_is_validated_before_processing_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "global_prune_ratio"):
            GlobalTopKAllocation().allocate({}, 1.1)

    def test_other_allocators_handle_empty_layers_and_validate_ratio(self) -> None:
        scores = {"empty": torch.empty(0)}

        self.assertEqual(GammaAdaptiveAllocation().allocate(scores, 0.5), {"empty": 0.0})
        self.assertEqual(WeibullAdaptiveAllocation().allocate(scores, 0.5), {"empty": 0.0})
        self.assertEqual(UniformAllocation().allocate(scores, 0.5), {"empty": 0.5})

        with self.assertRaisesRegex(ValueError, "global_prune_ratio"):
            GammaAdaptiveAllocation().allocate(scores, float("nan"))
        with self.assertRaisesRegex(ValueError, "global_prune_ratio"):
            UniformAllocation().allocate(scores, True)

    def test_gamma_adaptive_keeps_exact_budget_across_discrete_ties(self) -> None:
        scores = {"left": torch.ones(3), "right": torch.ones(2)}

        rates = GammaAdaptiveAllocation().allocate(scores, 0.4)

        self.assertEqual(rates, {"left": 2 / 3, "right": 0.0})
        self.assertEqual(
            sum(int(scores[name].numel() * rates[name]) for name in scores),
            2,
        )

    def test_weibull_adaptive_honors_exact_budget_and_layer_cap(self) -> None:
        scores = {
            "left": torch.linspace(0.1, 2.0, 20),
            "right": torch.linspace(0.2, 3.0, 20),
        }
        allocator = WeibullAdaptiveAllocation(max_layer_ratio=0.6)

        rates = allocator.allocate(scores, 0.5)

        self.assertTrue(all(rate <= 0.6 for rate in rates.values()))
        self.assertEqual(
            sum(int(scores[name].numel() * rates[name]) for name in scores),
            20,
        )
        with self.assertRaisesRegex(ValueError, "cannot meet"):
            WeibullAdaptiveAllocation(max_layer_ratio=0.4).allocate(scores, 0.5)

    def test_count_ratios_round_trip_through_legacy_integer_conversion(self) -> None:
        scores = {"layer": torch.ones(22)}

        rates = GammaAdaptiveAllocation().allocate(scores, 0.7)

        self.assertEqual(int(scores["layer"].numel() * rates["layer"]), 15)

    def test_weibull_layer_cap_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_layer_ratio"):
            WeibullAdaptiveAllocation(max_layer_ratio=1.1)


if __name__ == "__main__":
    unittest.main()
