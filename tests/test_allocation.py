import unittest

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

    def test_empty_layers_do_not_divide_by_zero(self) -> None:
        scores = {
            "empty": torch.empty(0),
            "active": torch.tensor([1.0, 2.0]),
        }

        rates = GlobalTopKAllocation().allocate(scores, 0.5)

        self.assertEqual(rates["empty"], 0.0)
        self.assertEqual(rates["active"], 0.5)

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

    def test_weibull_layer_cap_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_layer_ratio"):
            WeibullAdaptiveAllocation(max_layer_ratio=1.1)


if __name__ == "__main__":
    unittest.main()
