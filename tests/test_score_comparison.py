import unittest

import torch

from experiments.lib.importance_compare.score_comparison import (
    compute_gamma,
    compute_mask_iou,
    compute_rank_correlation,
    compute_relative_l2_error,
)


class TestScoreComparison(unittest.TestCase):
    def test_mask_iou_uses_an_exact_global_budget_under_ties(self) -> None:
        scores_ref = {"layer": torch.ones(5)}
        scores_approx = {"layer": torch.ones(5)}

        metrics = compute_mask_iou(scores_ref, scores_approx, 0.4)

        self.assertEqual(metrics, {"__global__": 1.0, "__agreement__": 1.0})

    def test_comparison_inputs_are_aligned_and_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "share at least one"):
            compute_relative_l2_error({"left": torch.ones(2)}, {"right": torch.ones(2)})
        with self.assertRaisesRegex(ValueError, "shapes"):
            compute_mask_iou(
                {"layer": torch.ones(1, 2)},
                {"layer": torch.ones(2)},
                0.5,
            )
        with self.assertRaisesRegex(ValueError, "prune_ratio"):
            compute_mask_iou({"layer": torch.ones(2)}, {"layer": torch.ones(2)}, 1.1)

    def test_constant_rank_correlation_does_not_hide_one_sided_degeneracy(self) -> None:
        same = compute_rank_correlation(
            {"layer": torch.ones(3)},
            {"layer": torch.ones(3)},
        )
        different = compute_rank_correlation(
            {"layer": torch.ones(3)},
            {"layer": torch.full((3,), 2.0)},
        )
        varying = compute_rank_correlation(
            {"layer": torch.ones(3)},
            {"layer": torch.tensor([1.0, 2.0, 3.0])},
        )

        self.assertEqual(same["layer"], {"spearman": 1.0, "pearson": 1.0})
        self.assertEqual(different["layer"], {"spearman": 0.0, "pearson": 0.0})
        self.assertEqual(varying["layer"], {"spearman": 0.0, "pearson": 0.0})

    def test_gamma_matches_vit_constructor_token_counts(self) -> None:
        self.assertEqual(compute_gamma("vit-l-32", 128)["T"], 145)
        self.assertEqual(compute_gamma("vit-b-16", 128)["T"], 197)

        with self.assertRaisesRegex(ValueError, "seq_length"):
            compute_gamma("gpt2-small", 0)


if __name__ == "__main__":
    unittest.main()
