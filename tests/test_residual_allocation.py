import unittest

from experiments.lib.residual_recovery import (
    bounded_largest_remainder_counts,
    trust_region_counts,
)


class TestResidualAllocation(unittest.TestCase):
    def test_trust_region_counts_preserve_budget_and_order(self) -> None:
        counts = trust_region_counts(
            marginal_losses=[3.0, 1.0, 2.0],
            layer_sizes=[100, 100, 100],
            target=150,
            ratio=0.5,
            trust_radius=0.1,
            max_layer_ratio=0.8,
        )

        self.assertEqual(sum(counts), 150)
        self.assertGreater(counts[1], counts[2])
        self.assertGreater(counts[2], counts[0])
        self.assertTrue(all(40 <= count <= 60 for count in counts))

    def test_trust_region_counts_handle_unequal_layer_sizes(self) -> None:
        sizes = [101, 203, 307]
        target = 305
        counts = trust_region_counts(
            marginal_losses=[0.3, -0.1, 0.2],
            layer_sizes=sizes,
            target=target,
            ratio=target / sum(sizes),
            trust_radius=0.05,
            max_layer_ratio=0.8,
        )

        self.assertEqual(sum(counts), target)
        for count, size in zip(counts, sizes):
            self.assertTrue(0.44 <= count / size <= 0.56)

    def test_flat_marginals_fall_back_to_uniform(self) -> None:
        sizes = [11, 17, 23]
        target = 25
        counts = trust_region_counts(
            marginal_losses=[1.0, 17 / 11, 23 / 11],
            layer_sizes=sizes,
            target=target,
            ratio=target / sum(sizes),
            trust_radius=0.1,
            max_layer_ratio=0.8,
        )

        self.assertEqual(sum(counts), target)
        self.assertEqual(counts, [6, 8, 11])

    def test_flat_marginals_still_respect_small_layer_cap(self) -> None:
        counts = trust_region_counts(
            marginal_losses=[5.0, 11.0],
            layer_sizes=[5, 11],
            target=8,
            ratio=0.5,
            trust_radius=0.1,
            max_layer_ratio=0.55,
        )

        self.assertEqual(counts, [2, 6])

    def test_bounded_rounding_rejects_infeasible_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "Bounds cannot meet"):
            bounded_largest_remainder_counts(
                real_counts=[2.5, 2.5],
                target=7,
                lower_bounds=[1, 1],
                upper_bounds=[3, 3],
            )


if __name__ == "__main__":
    unittest.main()
