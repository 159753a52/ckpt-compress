import unittest

import torch

import experiments.lib.residual_allocation as residual_allocation
import experiments.lib.residual_budget as residual_budget
import experiments.lib.residual_spectral as residual_spectral
import experiments.lib.residual_weibull as residual_weibull
from experiments.lib.residual_allocation import (
    bounded_largest_remainder_counts,
    budget_tangent_dct_directions,
    directional_layer_counts,
    fit_weibull_mom,
    largest_remainder_counts,
    reconstruct_directional_gradient,
    trust_region_counts,
    weibull_counts,
)


class TestResidualAllocation(unittest.TestCase):
    def test_budget_facade_reexports_owner_functions(self) -> None:
        for name in residual_budget.__all__:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(residual_allocation, name),
                    getattr(residual_budget, name),
                )

    def test_spectral_facade_reexports_owner_functions(self) -> None:
        for name in residual_spectral.__all__:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(residual_allocation, name),
                    getattr(residual_spectral, name),
                )

    def test_weibull_facade_reexports_owner_functions(self) -> None:
        for name in residual_weibull.__all__:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(residual_allocation, name),
                    getattr(residual_weibull, name),
                )

    def test_weibull_counts_preserve_exact_budget_and_caps(self) -> None:
        fits = [
            fit_weibull_mom(torch.tensor([0.1, 0.2, 0.4, 0.8, 1.6, 3.2])),
            fit_weibull_mom(torch.tensor([0.5, 0.6, 0.8, 1.1, 1.5, 2.0])),
        ]

        counts, metadata = weibull_counts(
            fits,
            layer_sizes=[6, 6],
            target=6,
            ratio=0.5,
            max_layer_ratio=0.8,
        )

        self.assertTrue(all(fit["valid"] for fit in fits))
        self.assertEqual(sum(counts), 6)
        self.assertTrue(all(count <= 4 for count in counts))
        self.assertIsNone(metadata["fallback"])

    def test_weibull_counts_fall_back_when_all_fits_are_invalid(self) -> None:
        fits = [
            fit_weibull_mom(torch.zeros(5)),
            fit_weibull_mom(torch.zeros(7)),
        ]

        counts, metadata = weibull_counts(
            fits,
            layer_sizes=[5, 7],
            target=6,
            ratio=0.5,
            max_layer_ratio=0.8,
        )

        self.assertEqual(counts, [3, 3])
        self.assertEqual(metadata["fallback"], "all Weibull fits invalid")

    def test_weibull_fallback_respects_small_layer_cap(self) -> None:
        fits = [
            fit_weibull_mom(torch.zeros(1)),
            fit_weibull_mom(torch.zeros(3)),
        ]

        counts, metadata = weibull_counts(
            fits,
            layer_sizes=[1, 3],
            target=2,
            ratio=0.5,
            max_layer_ratio=0.75,
        )

        self.assertEqual(counts, [0, 2])
        self.assertEqual(metadata["fallback"], "all Weibull fits invalid")

    def test_weibull_counts_reject_misaligned_fit_list(self) -> None:
        valid_fit = fit_weibull_mom(torch.tensor([0.1, 0.2, 0.4, 0.8]))

        for fits, sizes in (([valid_fit], [4, 4]), ([valid_fit, valid_fit], [4])):
            with self.subTest(fit_count=len(fits), layer_count=len(sizes)):
                with self.assertRaisesRegex(ValueError, "same length"):
                    weibull_counts(
                        fits,
                        layer_sizes=sizes,
                        target=2,
                        ratio=0.5,
                        max_layer_ratio=0.75,
                    )

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

    def test_budget_rounding_rejects_truncated_or_non_finite_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            largest_remainder_counts([1.0, 2.0], target=1, capacities=[2])
        with self.assertRaisesRegex(ValueError, "finite"):
            largest_remainder_counts([float("nan")], target=1, capacities=[1])

    def test_empty_weibull_fit_and_zero_sized_trust_layer_are_explicit(self) -> None:
        empty_fit = residual_weibull.fit_weibull_mom(torch.empty(0))
        self.assertFalse(empty_fit["valid"])
        self.assertEqual(empty_fit["reason"], "empty values")

        with self.assertRaisesRegex(ValueError, "positive"):
            trust_region_counts(
                marginal_losses=[1.0, 2.0],
                layer_sizes=[0, 4],
                target=2,
                ratio=0.5,
                trust_radius=0.1,
                max_layer_ratio=0.8,
            )

    def test_spectral_directions_are_budget_tangent(self) -> None:
        sizes = [11, 17, 29, 43]
        directions = budget_tangent_dct_directions(sizes, rank=3)

        self.assertEqual(len(directions), 3)
        for direction in directions:
            self.assertAlmostEqual(
                sum(size * value for size, value in zip(sizes, direction)),
                0.0,
                places=10,
            )
            self.assertAlmostEqual(max(abs(value) for value in direction), 1.0)

    def test_directional_counts_preserve_exact_budget(self) -> None:
        sizes = [101, 203, 307, 409]
        target = 510
        direction = budget_tangent_dct_directions(sizes, rank=1)[0]

        plus = directional_layer_counts(
            direction, sizes, target, 0.5, 0.025, 0.8, 1.0
        )
        minus = directional_layer_counts(
            direction, sizes, target, 0.5, 0.025, 0.8, -1.0
        )

        self.assertEqual(sum(plus), target)
        self.assertEqual(sum(minus), target)
        self.assertNotEqual(plus, minus)
        self.assertTrue(
            all(0 <= count <= int(0.8 * size) for count, size in zip(plus, sizes))
        )
        self.assertTrue(
            all(0 <= count <= int(0.8 * size) for count, size in zip(minus, sizes))
        )

    def test_directional_reconstruction_matches_measurements(self) -> None:
        design = [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ]
        true_gradient = [3.0, -1.0, 2.0, 0.5]
        responses = [
            sum(value * gradient for value, gradient in zip(row, true_gradient))
            for row in design
        ]

        reconstructed, metadata = reconstruct_directional_gradient(design, responses)

        for row, response in zip(design, responses):
            predicted = sum(value * gradient for value, gradient in zip(row, reconstructed))
            self.assertAlmostEqual(predicted, response, places=12)
        self.assertEqual(metadata["rank"], 3)
        self.assertLess(metadata["response_residual_l2"], 1e-12)

    def test_directional_reconstruction_rejects_degenerate_design(self) -> None:
        with self.assertRaisesRegex(ValueError, "full row rank"):
            reconstruct_directional_gradient(
                [[1.0, -1.0], [2.0, -2.0]],
                [3.0, 6.0],
            )

if __name__ == "__main__":
    unittest.main()
