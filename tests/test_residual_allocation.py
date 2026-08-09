import unittest

import torch

from experiments.lib.residual_recovery import (
    apply_mask_from_device_states,
    bounded_largest_remainder_counts,
    budget_tangent_dct_directions,
    cache_mask_states_on_device,
    directional_layer_counts,
    layer_masks,
    layer_score_orders,
    reconstruct_directional_gradient,
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

    def test_cached_score_orders_match_direct_masks_with_ties(self) -> None:
        layers = [["left", "right"], ["last"]]
        scores = {
            "left": torch.tensor([3.0, 1.0, 1.0]),
            "right": torch.tensor([2.0, 1.0]),
            "last": torch.tensor([4.0, 2.0, 2.0, 3.0]),
        }
        counts = [3, 2]

        direct = layer_masks(layers, scores, counts)
        cached = layer_masks(
            layers,
            scores,
            counts,
            layer_score_orders(layers, scores),
        )

        self.assertEqual(direct.keys(), cached.keys())
        for name in direct:
            self.assertTrue(torch.equal(direct[name], cached[name]))

    def test_device_state_cache_applies_mask_exactly(self) -> None:
        model = torch.nn.Linear(2, 2, bias=False)
        current = {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        reference = {"weight": torch.tensor([[10.0, 20.0], [30.0, 40.0]])}
        current_device, reference_device = cache_mask_states_on_device(
            current, reference, ["weight"], "cpu"
        )

        apply_mask_from_device_states(
            model,
            current_device,
            reference_device,
            {"weight": torch.tensor([[True, False], [False, True]])},
            "cpu",
        )

        expected = torch.tensor([[1.0, 20.0], [30.0, 4.0]])
        self.assertTrue(torch.equal(model.weight, expected))

if __name__ == "__main__":
    unittest.main()
