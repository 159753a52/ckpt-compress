import itertools
import unittest

import numpy as np
import torch

from experiments.lib.quantile_allocation import (
    _solve_quantile_smooth_counts,
    solve_quantile_smooth_counts,
)
from experiments.lib.residual_allocation import (
    calibrate_quantile_smooth_allocation,
    uniform_counts,
)
from experiments.lib.residual_masks import (
    layer_score_orders,
)


class TestQuantileAllocation(unittest.TestCase):
    def test_legacy_core_alias_points_to_public_solver(self) -> None:
        self.assertIs(_solve_quantile_smooth_counts, solve_quantile_smooth_counts)

    def test_quantile_smooth_matches_small_discrete_optimum(self) -> None:
        layers = [["low"], ["high"], ["middle"]]
        scores = {
            "low": torch.ones(6),
            "high": torch.full((6,), 10.0),
            "middle": torch.full((6,), 2.0),
        }

        counts_by_smoothness, metadata = calibrate_quantile_smooth_allocation(
            layers,
            scores,
            layer_score_orders(layers, scores),
            target=9,
            ratio=0.5,
            max_layer_ratio=0.8,
            trust_radius=0.3,
            smoothness_values=[1.0],
            device="cpu",
        )

        def objective(counts: tuple[int, ...]) -> float:
            proxy_cost = sum(
                float(scores[name][:count].sum()) / 36.0
                for name, count in zip(("low", "high", "middle"), counts)
            )
            rates = [count / 6.0 for count in counts]
            smooth_penalty = 0.5 * sum(
                (right - left) ** 2 for left, right in zip(rates, rates[1:])
            )
            return proxy_cost + smooth_penalty

        feasible = [
            counts
            for counts in itertools.product(range(2, 5), repeat=3)
            if sum(counts) == 9
        ]
        discrete_optimum = min(feasible, key=objective)
        self.assertEqual(counts_by_smoothness[1.0], list(discrete_optimum))
        self.assertEqual(sum(counts_by_smoothness[1.0]), 9)
        self.assertEqual(metadata["model_forward_evaluations"], 0)
        self.assertEqual(metadata["batch_forward_evaluations"], 0)
        self.assertTrue(metadata["solutions"]["1.0"]["converged"])

    def test_quantile_smooth_large_lambda_reduces_rate_variation(self) -> None:
        layers = [["low"], ["high"], ["middle"]]
        scores = {
            "low": torch.ones(6),
            "high": torch.full((6,), 10.0),
            "middle": torch.full((6,), 2.0),
        }

        counts, _ = calibrate_quantile_smooth_allocation(
            layers,
            scores,
            layer_score_orders(layers, scores),
            target=9,
            ratio=0.5,
            max_layer_ratio=0.8,
            trust_radius=0.3,
            smoothness_values=[0.01, 10.0],
            device="cpu",
        )

        low_rates = [count / 6 for count in counts[0.01]]
        high_rates = [count / 6 for count in counts[10.0]]
        low_variation = sum(
            (right - left) ** 2 for left, right in zip(low_rates, low_rates[1:])
        )
        high_variation = sum(
            (right - left) ** 2 for left, right in zip(high_rates, high_rates[1:])
        )
        self.assertLess(high_variation, low_variation)
        self.assertEqual(counts[10.0], [3, 3, 3])

    def test_relative_quantile_cost_is_invariant_to_layer_scaling(self) -> None:
        layers = [["low"], ["high"], ["middle"]]
        scores = {
            "low": torch.tensor([1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
            "high": torch.tensor([2.0, 2.5, 3.0, 6.0, 7.0, 8.0]),
            "middle": torch.tensor([0.5, 1.0, 2.5, 3.5, 5.0, 9.0]),
        }
        scaled_scores = {
            "low": 7.0 * scores["low"],
            "high": 0.2 * scores["high"],
            "middle": 3.0 * scores["middle"],
        }

        original, original_metadata = calibrate_quantile_smooth_allocation(
            layers,
            scores,
            layer_score_orders(layers, scores),
            target=9,
            ratio=0.5,
            max_layer_ratio=0.8,
            trust_radius=0.3,
            smoothness_values=[0.1],
            device="cpu",
            normalization="layer_uniform_cost",
        )
        scaled, scaled_metadata = calibrate_quantile_smooth_allocation(
            layers,
            scaled_scores,
            layer_score_orders(layers, scaled_scores),
            target=9,
            ratio=0.5,
            max_layer_ratio=0.8,
            trust_radius=0.3,
            smoothness_values=[0.1],
            device="cpu",
            normalization="layer_uniform_cost",
        )

        self.assertEqual(original[0.1], scaled[0.1])
        original_solution = original_metadata["solutions"]["0.1"]
        scaled_solution = scaled_metadata["solutions"]["0.1"]
        for original_rate, scaled_rate in zip(
            original_solution["continuous_rates"],
            scaled_solution["continuous_rates"],
        ):
            self.assertAlmostEqual(original_rate, scaled_rate, places=12)
        self.assertAlmostEqual(
            original_solution["objective"], scaled_solution["objective"], places=12
        )

    def test_relative_quantile_cost_rejects_zero_uniform_cost(self) -> None:
        layers = [["zero"], ["positive"]]
        scores = {
            "zero": torch.zeros(4),
            "positive": torch.ones(4),
        }

        with self.assertRaisesRegex(ValueError, "nonzero uniform proxy costs"):
            calibrate_quantile_smooth_allocation(
                layers,
                scores,
                layer_score_orders(layers, scores),
                target=4,
                ratio=0.5,
                max_layer_ratio=0.75,
                trust_radius=0.25,
                smoothness_values=[0.1],
                device="cpu",
                normalization="layer_uniform_cost",
            )

    def test_quantile_facade_preserves_core_schema(self) -> None:
        layers = [["left"], ["center"], ["right"]]
        scores = {
            "left": torch.tensor([0.5, 1.0, 2.0, 4.0, 7.0, 8.0]),
            "center": torch.tensor([0.2, 0.8, 3.0, 3.5, 5.0, 9.0]),
            "right": torch.tensor([0.1, 1.5, 2.5, 4.5, 6.0, 10.0]),
        }
        orders = layer_score_orders(layers, scores)
        facade_counts, facade_metadata = calibrate_quantile_smooth_allocation(
            layers,
            scores,
            orders,
            target=9,
            ratio=0.5,
            max_layer_ratio=0.8,
            trust_radius=0.3,
            smoothness_values=[0.1, 1.0],
            device="cpu",
            normalization="layer_uniform_cost",
        )

        layer_sizes = np.asarray([6, 6, 6], dtype=np.int64)
        uniform_layer_counts = uniform_counts(layer_sizes.tolist(), 9, 0.5)
        sorted_scores = [
            scores[name].float().numpy()[order.numpy()]
            for name, order in zip(("left", "center", "right"), orders)
        ]
        core_counts, core_metadata = solve_quantile_smooth_counts(
            sorted_scores=sorted_scores,
            layer_sizes=layer_sizes,
            uniform_layer_counts=uniform_layer_counts,
            lower_bounds=np.asarray([2, 2, 2], dtype=np.int64),
            upper_bounds=np.asarray([4, 4, 4], dtype=np.int64),
            target=9,
            trust_radius=0.3,
            smoothness_values=[0.1, 1.0],
            normalization="layer_uniform_cost",
            max_iterations=300,
            tolerance=1e-9,
        )

        self.assertEqual(facade_counts, core_counts)
        self.assertEqual(
            set(facade_metadata),
            {
                "seconds",
                "score_materialization_seconds",
                "solve_seconds",
                "smoothness_values",
                "trust_radius",
                "difference_order",
                "normalization",
                "score_scale",
                "score_normalizers",
                "uniform_proxy_costs",
                "target_quantiles",
                "model_forward_evaluations",
                "batch_forward_evaluations",
                "uses_validation_data",
                "solutions",
            },
        )
        for key, value in core_metadata.items():
            if key != "solve_seconds":
                self.assertEqual(facade_metadata[key], value)


if __name__ == "__main__":
    unittest.main()
