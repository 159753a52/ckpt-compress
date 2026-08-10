import unittest
from unittest import mock

import torch
import torch.nn as nn

import experiments.lib.residual_short_methods as short_methods
from experiments.lib.residual_protocol import SHORT_GATE_METHODS


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Parameter(torch.zeros(2, 2))
        self.second = nn.Parameter(torch.zeros(4))


class TestResidualShortMethods(unittest.TestCase):
    def make_scope(self) -> short_methods.ShortResidualScope:
        delta = {
            "a": torch.tensor([1.0, -2.0, 3.0, -4.0]),
            "b": torch.tensor([-1.5, 2.5, -3.5, 4.5]),
        }
        return short_methods.ShortResidualScope(
            layers=[["a"], ["b"]],
            eligible_names=["a", "b"],
            layer_sizes=[4, 4],
            eligible_count=8,
            model_count=10,
            target_pruned=4,
            delta=delta,
            magnitude_scores={name: value.abs() for name, value in delta.items()},
        )

    def test_scope_preserves_residual_and_result_schema(self) -> None:
        model = TinyModel()
        current = {
            "first": torch.tensor([[1, -2], [3, -4]], dtype=torch.int64),
            "second": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        }
        reference = {
            "first": torch.zeros(2, 2),
            "second": torch.ones(4),
        }
        with mock.patch.object(
            short_methods,
            "eligible_layers",
            return_value=[["first"], ["second"]],
        ):
            scope = short_methods.build_short_residual_scope(
                model,
                current,
                reference,
                prune_ratio=0.375,
            )

        self.assertEqual(scope.layer_sizes, [4, 4])
        self.assertEqual(scope.eligible_count, 8)
        self.assertEqual(scope.model_count, 8)
        self.assertEqual(scope.target_pruned, 3)
        self.assertEqual(scope.delta["first"].dtype, torch.float32)
        torch.testing.assert_close(scope.magnitude_scores["first"], scope.delta["first"].abs())
        result = scope.to_result_dict(0.375)
        self.assertEqual(
            list(result),
            [
                "transformer_layers",
                "eligible_tensors",
                "eligible_parameters",
                "whole_model_parameters",
                "target_pruned",
                "target_eligible_sparsity",
                "target_whole_model_sparsity",
                "layer_sizes",
                "eligible_names",
            ],
        )
        self.assertEqual(result["target_whole_model_sparsity"], 3 / 8)

    def test_four_masks_preserve_order_budget_and_diagnostic_schema(self) -> None:
        scope = self.make_scope()
        taylor_scores = {
            "a": torch.tensor([0.1, 0.2, 2.0, 3.0]),
            "b": torch.tensor([0.3, 0.4, 4.0, 5.0]),
        }

        masks, allocation = short_methods.build_short_gate_masks(
            scope,
            taylor_scores,
            prune_ratio=0.5,
            max_layer_ratio=0.75,
        )
        diagnostics = short_methods.diagnose_short_gate_masks(
            scope,
            masks,
            taylor_scores,
        )

        self.assertEqual(tuple(masks), SHORT_GATE_METHODS)
        self.assertEqual(tuple(diagnostics), SHORT_GATE_METHODS)
        self.assertEqual(
            list(allocation),
            [
                "seconds",
                "weibull_fits",
                "uniform_layer_counts",
                "weibull_layer_counts",
                "weibull",
            ],
        )
        self.assertEqual(allocation["uniform_layer_counts"], [2, 2])
        self.assertEqual(sum(allocation["weibull_layer_counts"]), 4)
        for index, fit in enumerate(allocation["weibull_fits"]):
            self.assertEqual(fit["layer"], index)
        for metrics in diagnostics.values():
            self.assertEqual(metrics["pruned"], 4)
            self.assertEqual(metrics["eligible_sparsity"], 0.5)
            self.assertEqual(metrics["whole_model_sparsity"], 0.4)
            self.assertEqual(len(metrics["layer_rates"]), 2)
            self.assertEqual(
                set(metrics["overlap_with_exact"]),
                {"pruned_jaccard", "mask_disagreements"},
            )

    def test_diagnostics_reject_method_order_drift(self) -> None:
        scope = self.make_scope()
        with self.assertRaisesRegex(
            ValueError,
            "^Short-gate masks must follow the result method protocol$",
        ):
            short_methods.diagnose_short_gate_masks(scope, {}, {})


if __name__ == "__main__":
    unittest.main()
