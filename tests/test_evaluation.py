import unittest

import torch

from experiments.lib.evaluation import compute_quality_drop, evaluate


class TestEvaluation(unittest.TestCase):
    def test_empty_regression_batches_return_neutral_metrics(self) -> None:
        model = torch.nn.Linear(1, 1)

        metrics = evaluate(model, [], "reg", device="cpu")

        self.assertEqual(metrics, {"loss": 0.0, "pearson": 0.0})

    def test_quality_drop_preserves_normalized_direction(self) -> None:
        self.assertAlmostEqual(
            compute_quality_drop({"loss": 2.0}, {"loss": 2.2}, "lm"),
            10.0,
        )
        self.assertAlmostEqual(
            compute_quality_drop({"accuracy": 0.8}, {"accuracy": 0.76}, "cls"),
            5.0,
        )
        self.assertAlmostEqual(
            compute_quality_drop({"pearson": -0.5}, {"pearson": -0.4}, "reg"),
            -20.0,
        )

    def test_quality_drop_zero_baseline_is_explicit(self) -> None:
        self.assertEqual(
            compute_quality_drop({"loss": 0.0}, {"loss": 0.0}, "lm"),
            0.0,
        )
        self.assertEqual(
            compute_quality_drop({"loss": 0.0}, {"loss": 0.1}, "lm"),
            float("inf"),
        )
        self.assertEqual(
            compute_quality_drop({"accuracy": 0.0}, {"accuracy": 0.1}, "cls"),
            float("-inf"),
        )

    def test_quality_drop_rejects_missing_or_unknown_contracts(self) -> None:
        with self.assertRaisesRegex(KeyError, "baseline metrics must contain 'loss'"):
            compute_quality_drop({}, {"loss": 1.0}, "lm")
        with self.assertRaisesRegex(ValueError, "Unknown task_type"):
            compute_quality_drop({"loss": 1.0}, {"loss": 1.0}, "unknown")


if __name__ == "__main__":
    unittest.main()
