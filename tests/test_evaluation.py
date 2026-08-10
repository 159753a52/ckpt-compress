import unittest

import torch

from experiments.lib.evaluation import evaluate


class TestEvaluation(unittest.TestCase):
    def test_empty_regression_batches_return_neutral_metrics(self) -> None:
        model = torch.nn.Linear(1, 1)

        metrics = evaluate(model, [], "reg", device="cpu")

        self.assertEqual(metrics, {"loss": 0.0, "pearson": 0.0})


if __name__ == "__main__":
    unittest.main()
