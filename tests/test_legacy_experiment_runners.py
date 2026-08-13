import copy
import unittest
from unittest import mock

import numpy as np
import torch

import experiments.scripts.run_fault_tolerant_training as fault_runner
import experiments.scripts.run_fault_tolerant_training_clean as clean_fault_runner
import experiments.scripts.run_ablation_study as ablation_runner
import experiments.scripts.run_gamma_validation as gamma_runner
import experiments.scripts.run_method_comparison as comparison_runner


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, -1.0]]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.weight.transpose(0, 1)


class TestLegacyExperimentRunners(unittest.TestCase):
    def test_importance_gradient_collection_does_not_train_model(self) -> None:
        batch = {"images": torch.tensor([[2.0, -1.0]]), "labels": torch.tensor([0])}
        for runner in (fault_runner, clean_fault_runner):
            with self.subTest(runner=runner.__name__):
                model = TinyModel()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
                before_model = copy.deepcopy(model.state_dict())
                before_optimizer = copy.deepcopy(optimizer.state_dict())

                weights, gradients = runner.collect_gradients_inline(
                    model,
                    optimizer,
                    [batch],
                    "cpu",
                    num_steps=2,
                    task_type="cv",
                )

                self.assertTrue(gradients)
                self.assertEqual(before_optimizer, optimizer.state_dict())
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, before_model[name]), name)
                    self.assertTrue(torch.equal(weights[name], before_model[name]), name)

    def test_segment_lengths_preserve_exact_training_budget(self) -> None:
        for runner in (fault_runner, clean_fault_runner):
            with self.subTest(runner=runner.__name__):
                lengths = runner._segment_lengths(10, 2)
                self.assertEqual(lengths, [4, 3, 3])
                self.assertEqual(sum(lengths), 10)
                with self.assertRaisesRegex(ValueError, "one step per"):
                    runner._segment_lengths(2, 2)

    def test_fault_runner_masks_prune_exactly_across_score_ties(self) -> None:
        for runner in (fault_runner, clean_fault_runner):
            with self.subTest(runner=runner.__name__):
                model = TinyModel()
                masks = runner.apply_pruning_to_model(
                    model,
                    {"weight": torch.ones_like(model.weight)},
                    {"weight": 0.5},
                    "cpu",
                )
                self.assertEqual(int((masks["weight"] == 0).sum().item()), 1)

    def test_history_records_only_real_task_metric(self) -> None:
        for runner in (fault_runner, clean_fault_runner):
            with self.subTest(runner=runner.__name__):
                lm = runner._history_record({"loss": 1.0, "perplexity": 2.0}, "lm", 3, 0)
                self.assertEqual(lm, {"step": 3, "loss": 1.0, "segment": 0, "perplexity": 2.0})
                cls = runner._history_record({"loss": 1.0, "accuracy": 0.5}, "cls", 4, 1)
                self.assertEqual(cls, {"step": 4, "loss": 1.0, "segment": 1, "accuracy": 0.5})
                with self.assertRaisesRegex(KeyError, "perplexity"):
                    runner._history_record({"loss": 1.0}, "lm", 1, 0)

    def test_distribution_fit_failure_is_explicit_not_a_zero_pvalue(self) -> None:
        scores = {"layer": torch.linspace(0.1, 10.0, 100)}
        with mock.patch.object(
            gamma_runner,
            "_fit_gamma",
            side_effect=ValueError("synthetic fit failure"),
        ):
            [result] = gamma_runner.fit_and_test_distributions(scores)

        self.assertIsNone(result["gamma_pvalue"])
        self.assertIsNone(result["gamma_ks"])
        self.assertIn("synthetic fit failure", result["gamma_error"])
        self.assertTrue(np.isfinite(result["mean"]))

    def test_loss_increase_percentage_rejects_invalid_baselines(self) -> None:
        runners = (comparison_runner, gamma_runner, ablation_runner)
        for runner in runners:
            with self.subTest(runner=runner.__name__, baseline="zero"):
                with self.assertRaisesRegex(ValueError, "zero baseline loss"):
                    runner._loss_increase_pct(0.0, 0.0)
            for baseline in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(runner=runner.__name__, baseline=baseline):
                    with self.assertRaisesRegex(ValueError, "baseline loss must be finite"):
                        runner._loss_increase_pct(baseline, 1.0)
            self.assertAlmostEqual(runner._loss_increase_pct(2.0, 2.2), 10.0)

    def test_gamma_summary_rejects_zero_fitted_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one successful fit"):
            gamma_runner._gamma_fit_summary(
                [{"gamma_pvalue": None}, {"other_metric": 0.1}]
            )

        self.assertEqual(
            gamma_runner._gamma_fit_summary(
                [{"gamma_pvalue": 0.1}, {"gamma_pvalue": 0.01}, {"gamma_pvalue": None}]
            ),
            (1, 2, 3),
        )


if __name__ == "__main__":
    unittest.main()
