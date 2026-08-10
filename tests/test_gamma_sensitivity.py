import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from experiments.lib.gamma_sensitivity import (
    fit_gamma_mom_problem,
    solve_gamma_mom_rates,
)


class TestGammaSensitivity(unittest.TestCase):
    def test_fit_tracks_valid_and_fallback_layers(self) -> None:
        scores = {
            "valid": torch.arange(1.0, 11.0),
            "few": torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            "constant": torch.ones(10),
        }

        problem = fit_gamma_mom_problem(scores)

        self.assertEqual(problem.total_parameters, 31)
        self.assertEqual(problem.layers[0].positive_count, 10)
        self.assertTrue(problem.layers[0].valid)
        self.assertEqual(problem.layers[1].fallback_reason, "insufficient_positive")
        self.assertEqual(problem.layers[2].fallback_reason, "degenerate_moments")

    def test_solver_preserves_budget_with_fallback_layers(self) -> None:
        scores = {
            "small": torch.arange(1.0, 11.0),
            "wide": torch.arange(1.0, 11.0).square(),
            "few": torch.tensor([1.0, 2.0, 3.0]),
            "constant": torch.ones(10),
        }
        problem = fit_gamma_mom_problem(scores)

        rates, metadata = solve_gamma_mom_rates(problem, 0.3, xtol=1e-10)

        weighted = sum(
            layer.size * rates[layer.name] for layer in problem.layers
        ) / problem.total_parameters
        self.assertAlmostEqual(weighted, 0.3, places=8)
        self.assertEqual(rates["few"], 0.3)
        self.assertEqual(rates["constant"], 0.3)
        self.assertEqual(metadata["status"], "converged")
        self.assertLess(metadata["budget_residual"], 1e-8)

    def test_empty_boundaries_and_failed_bracket_are_explicit(self) -> None:
        empty = fit_gamma_mom_problem({})
        rates, metadata = solve_gamma_mom_rates(empty, 0.3)
        self.assertEqual(rates, {})
        self.assertEqual(metadata["status"], "empty")

        boundary = fit_gamma_mom_problem({"x": torch.ones(10)})
        zero, zero_metadata = solve_gamma_mom_rates(boundary, 0.0)
        self.assertEqual(zero, {"x": 0.0})
        self.assertEqual(zero_metadata["status"], "boundary")

        with mock.patch(
            "experiments.lib.gamma_sensitivity.stats.gamma.cdf",
            return_value=float("nan"),
        ):
            fallback, fallback_metadata = solve_gamma_mom_rates(
                fit_gamma_mom_problem({"x": torch.arange(1.0, 11.0)}),
                0.3,
                max_bracket_expansions=0,
            )
        self.assertEqual(fallback, {"x": 0.3})
        self.assertEqual(fallback_metadata["status"], "uniform_fallback")
        self.assertEqual(fallback_metadata["fallback"], "root_not_bracketed")

    def test_tolerance_sweep_fits_once_and_preserves_row_schema(self) -> None:
        import experiments.scripts.run_sensitivity_analysis as sensitivity

        args = SimpleNamespace(
            sweep_tol="1e-3,1e-6",
            device="cpu",
            alpha=0.5,
            hvp_batches=1,
            chunk_size=10,
            prune_ratio=0.3,
        )
        scores = {"left": torch.arange(1.0, 11.0)}
        model = torch.nn.Linear(1, 1)

        with mock.patch.object(
            sensitivity,
            "compute_scores_by_method",
            return_value={"second-order-hvp": scores},
        ), mock.patch.object(
            sensitivity,
            "apply_pruning",
            return_value=(model, {}, 0.3),
        ), mock.patch.object(
            sensitivity,
            "evaluate",
            return_value={"perplexity": 2.0},
        ), mock.patch.object(
            sensitivity,
            "fit_gamma_mom_problem",
            wraps=sensitivity.fit_gamma_mom_problem,
        ) as fit:
            rows = sensitivity.sweep_bisection_tol(
                args,
                model,
                None,
                None,
                "lm",
            )

        self.assertEqual(fit.call_count, 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            set(rows[0]),
            {"bisection_tol", "alloc_time_s", "actual_ratio", "perplexity"},
        )


if __name__ == "__main__":
    unittest.main()
