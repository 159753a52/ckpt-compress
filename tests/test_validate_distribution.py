import unittest
from unittest import mock

import numpy as np

from scripts.validate_distribution import fit_and_compare, main


class TestValidateDistribution(unittest.TestCase):
    def test_help_exits_before_fixed_checkpoint_scan(self) -> None:
        with mock.patch("scripts.validate_distribution.analyze_checkpoint") as analyze:
            with self.assertRaises(SystemExit) as raised:
                main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        analyze.assert_not_called()

    def test_all_competitors_produce_finite_ks_evidence(self) -> None:
        values = np.linspace(0.1, 10.0, 200, dtype=np.float64)

        result = fit_and_compare(values, "block.0")

        self.assertEqual(set(result), {"weibull", "gamma_mom", "gamma_mle", "lognormal"})
        for evidence in result.values():
            self.assertTrue(np.isfinite(evidence["ks_d"]))
            self.assertTrue(np.isfinite(evidence["ks_p"]))

    def test_fit_failure_is_not_silently_removed_from_comparison(self) -> None:
        values = np.linspace(0.1, 10.0, 200, dtype=np.float64)

        with (
            mock.patch(
                "scripts.validate_distribution.stats.weibull_min.fit",
                side_effect=ValueError("no convergence"),
            ),
            self.assertRaisesRegex(RuntimeError, "block.3: Weibull MLE fit failed"),
        ):
            fit_and_compare(values, "block.3")

    def test_degenerate_gamma_moments_fail_explicitly(self) -> None:
        values = np.ones(200, dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "block.7: positive samples must have"):
            fit_and_compare(values, "block.7")


if __name__ == "__main__":
    unittest.main()
