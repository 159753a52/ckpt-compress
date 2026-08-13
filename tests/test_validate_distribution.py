import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.validate_distribution import fit_and_compare, main


class TestValidateDistribution(unittest.TestCase):
    _CHECKPOINTS = (
        "checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt",
        "checkpoints/bert_large_sst2_1000steps/checkpoint_step_1000_final.pt",
        "checkpoints/bert_large_mnli_1000steps/checkpoint_step_1000_final.pt",
    )

    def test_help_exits_before_fixed_checkpoint_scan(self) -> None:
        with mock.patch("scripts.validate_distribution.analyze_checkpoint") as analyze:
            with self.assertRaises(SystemExit) as raised:
                main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        analyze.assert_not_called()

    def test_all_declared_checkpoints_missing_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch("scripts.validate_distribution.ROOT", Path(temp_dir)):
                with mock.patch("scripts.validate_distribution.analyze_checkpoint") as analyze:
                    self.assertEqual(main([]), 1)

        analyze.assert_not_called()

    def test_empty_analysis_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for checkpoint in self._CHECKPOINTS:
                path = root / checkpoint
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"checkpoint")
            with mock.patch("scripts.validate_distribution.ROOT", root):
                with mock.patch(
                    "scripts.validate_distribution.analyze_checkpoint",
                    return_value=(
                        {},
                        {"weibull_wins": 0, "gamma_wins": 0, "lognorm_wins": 0, "total": 0},
                    ),
                ) as analyze:
                    self.assertEqual(main([]), 1)

        self.assertEqual(analyze.call_count, len(self._CHECKPOINTS))

    def test_any_missing_declared_checkpoint_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing = root / self._CHECKPOINTS[0]
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"checkpoint")
            with mock.patch("scripts.validate_distribution.ROOT", root):
                with mock.patch(
                    "scripts.validate_distribution.analyze_checkpoint",
                    return_value=(
                        {},
                        {"weibull_wins": 0, "gamma_wins": 0, "lognorm_wins": 0, "total": 0},
                    ),
                ) as analyze:
                    self.assertEqual(main([]), 1)

        analyze.assert_called_once_with(str(existing), "GPT-2 Medium")

    def test_all_declared_checkpoints_analyze_successfully_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for checkpoint in self._CHECKPOINTS:
                path = root / checkpoint
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"checkpoint")
            with mock.patch("scripts.validate_distribution.ROOT", root):
                with mock.patch(
                    "scripts.validate_distribution.analyze_checkpoint",
                    return_value=(
                        {},
                        {"weibull_wins": 0, "gamma_wins": 0, "lognorm_wins": 0, "total": 1},
                    ),
                ) as analyze:
                    self.assertEqual(main([]), 0)

        self.assertEqual(analyze.call_count, len(self._CHECKPOINTS))

    def test_analysis_exception_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for checkpoint in self._CHECKPOINTS:
                path = root / checkpoint
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"checkpoint")
            with mock.patch("scripts.validate_distribution.ROOT", root):
                with mock.patch(
                    "scripts.validate_distribution.analyze_checkpoint",
                    side_effect=RuntimeError("fit failed"),
                ) as analyze:
                    self.assertEqual(main([]), 1)

        self.assertEqual(analyze.call_count, len(self._CHECKPOINTS))

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
