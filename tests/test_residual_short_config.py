import argparse
import unittest
from pathlib import Path

import experiments.lib.residual_recovery as residual_recovery
import experiments.lib.residual_short_config as short_config


class TestResidualShortConfig(unittest.TestCase):
    def test_parse_args_defaults_remain_stable(self) -> None:
        args = short_config.parse_args([])

        self.assertEqual(args.prune_ratio, 0.30)
        self.assertEqual(args.max_layer_ratio, 0.80)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.seq_length, 128)
        self.assertEqual(args.eval_batches, 20)
        self.assertEqual(args.hvp_batches, 1)
        self.assertEqual(args.train_batch_offset, 0)
        self.assertEqual(args.continuation_steps, 0)
        self.assertEqual(args.continuation_lr, 5e-5)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.device, "cuda")
        self.assertEqual(
            args.output_dir,
            short_config.ROOT / "results/diagnostics/v100_allocation_gate",
        )

    def test_parse_args_preserves_path_types_and_explicit_values(self) -> None:
        args = short_config.parse_args(
            [
                "--reference-checkpoint",
                "reference.pt",
                "--current-checkpoint",
                "current.pt",
                "--data-dir",
                "data",
                "--output-dir",
                "results",
                "--device",
                "cpu",
            ]
        )

        self.assertEqual(args.reference_checkpoint, Path("reference.pt"))
        self.assertEqual(args.current_checkpoint, Path("current.pt"))
        self.assertEqual(args.data_dir, Path("data"))
        self.assertEqual(args.output_dir, Path("results"))
        self.assertEqual(args.device, "cpu")

    def test_validation_preserves_error_messages_and_priority(self) -> None:
        base = vars(short_config.parse_args([]))
        cases = [
            (
                {"prune_ratio": 0.0, "hvp_batches": 0},
                "--prune-ratio must be in (0, 1)",
            ),
            (
                {"hvp_batches": 0},
                "HVP batches must be positive and step/offset counts non-negative",
            ),
            (
                {"continuation_steps": -1},
                "HVP batches must be positive and step/offset counts non-negative",
            ),
            (
                {"train_batch_offset": -1},
                "HVP batches must be positive and step/offset counts non-negative",
            ),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                args = argparse.Namespace(**{**base, **overrides})
                with self.assertRaises(ValueError) as raised:
                    short_config.validate_short_config(args)
                self.assertEqual(str(raised.exception), message)

        short_config.validate_short_config(argparse.Namespace(**base))

    def test_recovery_facade_reexports_config_functions(self) -> None:
        self.assertIs(residual_recovery.parse_args, short_config.parse_args)
        self.assertIs(
            residual_recovery.validate_short_config,
            short_config.validate_short_config,
        )


if __name__ == "__main__":
    unittest.main()
