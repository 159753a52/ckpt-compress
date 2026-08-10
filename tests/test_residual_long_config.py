import argparse
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import experiments.lib.residual_long_config as long_config
import experiments.lib.residual_method_assembly as method_assembly
import experiments.scripts.run_residual_recovery_long as long_runner


class TestResidualLongConfig(unittest.TestCase):
    def namespace(self, **overrides) -> argparse.Namespace:
        values = {
            "checkpoint": Path("checkpoint.pt"),
            "data_dir": Path("data"),
            "output_dir": Path("results"),
            "seeds": "42,43,44",
            "total_steps": 10,
            "recovery_step": 5,
            "prune_ratio": 0.5,
            "max_layer_ratio": 0.8,
            "batch_size": 2,
            "seq_length": 128,
            "hvp_batches": 2,
            "allocation_probe_batches": 1,
            "allocation_selection_batches": 1,
            "allocation_probe_radius": 0.1,
            "allocation_trust_radii": "0.025,0.05,0.1",
            "spectral_ranks": "",
            "spectral_probe_radius": 0.1,
            "spectral_trust_radius": 0.1,
            "quantile_smoothness_values": "",
            "quantile_trust_radius": 0.1,
            "quantile_cost_normalization": "global",
            "eval_batches": 4,
            "train_pool_batches": 14,
            "learning_rate": 5e-5,
            "device": "cpu",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def base_config(self) -> long_config.LongExperimentConfig:
        return long_config.normalize_long_config(self.namespace())

    def with_methods(self, config, **changes):
        return replace(config, methods=replace(config.methods, **changes))

    def test_normalize_preserves_collections_and_flat_result_schema(self) -> None:
        args = self.namespace(
            seeds="43,42,43",
            allocation_trust_radii="0.1,0.05,0.1",
            spectral_ranks="2,1,2",
            quantile_smoothness_values="0.2,0.1,0.2",
            quantile_cost_normalization="layer_uniform_cost",
        )

        config = long_config.normalize_long_config(args)

        self.assertEqual(config.seeds, (43, 42, 43))
        self.assertEqual(config.methods.trust_radii, (0.1, 0.05, 0.1))
        self.assertEqual(config.methods.spectral_ranks, (1, 2))
        self.assertEqual(config.methods.quantile_smoothness_values, (0.1, 0.2))
        self.assertIs(
            method_assembly.AdaptiveMethodConfig,
            long_config.AdaptiveMethodConfig,
        )

        result = config.to_result_dict()
        expected = {
            "checkpoint": str(args.checkpoint.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "seeds": [43, 42, 43],
            "total_steps": 10,
            "recovery_step": 5,
            "prune_ratio": 0.5,
            "max_layer_ratio": 0.8,
            "batch_size": 2,
            "seq_length": 128,
            "hvp_batches": 2,
            "allocation_probe_batches": 1,
            "allocation_selection_batches": 1,
            "allocation_probe_radius": 0.1,
            "allocation_trust_radii": [0.1, 0.05, 0.1],
            "spectral_ranks": [1, 2],
            "spectral_probe_radius": 0.1,
            "spectral_trust_radius": 0.1,
            "quantile_smoothness_values": [0.1, 0.2],
            "quantile_trust_radius": 0.1,
            "quantile_cost_normalization": "layer_uniform_cost",
            "eval_batches": 4,
            "train_pool_batches": 14,
            "learning_rate": 5e-5,
            "device": "cpu",
            "output_dir": str(args.output_dir.resolve()),
            "scheduler": "cosine",
            "continuation_steps": 5,
        }
        self.assertEqual(result, expected)
        self.assertEqual(list(result), list(expected))
        self.assertNotIn("methods", result)
        json.dumps(result, allow_nan=False)

    def test_validation_preserves_error_messages_and_priority(self) -> None:
        base = self.base_config()
        invalid_cases = [
            (
                replace(self.with_methods(base, trust_radii=()), seeds=()),
                "--allocation-trust-radii must contain positive values",
            ),
            (
                self.with_methods(base, trust_radii=(1.1,)),
                "--allocation-trust-radii values cannot exceed 1",
            ),
            (
                self.with_methods(
                    base,
                    spectral_ranks=(-1,),
                    quantile_smoothness_values=(-1.0,),
                ),
                "--spectral-ranks must contain positive integers",
            ),
            (
                replace(
                    self.with_methods(
                        base,
                        quantile_smoothness_values=(float("nan"),),
                    ),
                    recovery_step=0,
                ),
                "--quantile-smoothness-values must contain positive values",
            ),
            (
                self.with_methods(
                    base,
                    quantile_smoothness_values=(0.1,),
                    quantile_trust_radius=0.4,
                ),
                "--quantile-trust-radius must fit inside the layer-rate box",
            ),
            (replace(base, seeds=()), "At least one seed is required"),
            (
                replace(base, recovery_step=10),
                "--recovery-step must be strictly inside --total-steps",
            ),
            (
                self.with_methods(base, prune_ratio=0.0),
                "--prune-ratio must be in (0, 1)",
            ),
            (
                self.with_methods(base, max_layer_ratio=0.4),
                "--max-layer-ratio must be in [prune_ratio, 1]",
            ),
            (
                replace(base, allocation_probe_batches=0),
                "Allocation probe and selection batch counts must be positive",
            ),
            (
                self.with_methods(base, probe_radius=0.5),
                "--allocation-probe-radius must fit around --prune-ratio",
            ),
            (
                self.with_methods(base, probe_radius=0.1, max_layer_ratio=0.55),
                "--max-layer-ratio must leave room for the positive allocation probe",
            ),
            (
                self.with_methods(base, spectral_probe_radius=0.5),
                "--spectral-probe-radius must fit around --prune-ratio",
            ),
            (
                self.with_methods(
                    base,
                    probe_radius=0.025,
                    spectral_probe_radius=0.1,
                    max_layer_ratio=0.55,
                ),
                "--max-layer-ratio must leave room for the positive spectral probe",
            ),
            (
                self.with_methods(base, spectral_trust_radius=0.0),
                "--spectral-trust-radius must be in (0, 1]",
            ),
            (
                replace(base, train_pool_batches=13),
                "--train-pool-batches must be at least 14",
            ),
        ]

        for config, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaises(ValueError) as raised:
                    long_config.validate_long_config(config)
                self.assertEqual(str(raised.exception), message)

        long_config.validate_long_config(base)

    def test_runtime_validation_short_circuits_cpu_and_rejects_missing_cuda(self) -> None:
        cpu_config = self.base_config()
        with mock.patch.object(
            long_runner.torch.cuda,
            "is_available",
        ) as is_available:
            long_runner.validate_runtime_config(cpu_config)
        is_available.assert_not_called()

        cuda_config = replace(
            cpu_config,
            methods=replace(cpu_config.methods, device="cuda"),
        )
        with mock.patch.object(
            long_runner.torch.cuda,
            "is_available",
            return_value=False,
        ) as is_available:
            with self.assertRaisesRegex(
                RuntimeError,
                "^CUDA was requested but is unavailable$",
            ):
                long_runner.validate_runtime_config(cuda_config)
        is_available.assert_called_once_with()

    def test_parse_args_defaults_remain_stable(self) -> None:
        args = long_runner.parse_args([])

        self.assertEqual(args.seeds, "42,43,44")
        self.assertEqual(args.total_steps, 100)
        self.assertEqual(args.recovery_step, 50)
        self.assertEqual(args.allocation_trust_radii, "0.025,0.05,0.1")
        self.assertEqual(args.spectral_ranks, "")
        self.assertEqual(args.quantile_smoothness_values, "")
        self.assertEqual(args.device, "cuda")


if __name__ == "__main__":
    unittest.main()
