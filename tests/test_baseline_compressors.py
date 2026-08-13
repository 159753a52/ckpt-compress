import pickle
import unittest

import torch

from baselines.excp.excp import ExCPCompressor, ExCPConfig
from baselines.excp.quantization import pack_int4, unpack_int4
from baselines.inshrinkerator.approx_kmeans import (
    approx_kmeans,
    quantize_to_centers,
    weighted_kmeans_plusplus_init,
)
from baselines.inshrinkerator.delta_encoding import rle_decode, rle_encode
from baselines.inshrinkerator.inshrinkerator import InshrinkeratorCompressor, InshrinkeratorConfig
from baselines.inshrinkerator.partition import PartitionConfig, partition
from baselines.inshrinkerator.sketch import QuantileSketch


class TestInshrinkeratorCompressor(unittest.TestCase):
    def test_incremental_indices_round_trip_across_two_checkpoints(self) -> None:
        torch.manual_seed(7)
        compressor = InshrinkeratorCompressor(
            InshrinkeratorConfig(
                n_bins=4,
                protect_fraction=0.0,
                prune_fraction=0.0,
                use_gzip=False,
            )
        )
        first_weights = {"weight": torch.linspace(-2.0, 2.0, 24).reshape(4, 6)}
        second_weights = {"weight": first_weights["weight"] * 0.7 + 0.13}
        gradients = {"weight": torch.linspace(0.1, 1.0, 24).reshape(4, 6)}

        first_payload = compressor.compress(first_weights, gradients)
        first_indices = compressor.get_quantized_indices(first_payload)
        second_payload = compressor.compress(
            second_weights,
            gradients,
            prev_quantized=first_indices,
        )
        second_indices = compressor.get_quantized_indices(
            second_payload,
            prev_quantized=first_indices,
        )
        reconstructed = compressor.decompress(
            second_payload,
            prev_quantized=first_indices,
        )
        second_data = pickle.loads(second_payload)["weights"]["weight"]
        centers = torch.from_numpy(second_data["centers"])
        expected_indices = quantize_to_centers(second_weights["weight"].abs(), centers)

        self.assertEqual(second_indices["weight"].shape, second_weights["weight"].shape)
        self.assertTrue(torch.equal(second_indices["weight"], expected_indices))
        self.assertTrue(torch.isfinite(reconstructed["weight"]).all())
        self.assertEqual(reconstructed["weight"].shape, second_weights["weight"].shape)

    def test_incremental_index_extraction_requires_previous_indices(self) -> None:
        torch.manual_seed(11)
        compressor = InshrinkeratorCompressor(
            InshrinkeratorConfig(
                n_bins=4,
                protect_fraction=0.0,
                prune_fraction=0.0,
                use_gzip=False,
            )
        )
        weights = {"weight": torch.arange(16, dtype=torch.float32).reshape(4, 4)}
        gradients = {"weight": torch.ones(4, 4)}
        first = compressor.compress(weights, gradients)
        first_indices = compressor.get_quantized_indices(first)
        second = compressor.compress(weights, gradients, prev_quantized=first_indices)

        with self.assertRaisesRegex(ValueError, "require the previous"):
            compressor.get_quantized_indices(second)

    def test_incremental_decode_uses_configured_bin_modulus_for_small_layers(self) -> None:
        compressor = InshrinkeratorCompressor(
            InshrinkeratorConfig(
                n_bins=16,
                protect_fraction=0.0,
                prune_fraction=0.0,
                use_gzip=False,
            )
        )
        gradients = {"weight": torch.ones(8)}
        first = compressor.compress({"weight": torch.ones(8)}, gradients)
        first_indices = compressor.get_quantized_indices(first)
        second = compressor.compress(
            {"weight": torch.full((8,), 1.25)},
            gradients,
            prev_quantized=first_indices,
        )

        second_indices = compressor.get_quantized_indices(
            second,
            prev_quantized=first_indices,
        )
        reconstructed = compressor.decompress(second, prev_quantized=first_indices)

        self.assertTrue(torch.equal(second_indices["weight"], torch.zeros(8, dtype=torch.long)))
        self.assertEqual(reconstructed["weight"].shape, (8,))

    def test_rle_rejects_truncated_or_wrong_length_payloads(self) -> None:
        encoded = rle_encode(torch.tensor([1, 1, 2], dtype=torch.long))

        with self.assertRaisesRegex(ValueError, "complete int16"):
            rle_decode(encoded[:-1], 3)
        with self.assertRaisesRegex(ValueError, "expected 4"):
            rle_decode(encoded, 4)

    def test_rle_splits_runs_that_exceed_int16_capacity(self) -> None:
        values = torch.full((100_000,), 7, dtype=torch.long)

        decoded = rle_decode(rle_encode(values), values.numel())

        self.assertTrue(torch.equal(decoded, values))

    def test_sketch_accepts_multidimensional_values_and_validates_queries(self) -> None:
        sketch = QuantileSketch(alpha=0.01)
        sketch.add(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        self.assertEqual(sketch.count, 4)
        self.assertGreater(sketch.quantile(0.5), 0.0)
        with self.assertRaisesRegex(ValueError, "q must"):
            sketch.quantile(1.1)
        with self.assertRaisesRegex(ValueError, "different alpha"):
            sketch.merge(QuantileSketch(alpha=0.02))

    def test_approx_kmeans_preserves_dtype_and_rejects_invalid_inputs(self) -> None:
        empty = torch.empty(0, dtype=torch.float64)
        self.assertEqual(approx_kmeans(empty, 2).dtype, torch.float64)
        centers = approx_kmeans(torch.tensor([0.0, 1.0, 2.0, 3.0]), 2)
        self.assertTrue(torch.isfinite(centers).all())

        with self.assertRaisesRegex(ValueError, "k must"):
            approx_kmeans(torch.ones(2), 0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            approx_kmeans(torch.tensor([-1.0, 1.0]), 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            approx_kmeans(torch.tensor([float("nan")]), 1)

    def test_weighted_initializer_handles_zero_weights_without_nan(self) -> None:
        torch.manual_seed(3)
        centers = weighted_kmeans_plusplus_init(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.zeros(3),
            2,
        )

        self.assertEqual(centers.shape, (2,))
        self.assertTrue(torch.isfinite(centers).all())

    def test_config_rejects_invalid_partition_or_bin_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_bins"):
            InshrinkeratorConfig(n_bins=0)
        with self.assertRaisesRegex(ValueError, "prune_fraction"):
            InshrinkeratorConfig(prune_fraction=1.1)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            InshrinkeratorConfig(protect_fraction=0.6, prune_fraction=0.5)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            PartitionConfig(protect_fraction=0.6, prune_fraction=0.5)

    def test_partition_full_pruning_and_mask_device_contract(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0])
        result = partition(
            values,
            torch.ones_like(values),
            PartitionConfig(protect_fraction=0.0, prune_fraction=1.0),
        )

        self.assertEqual(result.prune_mask.tolist(), [1, 1, 1])
        self.assertEqual(result.quantize_mask.tolist(), [0, 0, 0])
        self.assertEqual(result.prune_mask.device, values.device)

    def test_partition_uses_exact_stable_counts_across_ties(self) -> None:
        values = torch.ones(10)
        gradients = torch.ones_like(values)

        result = partition(
            values,
            gradients,
            PartitionConfig(protect_fraction=0.0, prune_fraction=0.2),
        )
        self.assertEqual(int(result.prune_mask.sum().item()), 2)
        self.assertEqual(int(result.quantize_mask.sum().item()), 8)
        self.assertEqual(result.prune_mask.tolist(), [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

        protected = partition(
            values,
            gradients,
            PartitionConfig(protect_fraction=0.1, prune_fraction=0.2),
        )
        self.assertEqual(int(protected.protect_mask.sum().item()), 1)
        self.assertEqual(int(protected.prune_mask.sum().item()), 2)
        self.assertEqual(int(protected.quantize_mask.sum().item()), 7)

    def test_partition_rejects_malformed_numeric_inputs(self) -> None:
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            partition([1.0], torch.ones(1), PartitionConfig())
        with self.assertRaisesRegex(ValueError, "same shape"):
            partition(torch.ones(2), torch.ones(3), PartitionConfig())
        with self.assertRaisesRegex(ValueError, "finite"):
            partition(
                torch.tensor([float("nan")]),
                torch.ones(1),
                PartitionConfig(),
            )

    def test_compress_accepts_requires_grad_and_validates_gradient_shape(self) -> None:
        compressor = InshrinkeratorCompressor(
            InshrinkeratorConfig(
                n_bins=2,
                protect_fraction=0.0,
                prune_fraction=0.0,
                use_gzip=False,
            )
        )
        weight = torch.arange(6, dtype=torch.float32, requires_grad=True).reshape(2, 3)

        payload = compressor.compress({"weight": weight}, {"weight": torch.ones_like(weight)})
        reconstructed = compressor.decompress(payload)

        self.assertEqual(reconstructed["weight"].shape, weight.shape)
        with self.assertRaisesRegex(ValueError, "Gradient shape"):
            compressor.compress({"weight": weight}, {"weight": torch.ones(3)})

    def test_compress_requires_complete_gradient_keys(self) -> None:
        compressor = InshrinkeratorCompressor(InshrinkeratorConfig(use_gzip=False))
        weights = {"left": torch.ones(2), "right": torch.ones(2)}

        with self.assertRaisesRegex(ValueError, r"missing=\['right'\]"):
            compressor.compress(weights, {"left": torch.ones(2)})
        with self.assertRaisesRegex(ValueError, r"extra=\['unused'\]"):
            compressor.compress(
                weights,
                {
                    "left": torch.ones(2),
                    "right": torch.ones(2),
                    "unused": torch.ones(2),
                },
            )

    def test_round_trip_preserves_supported_weight_dtypes(self) -> None:
        compressor = InshrinkeratorCompressor(
            InshrinkeratorConfig(
                n_bins=2,
                protect_fraction=0.0,
                prune_fraction=0.0,
                use_gzip=False,
            )
        )
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                weight = torch.arange(1, 7, dtype=dtype).reshape(2, 3)
                payload = compressor.compress(
                    {"weight": weight},
                    {"weight": torch.ones_like(weight)},
                )
                reconstructed = compressor.decompress(payload)["weight"]
                self.assertEqual(reconstructed.dtype, dtype)

    def test_protected_bfloat16_values_do_not_overflow_storage(self) -> None:
        compressor = InshrinkeratorCompressor(
            InshrinkeratorConfig(
                n_bins=2,
                protect_fraction=1.0,
                prune_fraction=0.0,
                use_gzip=False,
            )
        )
        weight = torch.tensor([1.0e10, -1.0e10], dtype=torch.bfloat16)

        payload = compressor.compress(
            {"weight": weight},
            {"weight": torch.ones_like(weight)},
        )
        reconstructed = compressor.decompress(payload)["weight"]

        self.assertTrue(torch.isfinite(reconstructed).all())
        self.assertTrue(torch.equal(reconstructed, weight))


class TestExCPCompressor(unittest.TestCase):
    def test_first_and_residual_checkpoints_round_trip_without_fake_state(self) -> None:
        torch.manual_seed(19)
        compressor = ExCPCompressor(ExCPConfig(alpha=0.0, beta=0.0, use_gzip=False))
        first_weights = {"weight": torch.linspace(-1.0, 1.0, 15).reshape(3, 5)}
        second_weights = {"weight": first_weights["weight"] + 0.2}
        optimizer = {
            "weight": {
                "exp_avg": torch.linspace(-0.2, 0.2, 15).reshape(3, 5),
                "exp_avg_sq": torch.ones(3, 5),
            }
        }

        first_payload = compressor.compress(first_weights, optimizer)
        first_reconstructed, first_optimizer = compressor.decompress(first_payload)
        second_payload = compressor.compress(
            second_weights,
            optimizer,
            prev_W_hat=first_reconstructed,
        )
        second_reconstructed, second_optimizer = compressor.decompress(
            second_payload,
            prev_W_hat=first_reconstructed,
        )

        self.assertEqual(second_reconstructed["weight"].shape, (3, 5))
        self.assertEqual(second_reconstructed["weight"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(second_reconstructed["weight"]).all())
        self.assertEqual(set(first_optimizer["weight"]), {"exp_avg"})
        self.assertEqual(set(second_optimizer["weight"]), {"exp_avg"})

    def test_config_and_int4_payload_boundaries_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_bits"):
            ExCPConfig(n_bits=5)
        with self.assertRaisesRegex(ValueError, "p must"):
            ExCPConfig(p=101.0)
        with self.assertRaisesRegex(ValueError, r"\[0, 15\]"):
            pack_int4(torch.tensor([16]))
        with self.assertRaisesRegex(ValueError, "too short"):
            unpack_int4(torch.tensor([0], dtype=torch.uint8), 3)

    def test_compress_rejects_mismatched_optimizer_state(self) -> None:
        compressor = ExCPCompressor(ExCPConfig(use_gzip=False))
        weights = {"weight": torch.ones(2, 2)}
        optimizer = {
            "weight": {
                "exp_avg": torch.ones(3),
                "exp_avg_sq": torch.ones(2, 2),
            }
        }

        with self.assertRaisesRegex(ValueError, "exp_avg shape"):
            compressor.compress(weights, optimizer)

    def test_compress_rejects_negative_second_moment(self) -> None:
        compressor = ExCPCompressor(ExCPConfig(use_gzip=False))
        weights = {"weight": torch.ones(2, 2)}
        optimizer = {
            "weight": {
                "exp_avg": torch.ones(2, 2),
                "exp_avg_sq": -torch.ones(2, 2),
            }
        }

        with self.assertRaisesRegex(ValueError, "non-negative"):
            compressor.compress(weights, optimizer)

    def test_baseline_compressors_are_deterministic_without_global_rng_side_effects(self) -> None:
        weights = {"weight": torch.linspace(-3.0, 3.0, 128).reshape(16, 8)}
        gradients = {"weight": torch.linspace(0.1, 1.0, 128).reshape(16, 8)}
        optimizer = {
            "weight": {
                "exp_avg": gradients["weight"],
                "exp_avg_sq": torch.ones(16, 8),
            }
        }
        cases = (
            (
                InshrinkeratorCompressor(
                    InshrinkeratorConfig(
                        n_bins=4,
                        protect_fraction=0.0,
                        prune_fraction=0.0,
                        use_gzip=False,
                        seed=7,
                    )
                ),
                lambda compressor: compressor.compress(weights, gradients),
            ),
            (
                ExCPCompressor(ExCPConfig(alpha=0.0, beta=0.0, use_gzip=False, seed=7)),
                lambda compressor: compressor.compress(weights, optimizer),
            ),
        )
        for compressor, compress in cases:
            with self.subTest(compressor=type(compressor).__name__):
                torch.manual_seed(123)
                state_before = torch.get_rng_state().clone()
                first = compress(compressor)
                self.assertTrue(torch.equal(torch.get_rng_state(), state_before))
                _ = torch.rand(100)
                second = compress(compressor)
                self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
