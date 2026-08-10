import json
import unittest

import torch

import experiments.lib.residual_masks as residual_masks
import experiments.lib.residual_method_assembly as residual_method_assembly
import experiments.lib.residual_methods as residual_methods
import experiments.lib.residual_protocol as residual_protocol
import experiments.lib.residual_reporting as residual_reporting
import experiments.lib.residual_runtime as residual_runtime
import experiments.lib.residual_training as residual_training
import experiments.scripts.run_residual_recovery_long as long_runner


class TestResidualLongExperiment(unittest.TestCase):
    def test_seeded_batch_sampling_rejects_invalid_counts(self) -> None:
        pool = [{"input_ids": torch.tensor([[0]]), "labels": torch.tensor([[0]])}]

        with self.assertRaisesRegex(ValueError, "non-negative"):
            residual_training.seeded_training_batches(pool, -1, seed=0)
        with self.assertRaisesRegex(ValueError, "pool size"):
            residual_training.seeded_training_batches(pool, 2, seed=0)

    def test_partition_rejects_negative_counts_and_step_overflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "hvp_batches"):
            residual_training.partition_seed_batches(
                [], 1, 0, -1, 0, 0, seed=0
            )
        with self.assertRaisesRegex(ValueError, "recovery_step cannot"):
            residual_training.partition_seed_batches(
                [], 1, 2, 0, 0, 0, seed=0
            )

    def test_partition_is_deterministic_disjoint_and_rng_local(self) -> None:
        pool = [
            {
                "input_ids": torch.tensor([[index, index + 1]]),
                "labels": torch.tensor([[index, index + 1]]),
            }
            for index in range(10)
        ]
        rng_before = torch.get_rng_state().clone()

        partition = residual_training.partition_seed_batches(
            pool,
            total_steps=4,
            recovery_step=2,
            hvp_batches=2,
            allocation_probe_batches=1,
            allocation_selection_batches=1,
            seed=42,
        )

        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        generator = torch.Generator().manual_seed(42)
        expected_indices = torch.randperm(len(pool), generator=generator)[:8].tolist()
        self.assertEqual(partition.selected_pool_indices, expected_indices)
        self.assertEqual(
            [
                len(partition.pre_recovery),
                len(partition.scoring),
                len(partition.allocation_probe),
                len(partition.allocation_selection),
                len(partition.continuation),
            ],
            [2, 2, 1, 1, 2],
        )
        combined = (
            partition.pre_recovery
            + partition.scoring
            + partition.allocation_probe
            + partition.allocation_selection
            + partition.continuation
        )
        self.assertEqual([id(batch) for batch in combined], [id(pool[i]) for i in expected_indices])
        self.assertEqual(
            partition.data_hashes(),
            {
                "pre_recovery": residual_runtime.batch_hash(partition.pre_recovery),
                "scoring": residual_runtime.batch_hash(partition.scoring),
                "allocation_probe": residual_runtime.batch_hash(partition.allocation_probe),
                "allocation_selection": residual_runtime.batch_hash(
                    partition.allocation_selection
                ),
                "continuation": residual_runtime.batch_hash(partition.continuation),
            },
        )

    def test_base_masks_preserve_method_order_budget_and_score_routing(self) -> None:
        layers = [["layer0"], ["layer1"]]
        magnitude_scores = {
            "layer0": torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
            "layer1": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        }
        components = {
            "first_order": {
                "layer0": torch.tensor([1.0, 3.0, 5.0, 2.0, 4.0, 6.0]),
                "layer1": torch.tensor([6.0, 4.0, 2.0, 5.0, 3.0, 1.0]),
            },
            "second_order": {
                "layer0": torch.tensor([2.0, 1.0, 4.0, 3.0, 6.0, 5.0]),
                "layer1": torch.tensor([5.0, 6.0, 3.0, 4.0, 1.0, 2.0]),
            },
            "taylor": {
                "layer0": torch.tensor([0.1, 0.2, 0.4, 0.8, 1.6, 3.2]),
                "layer1": torch.tensor([0.5, 0.6, 0.8, 1.1, 1.5, 2.0]),
            },
        }
        orders = residual_masks.layer_score_orders(layers, components["taylor"])

        masks, metadata = residual_methods.build_masks(
            layers,
            magnitude_scores,
            components,
            prune_ratio=0.5,
            max_layer_ratio=0.8,
            taylor_score_orders=orders,
        )

        self.assertEqual(tuple(masks), residual_protocol.BASE_METHODS)
        self.assertEqual(metadata["eligible_parameters"], 12)
        self.assertEqual(metadata["target_pruned"], 6)
        self.assertEqual(metadata["layer_sizes"], [6, 6])
        self.assertEqual(metadata["uniform_layer_counts"], [3, 3])
        self.assertEqual(sum(metadata["weibull_layer_counts"]), 6)
        for method, method_masks in masks.items():
            with self.subTest(method=method):
                self.assertEqual(sum((~mask).sum().item() for mask in method_masks.values()), 6)
        for method in residual_protocol.BASE_METHODS[:4]:
            with self.subTest(method=method):
                self.assertEqual(
                    [int((~masks[method][name]).sum().item()) for name in ("layer0", "layer1")],
                    [3, 3],
                )

        expected_scores = {
            "residual_magnitude_uniform": magnitude_scores,
            "first_order_uniform": components["first_order"],
            "second_order_uniform": components["second_order"],
            "taylor_uniform": components["taylor"],
            "taylor_probe_trust": components["taylor"],
            "taylor_spectral_k1": components["taylor"],
            "taylor_quantile_global_smooth_l0p1": components["taylor"],
        }
        for method, expected in expected_scores.items():
            with self.subTest(method=method):
                self.assertIs(
                    residual_methods.score_for_method(method, magnitude_scores, components),
                    expected,
                )

    def test_method_diagnostics_preserve_costs_rates_regret_and_overlap(self) -> None:
        layers = [["layer0"], ["layer1"]]
        components = {
            "first_order": {
                "layer0": torch.tensor([4.0, 3.0]),
                "layer1": torch.tensor([2.0, 1.0]),
            },
            "second_order": {
                "layer0": torch.tensor([1.0, 2.0]),
                "layer1": torch.tensor([3.0, 4.0]),
            },
            "taylor": {
                "layer0": torch.tensor([1.0, 2.0]),
                "layer1": torch.tensor([3.0, 4.0]),
            },
        }
        magnitude_scores = {
            "layer0": torch.tensor([10.0, 20.0]),
            "layer1": torch.tensor([30.0, 40.0]),
        }
        method_masks = {
            "layer0": torch.tensor([True, False]),
            "layer1": torch.tensor([False, True]),
        }
        exact_masks = {
            "layer0": torch.tensor([False, False]),
            "layer1": torch.tensor([True, True]),
        }

        metrics = residual_methods.compute_method_diagnostics(
            "residual_magnitude_uniform",
            method_masks,
            exact_masks,
            exact_taylor_proxy_cost=3.0,
            layers=layers,
            magnitude_scores=magnitude_scores,
            components=components,
            eligible_parameters=4,
        )

        self.assertEqual(
            list(metrics),
            [
                "pruned",
                "proxy_cost",
                "selection_score_cost",
                "eligible_sparsity",
                "layer_rates",
                "taylor_regret_vs_exact",
                "overlap_with_taylor_exact",
            ],
        )
        self.assertEqual(metrics["pruned"], 2)
        self.assertEqual(metrics["proxy_cost"], 5.0)
        self.assertEqual(metrics["selection_score_cost"], 50.0)
        self.assertEqual(metrics["eligible_sparsity"], 0.5)
        self.assertEqual(metrics["layer_rates"], [0.5, 0.5])
        self.assertAlmostEqual(metrics["taylor_regret_vs_exact"], 2.0 / 3.0)
        self.assertEqual(
            metrics["overlap_with_taylor_exact"],
            {"pruned_jaccard": 1.0 / 3.0, "mask_disagreements": 2},
        )

        zero_exact_metrics = residual_methods.compute_method_diagnostics(
            "taylor_uniform",
            method_masks={"layer0": torch.tensor([True, False])},
            exact_masks={"layer0": torch.tensor([False, True])},
            exact_taylor_proxy_cost=0.0,
            layers=[["layer0"]],
            magnitude_scores={"layer0": torch.tensor([0.0, 1.0])},
            components={
                "first_order": {"layer0": torch.tensor([0.0, 1.0])},
                "second_order": {"layer0": torch.tensor([0.0, 1.0])},
                "taylor": {"layer0": torch.tensor([0.0, 1.0])},
            },
            eligible_parameters=2,
        )
        self.assertEqual(
            zero_exact_metrics["taylor_regret_vs_exact"],
            1.0 / 1e-30,
        )

    def test_aggregate_preserves_schema_seed_order_and_paired_deltas(self) -> None:
        methods = list(residual_protocol.FIXED_METHODS) + [
            "taylor_spectral_k1",
            "taylor_quantile_global_smooth_l0p1",
        ]

        def seed_result(offset: float) -> dict:
            method_results = {}
            for index, method in enumerate(methods):
                immediate = 10.0 - index + offset
                method_results[method] = {
                    "immediate": {"perplexity": immediate},
                    "final": {"perplexity": immediate - 2.0},
                }
            return {
                "current": {"perplexity": 20.0 + offset},
                "no_compression_final": {"perplexity": 18.0 + offset},
                "methods": method_results,
            }

        aggregate = residual_reporting.aggregate(
            {"43": seed_result(1.0), "42": seed_result(0.0)}
        )

        self.assertEqual(
            list(aggregate),
            [
                "current_perplexity",
                "no_compression_final_perplexity",
                "methods",
                "paired_comparisons",
            ],
        )
        self.assertEqual(aggregate["current_perplexity"]["values"], [20.0, 21.0])
        self.assertEqual(
            aggregate["methods"]["taylor_uniform"]["immediate_perplexity"]["values"],
            [7.0, 8.0],
        )
        self.assertEqual(
            aggregate["methods"]["taylor_uniform"][
                "paired_immediate_delta_vs_magnitude"
            ]["values"],
            [-3.0, -3.0],
        )
        comparisons = aggregate["paired_comparisons"]
        self.assertEqual(
            comparisons["taylor_vs_first_order"]["immediate_perplexity_delta"]["values"],
            [-2.0, -2.0],
        )
        self.assertEqual(
            comparisons["spectral_k1_vs_probe_trust"]["immediate_perplexity_delta"]
            ["values"],
            [-1.0, -1.0],
        )
        self.assertEqual(
            comparisons["quantile_global_smooth_l0p1_vs_weibull"]
            ["immediate_perplexity_delta"]["values"],
            [-4.0, -4.0],
        )
        self.assertEqual(
            list(residual_reporting.summarize_values([3.0])),
            ["values", "mean", "std", "ci95"],
        )
        self.assertIsNone(residual_reporting.summarize_values([3.0])["std"])
        json.dumps(aggregate, allow_nan=False)

    def test_reporting_rejects_empty_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "values must not be empty"):
            residual_reporting.summarize_values([])
        with self.assertRaisesRegex(ValueError, "seed_results must not be empty"):
            residual_reporting.aggregate({})

    def test_method_identifier_protocol_preserves_existing_strings(self) -> None:
        self.assertEqual(
            residual_protocol.FIXED_METHODS,
            (
                "residual_magnitude_uniform",
                "first_order_uniform",
                "second_order_uniform",
                "taylor_uniform",
                "taylor_weibull_mom",
                "taylor_exact_global",
                "taylor_probe_trust",
            ),
        )
        self.assertEqual(residual_protocol.NO_COMPRESSION_METHOD, "no_compression")
        self.assertEqual(residual_protocol.spectral_method_id(3), "taylor_spectral_k3")
        self.assertEqual(
            residual_protocol.quantile_method_id("global", 0.1),
            "taylor_quantile_global_smooth_l0p1",
        )
        self.assertEqual(
            residual_protocol.quantile_method_id("layer_uniform_cost", 0.1),
            "taylor_quantile_relative_smooth_l0p1",
        )

    def test_runner_reexports_moved_public_helpers(self) -> None:
        aliases = {
            "AdaptiveMethodConfig": residual_method_assembly.AdaptiveMethodConfig,
            "clone_model_state_to_cpu": residual_training.clone_model_state_to_cpu,
            "seeded_training_batches": residual_training.seeded_training_batches,
            "build_optimizer": residual_training.build_optimizer,
            "train_segment": residual_training.train_segment,
            "build_masks": residual_methods.build_masks,
            "score_for_method": residual_methods.score_for_method,
            "summarize_values": residual_reporting.summarize_values,
            "float_slug": residual_methods.float_slug,
            "aggregate": residual_reporting.aggregate,
        }
        for name, owner in aliases.items():
            with self.subTest(name=name):
                self.assertIs(getattr(long_runner, name), owner)


if __name__ == "__main__":
    unittest.main()
