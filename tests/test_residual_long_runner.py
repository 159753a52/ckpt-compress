import argparse
import copy
import io
import json
import math
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Mapping, Sequence
from unittest import mock

import torch

import experiments.lib.residual_long_config as long_config
import experiments.lib.residual_method_assembly as method_assembly
import experiments.scripts.run_residual_recovery_long as long_runner
from experiments.lib.residual_masks import layer_masks, layer_score_orders
from experiments.lib.residual_methods import build_masks, quantile_method_id, spectral_method_id
from experiments.lib.residual_protocol import FIXED_METHODS, NO_COMPRESSION_METHOD
from experiments.lib.residual_training import partition_seed_batches


class TinyLongRunModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer0 = torch.nn.Parameter(torch.zeros(4))
        self.layer1 = torch.nn.Parameter(torch.zeros(4))


def clone_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


class TestResidualLongRunner(unittest.TestCase):
    def assert_nested_state_equal(
        self,
        actual,
        expected,
        path: str = "state",
    ) -> None:
        if torch.is_tensor(expected):
            self.assertTrue(torch.is_tensor(actual), path)
            self.assertTrue(torch.equal(actual, expected), path)
            return
        if isinstance(expected, Mapping):
            self.assertIsInstance(actual, Mapping, path)
            self.assertEqual(set(actual), set(expected), path)
            for key in expected:
                self.assert_nested_state_equal(
                    actual[key],
                    expected[key],
                    f"{path}.{key}",
                )
            return
        if isinstance(expected, (list, tuple)):
            self.assertIsInstance(actual, type(expected), path)
            self.assertEqual(len(actual), len(expected), path)
            for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
                self.assert_nested_state_equal(
                    actual_item,
                    expected_item,
                    f"{path}[{index}]",
                )
            return
        self.assertEqual(actual, expected, path)

    def assert_batch_sequence_identity(
        self,
        actual: Sequence[Mapping[str, torch.Tensor]],
        expected: Sequence[Mapping[str, torch.Tensor]],
    ) -> None:
        self.assertEqual(len(actual), len(expected))
        for actual_batch, expected_batch in zip(actual, expected):
            self.assertIs(actual_batch, expected_batch)

    def assert_state_equal(
        self,
        actual: Mapping[str, torch.Tensor],
        expected: Mapping[str, torch.Tensor],
    ) -> None:
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_run_seed_preserves_stage_order_state_isolation_and_schema(self) -> None:
        seed = 42
        args = argparse.Namespace(
            checkpoint=Path("checkpoint.pt"),
            data_dir=Path("data"),
            output_dir=Path("results"),
            seeds=str(seed),
            total_steps=2,
            recovery_step=1,
            hvp_batches=1,
            allocation_probe_batches=1,
            allocation_selection_batches=1,
            prune_ratio=0.5,
            max_layer_ratio=0.8,
            batch_size=1,
            seq_length=2,
            allocation_probe_radius=0.1,
            allocation_trust_radii="0.1",
            spectral_ranks="1",
            spectral_probe_radius=0.1,
            spectral_trust_radius=0.1,
            quantile_smoothness_values="0.1",
            quantile_trust_radius=0.1,
            quantile_cost_normalization="global",
            eval_batches=1,
            train_pool_batches=5,
            learning_rate=1e-3,
            device="cpu",
        )
        config = long_config.normalize_long_config(args)
        training_pool = [
            {
                "input_ids": torch.tensor([[index, index + 1]]),
                "labels": torch.tensor([[index, index + 1]]),
            }
            for index in range(5)
        ]
        eval_batches = [
            {
                "input_ids": torch.tensor([[100, 101]]),
                "labels": torch.tensor([[100, 101]]),
            }
        ]
        expected_partition = partition_seed_batches(
            training_pool,
            args.total_steps,
            args.recovery_step,
            args.hvp_batches,
            args.allocation_probe_batches,
            args.allocation_selection_batches,
            seed,
        )

        template = TinyLongRunModel()
        reference_state = {
            name: value.detach().clone() for name, value in template.state_dict().items()
        }
        reference_optimizer = torch.optim.AdamW(
            template.parameters(),
            lr=5e-5,
            weight_decay=0.01,
        )
        reference_optimizer_state = copy.deepcopy(reference_optimizer.state_dict())

        output_path = Path("result.json")
        results: dict[str, object] = {"seed_results": {}}
        created_models: list[TinyLongRunModel] = []
        events: list[str] = []
        writes: list[dict[str, object]] = []
        evaluated_states: list[dict[str, torch.Tensor]] = []
        train_calls: list[dict[str, object]] = []
        post_step_optimizer_state: dict[str, object] = {}
        post_step_scheduler_state: dict[str, object] = {}

        layers = [["layer0"], ["layer1"]]
        components = {
            "first_order": {
                "layer0": torch.tensor([4.0, 1.0, 3.0, 2.0]),
                "layer1": torch.tensor([2.0, 3.0, 1.0, 4.0]),
            },
            "second_order": {
                "layer0": torch.tensor([1.0, 3.0, 2.0, 4.0]),
                "layer1": torch.tensor([4.0, 2.0, 3.0, 1.0]),
            },
            "taylor": {
                "layer0": torch.tensor([0.1, 0.2, 0.4, 0.8]),
                "layer1": torch.tensor([0.5, 0.6, 0.8, 1.1]),
            },
        }
        scoring_metadata = {
            "total_seconds": 0.0,
            "layer_seconds": [0.0, 0.0],
            "hvp_batches": 1,
        }

        def model_factory() -> TinyLongRunModel:
            events.append("factory")
            model = TinyLongRunModel()
            created_models.append(model)
            return model

        def evaluate_side_effect(
            model: torch.nn.Module,
            batches: Sequence[Mapping[str, torch.Tensor]],
            device: str,
        ) -> dict[str, float | int]:
            self.assertIs(batches, eval_batches)
            self.assertEqual(device, "cpu")
            events.append("eval")
            evaluated_states.append(clone_parameters(model))
            index = len(evaluated_states) - 1
            return {
                "loss": float(index),
                "perplexity": float(100 + index),
                "seconds": 0.0,
                "batches": 1,
                "tokens": 2,
            }

        def train_side_effect(
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            batches: Sequence[Mapping[str, torch.Tensor]],
            train_seed: int,
            device: str,
        ) -> dict[str, object]:
            call_index = len(train_calls)
            phase = "pre" if call_index == 0 else "continuation"
            events.append(f"train:{phase}")
            state = clone_parameters(model)
            train_calls.append(
                {
                    "phase": phase,
                    "state": state,
                    "batches": batches,
                    "seed": train_seed,
                    "device": device,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "optimizer_state": copy.deepcopy(optimizer.state_dict()),
                    "scheduler_state": copy.deepcopy(scheduler.state_dict()),
                }
            )
            if phase == "pre":
                self.assertEqual(train_seed, seed)
                for parameter in model.parameters():
                    self.assertTrue(torch.equal(parameter, torch.zeros_like(parameter)))
                for index, parameter in enumerate(model.parameters(), start=1):
                    parameter.grad = torch.full_like(parameter, 0.25 * index)
                optimizer.step()
                scheduler.step()
                post_step_optimizer_state.update(copy.deepcopy(optimizer.state_dict()))
                post_step_scheduler_state.update(copy.deepcopy(scheduler.state_dict()))
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.fill_(1.0)
            else:
                self.assertEqual(train_seed, seed + 1_000_000)
            return {
                "steps": len(batches),
                "seconds": 0.0,
                "train_losses": [0.0] * len(batches),
                "learning_rates": [float(optimizer.param_groups[0]["lr"])] * len(batches),
            }

        def scoring_side_effect(
            model: torch.nn.Module,
            batches: Sequence[Mapping[str, torch.Tensor]],
            scoring_layers: Sequence[Sequence[str]],
            delta: Mapping[str, torch.Tensor],
            device: str,
            return_components: bool = False,
        ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, object]]:
            events.append("score")
            self.assert_batch_sequence_identity(batches, expected_partition.scoring)
            self.assertEqual(scoring_layers, layers)
            self.assertEqual(device, "cpu")
            self.assertTrue(return_components)
            for name in ("layer0", "layer1"):
                self.assertTrue(torch.equal(delta[name], torch.ones(4)))
                self.assertTrue(torch.equal(dict(model.named_parameters())[name], torch.ones(4)))
            return components, copy.deepcopy(scoring_metadata)

        def trust_side_effect(*call_args, **call_kwargs):
            del call_kwargs
            events.append("trust")
            self.assert_batch_sequence_identity(
                call_args[9],
                expected_partition.allocation_probe,
            )
            self.assert_batch_sequence_identity(
                call_args[10],
                expected_partition.allocation_selection,
            )
            return [3, 1], {"selected_trust_radius": 0.1}

        def spectral_side_effect(*call_args, **call_kwargs):
            del call_kwargs
            events.append("spectral")
            self.assert_batch_sequence_identity(
                call_args[8],
                expected_partition.allocation_probe,
            )
            return {1: [1, 3]}, {"direction_evaluations": 2}

        def quantile_side_effect(*call_args, **call_kwargs):
            del call_kwargs
            events.append("quantile")
            return {0.1: [2, 2]}, {
                "model_forward_evaluations": 0,
                "batch_forward_evaluations": 0,
            }

        def write_side_effect(path: Path, payload: Mapping) -> None:
            self.assertEqual(path, output_path)
            self.assertIs(payload, results)
            events.append("write")
            writes.append(copy.deepcopy(payload))

        context = long_runner.LongRunContext(
            config=config,
            output_path=output_path,
            results=results,
            model_factory=model_factory,
            training_pool=training_pool,
            eval_batches=eval_batches,
            reference_state=reference_state,
            reference_optimizer_state=reference_optimizer_state,
        )

        with (
            mock.patch.object(
                long_runner,
                "evaluate_lm",
                side_effect=evaluate_side_effect,
            ),
            mock.patch.object(
                long_runner,
                "train_segment",
                side_effect=train_side_effect,
            ),
            mock.patch.object(
                long_runner,
                "eligible_layers",
                side_effect=lambda model: events.append("eligible") or layers,
            ),
            mock.patch.object(
                long_runner,
                "compute_block_taylor_scores",
                side_effect=scoring_side_effect,
            ),
            mock.patch.object(
                method_assembly,
                "calibrate_trust_region_allocation",
                side_effect=trust_side_effect,
            ),
            mock.patch.object(
                method_assembly,
                "calibrate_spectral_allocation",
                side_effect=spectral_side_effect,
            ),
            mock.patch.object(
                method_assembly,
                "calibrate_quantile_smooth_allocation",
                side_effect=quantile_side_effect,
            ),
            mock.patch.object(
                long_runner,
                "write_json",
                side_effect=write_side_effect,
            ),
            redirect_stdout(io.StringIO()),
        ):
            result = long_runner.run_seed(context, seed, seed_index=1, seed_count=1)

        self.assertIsNone(result)
        expected_methods = [
            *FIXED_METHODS,
            spectral_method_id(1),
            quantile_method_id("global", 0.1),
        ]
        self.assertEqual(len(expected_methods), 9)
        self.assertEqual(len(writes), 22)
        self.assertEqual(len(evaluated_states), 21)
        self.assertEqual(len(train_calls), 11)
        self.assertEqual(
            events,
            [
                "write",
                "factory",
                "eval",
                "train:pre",
                "eval",
                "write",
                "eligible",
                "score",
                "trust",
                "spectral",
                "quantile",
                *[event for _ in expected_methods for event in ("eval", "write")],
                *[
                    event
                    for _ in [NO_COMPRESSION_METHOD, *expected_methods]
                    for event in ("train:continuation", "eval", "write")
                ],
                "write",
            ],
        )

        first_seed = writes[0]["seed_results"][str(seed)]
        self.assertEqual(
            set(first_seed),
            {"seed", "selected_pool_indices", "data_hashes", "methods"},
        )
        self.assertEqual(first_seed["methods"], {})
        self.assertEqual(
            first_seed["selected_pool_indices"],
            expected_partition.selected_pool_indices,
        )
        self.assertEqual(first_seed["data_hashes"], expected_partition.data_hashes())

        second_seed = writes[1]["seed_results"][str(seed)]
        self.assertEqual(
            set(second_seed),
            {
                "seed",
                "selected_pool_indices",
                "data_hashes",
                "methods",
                "reference",
                "pre_recovery_training",
                "current",
            },
        )
        after_immediate = writes[10]["seed_results"][str(seed)]
        self.assertEqual(list(after_immediate["methods"]), expected_methods)
        for method in expected_methods:
            self.assertIn("immediate", after_immediate["methods"][method])
            self.assertNotIn("continuation", after_immediate["methods"][method])
        self.assertNotIn("no_compression_final", after_immediate)

        after_no_compression = writes[11]["seed_results"][str(seed)]
        self.assertIn("no_compression_continuation", after_no_compression)
        self.assertIn("no_compression_final", after_no_compression)
        self.assertNotIn("continuation", after_no_compression["methods"][expected_methods[0]])

        final_seed = results["seed_results"][str(seed)]
        self.assertEqual(
            set(final_seed),
            {
                "seed",
                "selected_pool_indices",
                "data_hashes",
                "methods",
                "reference",
                "pre_recovery_training",
                "current",
                "scoring",
                "allocation",
                "no_compression_continuation",
                "no_compression_final",
                "wall_seconds",
            },
        )
        self.assertTrue(math.isfinite(final_seed["wall_seconds"]))
        self.assertGreaterEqual(final_seed["wall_seconds"], 0.0)
        self.assertEqual(final_seed["scoring"]["peak_gpu_memory_bytes"], 0)
        self.assertEqual(final_seed["allocation"]["whole_model_parameters"], 8)
        self.assertEqual(
            final_seed["allocation"]["quantile_smooth_layer_counts"],
            {"0.1": [2, 2]},
        )
        json.dumps(final_seed, allow_nan=False)

        zero_state = {"layer0": torch.zeros(4), "layer1": torch.zeros(4)}
        current = {"layer0": torch.ones(4), "layer1": torch.ones(4)}
        self.assert_state_equal(evaluated_states[0], zero_state)
        self.assert_state_equal(evaluated_states[1], current)
        self.assertEqual(
            [call["seed"] for call in train_calls],
            [seed] + [seed + 1_000_000] * 10,
        )
        self.assert_batch_sequence_identity(
            train_calls[0]["batches"],
            expected_partition.pre_recovery,
        )
        self.assertEqual(reference_optimizer_state["state"], {})
        self.assertTrue(post_step_optimizer_state["state"])
        self.assertNotEqual(
            set(post_step_optimizer_state["state"]),
            set(reference_optimizer_state["state"]),
        )
        post_step_learning_rate = post_step_optimizer_state["param_groups"][0]["lr"]
        self.assertNotEqual(post_step_learning_rate, args.learning_rate)
        for call in train_calls[1:]:
            self.assert_batch_sequence_identity(
                call["batches"],
                expected_partition.continuation,
            )
            self.assertEqual(call["device"], "cpu")
            self.assertAlmostEqual(call["learning_rate"], post_step_learning_rate)
            self.assertTrue(call["optimizer_state"]["state"])
            self.assertNotEqual(
                set(call["optimizer_state"]["state"]),
                set(reference_optimizer_state["state"]),
            )
            self.assert_nested_state_equal(
                call["optimizer_state"],
                post_step_optimizer_state,
                "optimizer_state",
            )
            self.assert_nested_state_equal(
                call["scheduler_state"],
                post_step_scheduler_state,
                "scheduler_state",
            )

        self.assert_state_equal(train_calls[1]["state"], current)

        magnitude_scores = {name: torch.ones(4) for name in ("layer0", "layer1")}
        orders = layer_score_orders(layers, components["taylor"])
        expected_masks, _ = build_masks(
            layers,
            magnitude_scores,
            components,
            args.prune_ratio,
            args.max_layer_ratio,
            orders,
        )
        expected_masks["taylor_probe_trust"] = layer_masks(
            layers,
            components["taylor"],
            [3, 1],
            orders,
        )
        expected_masks[spectral_method_id(1)] = layer_masks(
            layers,
            components["taylor"],
            [1, 3],
            orders,
        )
        expected_masks[quantile_method_id("global", 0.1)] = layer_masks(
            layers,
            components["taylor"],
            [2, 2],
            orders,
        )
        for method_index, method in enumerate(expected_methods):
            immediate_state = evaluated_states[2 + method_index]
            continuation_state = train_calls[2 + method_index]["state"]
            final_state = evaluated_states[12 + method_index]
            expected_state = {
                name: expected_masks[method][name].to(dtype=torch.float32)
                for name in ("layer0", "layer1")
            }
            self.assert_state_equal(immediate_state, expected_state)
            for name in ("layer0", "layer1"):
                self.assertTrue(torch.equal(immediate_state[name], continuation_state[name]))
                self.assertTrue(torch.equal(continuation_state[name], final_state[name]))

        self.assertEqual(
            [
                final_seed["methods"][method]["immediate"]["perplexity"]
                for method in expected_methods
            ],
            [float(value) for value in range(102, 111)],
        )
        self.assertEqual(final_seed["no_compression_final"]["perplexity"], 111.0)
        self.assertEqual(
            [final_seed["methods"][method]["final"]["perplexity"] for method in expected_methods],
            [float(value) for value in range(112, 121)],
        )

        for name, value in reference_state.items():
            self.assertTrue(torch.equal(value, torch.zeros_like(value)), name)
        self.assertEqual(len(created_models), 1)
        self.assert_state_equal(
            clone_parameters(created_models[0]),
            expected_masks[expected_methods[-1]],
        )


if __name__ == "__main__":
    unittest.main()
