import copy
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

import experiments.lib.residual_recovery as residual_recovery
import experiments.lib.residual_short_experiment as short_runner
import experiments.scripts.run_v100_allocation_gate as short_cli
from experiments.lib.residual_protocol import SHORT_GATE_METHODS
from experiments.lib.residual_short_methods import ShortResidualScope


class TrackingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))
        self.loaded_marker = None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        marker = float(state_dict["weight"][0].item())
        self.loaded_marker = "current" if marker == 2.0 else "reference"
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class TestResidualShortExperiment(unittest.TestCase):
    def test_recovery_facade_reexports_short_experiment_entrypoints(self) -> None:
        self.assertIs(residual_recovery.main, short_runner.main)
        self.assertIs(short_cli.main, short_runner.main)
        self.assertIs(
            residual_recovery.continue_training,
            short_runner.continue_training,
        )

    def test_continue_training_restores_state_and_overrides_only_learning_rate(self) -> None:
        model = TrackingModel()
        source_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
        )
        optimizer_state = source_optimizer.state_dict()
        real_adamw = torch.optim.AdamW
        optimizers = []

        def build_optimizer(*args, **kwargs):
            optimizer = real_adamw(*args, **kwargs)
            optimizers.append(optimizer)
            return optimizer

        def loss_for_batch(model_arg, batch, _device):
            return ((model_arg.weight - batch["target"]) ** 2).mean()

        batches = [
            {"target": torch.ones(4)},
            {"target": torch.full((4,), 2.0)},
        ]
        with mock.patch.object(short_runner, "set_seed") as set_seed:
            with mock.patch.object(short_runner, "lm_loss", side_effect=loss_for_batch):
                with mock.patch.object(
                    short_runner.torch.optim,
                    "AdamW",
                    side_effect=build_optimizer,
                ):
                    result = short_runner.continue_training(
                        model,
                        optimizer_state,
                        batches,
                        learning_rate=2e-4,
                        seed=1042,
                        device="cpu",
                    )

        set_seed.assert_called_once_with(1042)
        self.assertEqual(len(optimizers), 1)
        self.assertEqual(optimizers[0].param_groups[0]["lr"], 2e-4)
        self.assertNotIn("initial_lr", optimizers[0].param_groups[0])
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["learning_rate"], 2e-4)
        self.assertEqual(len(result["train_losses"]), 2)
        self.assertGreaterEqual(result["seconds"], 0.0)
        self.assertTrue(model.training)
        self.assertIsNone(model.weight.grad)

    def run_gate(self, continuation_steps: int):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        reference_path = root / "reference.pt"
        current_path = root / "current.pt"
        reference_path.write_bytes(b"reference")
        current_path.write_bytes(b"current")
        args = short_runner.parse_args(
            [
                "--reference-checkpoint",
                str(reference_path),
                "--current-checkpoint",
                str(current_path),
                "--data-dir",
                str(root / "data"),
                "--output-dir",
                str(root / "results"),
                "--eval-batches",
                "1",
                "--continuation-steps",
                str(continuation_steps),
                "--device",
                "cpu",
            ]
        )
        reference_state = {"weight": torch.ones(4)}
        current_state = {"weight": torch.full((4,), 2.0)}
        delta = {"weight": torch.ones(4)}
        scope = ShortResidualScope(
            layers=[["weight"]],
            eligible_names=["weight"],
            layer_sizes=[4],
            eligible_count=4,
            model_count=4,
            target_pruned=1,
            prune_ratio=0.30,
            delta=delta,
            magnitude_scores=delta,
        )
        taylor_scores = {"weight": torch.arange(1.0, 5.0)}
        masks = {
            method: {"weight": torch.tensor([False, True, True, True])}
            for method in SHORT_GATE_METHODS
        }
        method_diagnostics = {
            method: {"pruned": 1, "proxy_cost": float(index + 1)}
            for index, method in enumerate(SHORT_GATE_METHODS)
        }
        allocation = {"seconds": 0.01}
        model = TrackingModel()
        snapshots = []
        restore_order = []
        continuation_markers = []

        def write_snapshot(_path, payload):
            snapshots.append(copy.deepcopy(payload))

        def load_batches(path, _tokenizer, _batch_size, _seq_length, count, offset=0):
            prefix = 100 if path.name == "valid.txt" else offset
            return [{"tokens": torch.tensor([prefix + index])} for index in range(count)]

        mask_names = {id(method_masks): method for method, method_masks in masks.items()}

        def restore(model_arg, _current, _reference, method_masks, _device):
            method = mask_names[id(method_masks)]
            restore_order.append(method)
            model_arg.loaded_marker = method

        def continue_trajectory(model_arg, _state, batches, lr, seed, device):
            continuation_markers.append(model_arg.loaded_marker)
            return {
                "steps": len(batches),
                "learning_rate": lr,
                "seconds": 0.02,
                "train_losses": [1.0] * len(batches),
                "seed": seed,
                "device": device,
            }

        gpt2_module = types.ModuleType("dacp.models.gpt2")
        get_model = mock.Mock(return_value=model)
        gpt2_module.get_gpt2_medium = get_model
        data_loader_module = types.ModuleType("dacp.utils.data_loader")
        load_tokenizer = mock.Mock(return_value=object())
        data_loader_module._load_gpt2_tokenizer = load_tokenizer

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(
                    sys.modules,
                    {
                        "dacp.models.gpt2": gpt2_module,
                        "dacp.utils.data_loader": data_loader_module,
                    },
                )
            )
            stack.enter_context(
                mock.patch.object(short_runner, "parse_args", return_value=args)
            )
            stack.enter_context(
                mock.patch.object(
                    short_runner.torch.cuda,
                    "is_available",
                    return_value=False,
                )
            )
            stack.enter_context(mock.patch.object(short_runner, "configure_hf_offline"))
            stack.enter_context(mock.patch.object(short_runner, "set_seed"))
            stack.enter_context(
                mock.patch.object(
                    short_runner,
                    "write_json",
                    side_effect=write_snapshot,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    short_runner,
                    "sha256_file",
                    side_effect=lambda path: path.stem,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    short_runner,
                    "checkpoint_state",
                    side_effect=lambda path: (
                        reference_state if path == reference_path else current_state
                    ),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    short_runner,
                    "load_token_batches",
                    side_effect=load_batches,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    short_runner,
                    "batch_hash",
                    side_effect=lambda batches: f"hash-{len(batches)}",
                )
            )
            stack.enter_context(
                mock.patch.object(
                    short_runner,
                    "build_short_residual_scope",
                    return_value=scope,
                )
            )
            evaluate = stack.enter_context(
                mock.patch.object(
                    short_runner,
                    "evaluate_lm",
                    side_effect=lambda *_args: {
                        "loss": 1.0,
                        "perplexity": 2.0,
                    },
                )
            )
            stack.enter_context(mock.patch.object(short_runner, "reset_peak_memory"))
            stack.enter_context(
                mock.patch.object(
                    short_runner,
                    "compute_block_taylor_scores",
                    return_value=(taylor_scores, {"seconds": 0.03}),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    short_runner,
                    "peak_memory_bytes",
                    return_value=123,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    short_runner,
                    "build_short_gate_masks",
                    return_value=(masks, allocation),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    short_runner,
                    "diagnose_short_gate_masks",
                    return_value=method_diagnostics,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    short_runner,
                    "restore_with_mask",
                    side_effect=restore,
                )
            )
            optimizer_state = stack.enter_context(
                mock.patch.object(
                    short_runner,
                    "checkpoint_optimizer_state",
                    return_value={"optimizer": "state"},
                )
            )
            continue_mock = stack.enter_context(
                mock.patch.object(
                    short_runner,
                    "continue_training",
                    side_effect=continue_trajectory,
                )
            )
            stack.enter_context(mock.patch.object(short_runner, "empty_device_cache"))
            stack.enter_context(mock.patch("builtins.print"))
            short_runner.main()

        return {
            "snapshots": snapshots,
            "restore_order": restore_order,
            "continuation_markers": continuation_markers,
            "evaluate_calls": evaluate.call_count,
            "continue_calls": continue_mock.call_args_list,
            "optimizer_state_calls": optimizer_state.call_count,
            "get_model_calls": get_model.call_args_list,
            "load_tokenizer_calls": load_tokenizer.call_count,
        }

    def test_main_preserves_incremental_write_and_method_order_without_continuation(self) -> None:
        run = self.run_gate(continuation_steps=0)
        snapshots = run["snapshots"]

        self.assertEqual(len(snapshots), 10)
        self.assertEqual(snapshots[0]["status"], "started")
        self.assertEqual(snapshots[0]["methods"], {})
        self.assertIn("parameter_scope", snapshots[1])
        self.assertIn("baselines", snapshots[2])
        self.assertIn("scoring", snapshots[3])
        self.assertEqual(tuple(snapshots[4]["methods"]), SHORT_GATE_METHODS)
        for completed, snapshot in enumerate(snapshots[5:9], start=1):
            evaluated = [
                method
                for method in SHORT_GATE_METHODS
                if "evaluation" in snapshot["methods"][method]
            ]
            self.assertEqual(evaluated, list(SHORT_GATE_METHODS[:completed]))
        self.assertEqual(snapshots[-1]["status"], "complete")
        self.assertEqual(run["restore_order"], list(SHORT_GATE_METHODS))
        self.assertEqual(run["evaluate_calls"], 2 + len(SHORT_GATE_METHODS))
        self.assertEqual(run["optimizer_state_calls"], 0)
        self.assertEqual(run["continue_calls"], [])

    def test_main_preserves_continuation_order_seed_and_trajectory_reset(self) -> None:
        run = self.run_gate(continuation_steps=1)
        snapshots = run["snapshots"]

        self.assertEqual(len(snapshots), 15)
        self.assertEqual(
            run["restore_order"],
            [*SHORT_GATE_METHODS, *SHORT_GATE_METHODS],
        )
        self.assertEqual(
            run["continuation_markers"],
            ["current", *SHORT_GATE_METHODS],
        )
        self.assertEqual(run["optimizer_state_calls"], 1)
        self.assertEqual(len(run["continue_calls"]), 1 + len(SHORT_GATE_METHODS))
        for call in run["continue_calls"]:
            self.assertEqual(call.args[4], 1042)
            self.assertEqual(call.args[5], "cpu")
            self.assertEqual(len(call.args[2]), 1)
        self.assertIn("continuation_no_compression", snapshots[9])
        for completed, snapshot in enumerate(snapshots[10:14], start=1):
            continued = [
                method
                for method in SHORT_GATE_METHODS
                if "continuation" in snapshot["methods"][method]
            ]
            self.assertEqual(continued, list(SHORT_GATE_METHODS[:completed]))
        self.assertEqual(snapshots[-1]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
