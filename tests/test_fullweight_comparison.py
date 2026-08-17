import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch

import experiments.scripts.run_inshrinkerator_like_search as search_script
import experiments.scripts.run_paper_fullweight_comparison as comparison_script
from experiments.lib.fullweight_comparison import (
    DACP_FULLWEIGHT_METHOD_ID,
    INSHRINKERATOR_STYLE_FULLWEIGHT_METHOD_ID,
    apply_zero_masks_to_state,
    build_dacp_fullweight_mask,
    build_inshrinkerator_style_fullweight_mask,
    validate_search_result_provenance,
)
from experiments.lib.residual_runtime import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def _parameters() -> dict[str, torch.Tensor]:
    return {
        "transformer.h.0.attn.c_attn.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "transformer.h.0.mlp.c_fc.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    }


def _layers() -> list[list[str]]:
    return [
        ["transformer.h.0.attn.c_attn.weight"],
        ["transformer.h.0.mlp.c_fc.weight"],
    ]


def _search_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "model": "gpt2-medium",
        "dataset": "wikitext103",
        "seed": 42,
        "checkpoint": str((ROOT / "checkpoint.pt").resolve()),
        "checkpoint_sha256": "a" * 64,
        "source_digest": "b" * 64,
        "scoring_batch_sha256": "c" * 64,
        "evaluation_batch_sha256": "d" * 64,
        "scoring_batch_count": 2,
        "evaluation_batch_count": 2,
        "protect_fraction": 0.005,
        "best_metric": "magnitude",
        "per_type_ratios": {"attn": 0.75, "mlp": 0.0},
        "actual_global_ratio": 0.5,
        "quality_drop": 1.0,
    }
    payload.update(overrides)
    return payload


class TestFullWeightComparison(unittest.TestCase):
    def test_both_methods_share_exact_target_and_direct_zeroing(self) -> None:
        parameters = _parameters()
        layers = _layers()
        taylor_scores = {name: value.abs() for name, value in parameters.items()}
        masks = {name: torch.tensor([[True, False], [True, False]]) for name in parameters}
        allocation = {
            "target_pruned": 4,
            "weibull_fits": [{"layer": index} for index in range(2)],
            "weibull_layer_counts": [2, 2],
            "weibull": {"fallback": "toy"},
        }
        search = _search_payload()
        with mock.patch(
            "experiments.lib.fullweight_comparison.build_weibull_mask",
            return_value=(masks, allocation),
        ):
            dacp = build_dacp_fullweight_mask(parameters, layers, taylor_scores, 0.5)
        style = build_inshrinkerator_style_fullweight_mask(
            parameters,
            layers,
            taylor_scores,
            search,
            0.5,
        )

        self.assertEqual(dacp.method_id, DACP_FULLWEIGHT_METHOD_ID)
        self.assertEqual(dacp.fidelity, "native")
        self.assertEqual(dacp.target_pruned, 4)
        self.assertEqual(dacp.actual_pruned, 4)
        self.assertEqual(style.method_id, INSHRINKERATOR_STYLE_FULLWEIGHT_METHOD_ID)
        self.assertEqual(style.fidelity, "style")
        self.assertEqual(style.target_pruned, dacp.target_pruned)
        self.assertEqual(style.actual_pruned, dacp.actual_pruned)
        self.assertEqual(dacp.evidence["probe"], "current_full_weight_theta")
        self.assertTrue(dacp.evidence["allocation_provenance"]["direct_zeroing"])

        state = {name: value.clone() for name, value in parameters.items()}
        zeroed = apply_zero_masks_to_state(state, style.masks)
        self.assertTrue(all(torch.equal(state[name], parameters[name]) for name in state))
        self.assertTrue(
            all(
                torch.equal(
                    zeroed[name][~style.masks[name]],
                    torch.zeros_like(zeroed[name][~style.masks[name]]),
                )
                for name in state
            )
        )

    def test_per_type_projection_and_metric_routing(self) -> None:
        parameters = _parameters()
        scores = {name: value.clone() for name, value in parameters.items()}
        search = _search_payload(
            best_metric="sensitivity",
            per_type_ratios={"attn": 1.0, "mlp": 0.0},
            actual_global_ratio=0.5,
        )
        assembly = build_inshrinkerator_style_fullweight_mask(
            parameters,
            _layers(),
            scores,
            search,
            0.5,
        )

        self.assertEqual(assembly.score_kind, "first_order")
        self.assertEqual(assembly.allocation_kind, "per_type")
        self.assertEqual(sum(assembly.evidence["per_type_counts"].values()), 4)
        self.assertEqual(assembly.evidence["search_best_metric"], "sensitivity")
        self.assertEqual(
            assembly.evidence["allocation_provenance"]["projection"],
            "bounded_largest_remainder_counts",
        )

    def test_scoring_probe_is_current_full_weight_theta(self) -> None:
        model = torch.nn.Linear(2, 2, bias=False)
        parameters = {"weight": model.weight.detach().clone()}
        layers = [["weight"]]
        blocks = [["weight"]]
        batches = [{"input_ids": torch.ones(1, 2), "labels": torch.ones(1, 2)}]
        first_order = {"weight": torch.ones_like(parameters["weight"])}
        taylor = {"weight": torch.full_like(parameters["weight"], 2.0)}
        with (
            mock.patch.object(
                comparison_script,
                "compute_block_first_order_scores",
                return_value=(first_order, {"score_kind": "first_order"}),
            ) as first_call,
            mock.patch.object(
                comparison_script,
                "compute_block_taylor_scores",
                return_value=(taylor, {"score_kind": "taylor_hvp"}),
            ) as taylor_call,
        ):
            families, _ = comparison_script._compute_fullweight_score_families(
                model,
                batches,
                layers,
                blocks,
                parameters,
                "cpu",
                "lm",
                "gpt2",
            )

        self.assertIs(first_call.call_args.args[3], parameters)
        self.assertIs(taylor_call.call_args.args[3], parameters)
        self.assertTrue(torch.equal(families["magnitude"]["weight"], parameters["weight"].abs()))

    def test_malformed_or_stale_search_provenance_fails_closed(self) -> None:
        checkpoint = ROOT / "checkpoint.pt"
        expected = {
            "model": "gpt2-medium",
            "dataset": "wikitext103",
            "seed": 42,
            "checkpoint": checkpoint,
            "checkpoint_sha256": "a" * 64,
            "source_digest": "b" * 64,
            "scoring_batch_sha256": "c" * 64,
            "evaluation_batch_sha256": "d" * 64,
            "scoring_batch_count": 2,
            "evaluation_batch_count": 2,
            "protect_fraction": 0.005,
        }
        for bad in ({"best_metric": "magnitude"}, {**_search_payload(), "best_metric": "unknown"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_search_result_provenance(bad, **expected)
        with self.assertRaisesRegex(ValueError, "source digest"):
            validate_search_result_provenance(
                _search_payload(source_digest="e" * 64),
                **expected,
            )

    def test_suite_prevents_overwrite_and_detects_resume_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite_path = root / "suite_manifest.json"
            template = {
                "schema_version": 1,
                "experiment_scope": "full_weight_pruning_only",
                "plan": {"ratios": ["p0p5"]},
                "provenance": {"source_digest": "a" * 64},
                "planned_ratios": ["p0p5"],
                "completed_ratios": {},
            }
            suite = comparison_script.FullWeightSuiteStore.open(
                suite_path,
                template,
                resume=False,
            )
            with self.assertRaises(FileExistsError):
                comparison_script.FullWeightSuiteStore.open(
                    suite_path,
                    template,
                    resume=False,
                )
            result_path = root / "ratio.json"
            result_payload = {
                "schema_version": 1,
                "experiment_scope": "full_weight_pruning_only",
                "status": "complete",
                "config": {"ratio": 0.5},
                "provenance": template["provenance"],
                "results": [{}, {}],
            }
            write_json(result_path, result_payload)
            suite.record_completed("p0p5", result_path)
            result_path.write_text(json.dumps({"tampered": True}), encoding="utf-8")
            resumed = comparison_script.FullWeightSuiteStore.open(
                suite_path,
                template,
                resume=True,
            )
            with self.assertRaisesRegex(ValueError, "digest"):
                resumed.completed_path("p0p5", result_payload)

    def test_fullweight_manifest_and_both_default_dry_runs_skip_runtime(self) -> None:
        manifest_path = ROOT / "experiments/configs/paper_fullweight_inshrinkerator.yaml"
        output = io.StringIO()
        with (
            mock.patch.object(
                comparison_script, "load_training_checkpoint", side_effect=AssertionError
            ),
            mock.patch.object(comparison_script, "get_data_loaders", side_effect=AssertionError),
            mock.patch.object(comparison_script, "load_model", side_effect=AssertionError),
            mock.patch.object(comparison_script, "_validate_device", side_effect=AssertionError),
            redirect_stdout(output),
        ):
            comparison_script.main(["--manifest", str(manifest_path), "--dry-run"])
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["experiment_scope"], "full_weight_pruning_only")
        self.assertEqual(plan["seed"], 42)
        self.assertEqual(plan["direct_weight_prune_ratios"], [0.2, 0.3, 0.4])

        search_output = io.StringIO()
        with (
            mock.patch.object(search_script, "load_model", side_effect=AssertionError),
            mock.patch.object(search_script, "get_data_loaders", side_effect=AssertionError),
            mock.patch.object(search_script, "sha256_file", side_effect=AssertionError),
            redirect_stdout(search_output),
        ):
            search_script.main(
                [
                    "--model",
                    "gpt2-medium",
                    "--dataset",
                    "wikitext103",
                    "--checkpoint",
                    str(ROOT / "missing.pt"),
                    "--dry-run",
                ]
            )
        self.assertEqual(json.loads(search_output.getvalue())["seed"], 42)


if __name__ == "__main__":
    unittest.main()
