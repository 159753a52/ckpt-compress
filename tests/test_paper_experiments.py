import hashlib
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

import experiments.lib.paper_runner as paper_runner
from experiments.lib.paper_baselines import resolve_method_contracts, validate_claim_gate
from experiments.lib.paper_manifest import PaperManifest, PaperWorkload, load_paper_manifest
from experiments.lib.paper_results import SCHEMA_VERSION, JobResultStore, SuiteResultStore
from experiments.lib.paper_runner import (
    PaperJob,
    aggregate_job_results,
    plan_jobs,
    run_method_trajectory,
)
from experiments.lib.residual_training import build_optimizer, partition_repeated_seed_batches
from experiments.scripts import run_paper_experiments as paper_script
from experiments.scripts.generate_paper_figures import (
    generate_allocation_figure,
    generate_distribution_fit_figure,
    load_completed_suite,
    load_fit_records,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-08-13T00:00:00+00:00"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_provenance() -> dict[str, object]:
    return {
        "manifest": "paper.yaml",
        "manifest_sha256": _digest("manifest"),
        "git_commit": "a" * 40,
        "git_dirty": False,
        "dirty_source_paths": [],
        "source_file_count": 1,
        "source_state_sha256": _digest("source"),
    }


class TinyPaperModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))


class TinyPaperLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        block = torch.nn.Module()
        block.weight = torch.nn.Parameter(torch.eye(4))
        transformer = torch.nn.Module()
        transformer.h = torch.nn.ModuleList([block])
        self.transformer = transformer

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
        return one_hot @ self.transformer.h[0].weight


def tiny_workload() -> PaperWorkload:
    return PaperWorkload(
        name="tiny",
        model="gpt2-medium",
        model_family="gpt2",
        dataset="wikitext103",
        task_type="lm",
        checkpoint=Path("checkpoint.pt"),
        checkpoint_step=1000,
        data_dir=Path("data"),
        metric="perplexity",
        seeds=(42, 43),
        prune_ratios=(0.5,),
        max_layer_ratio=0.95,
        recovery_counts=(1, 3),
        batch_size=1,
        seq_length=2,
        total_steps=4,
        hvp_batches=1,
        eval_batches=1,
        train_pool_batches=7,
        learning_rate=1e-3,
    )


def _write_complete_dacp_suite(root: Path) -> tuple[Path, Path]:
    workload = PaperWorkload(
        **{
            **tiny_workload().__dict__,
            "seeds": (42,),
            "recovery_counts": (1,),
            "total_steps": 2,
            "hvp_batches": 1,
            "train_pool_batches": 3,
        }
    )
    job = PaperJob(workload, 0.5, 1, (42,))
    pool = [
        {
            "input_ids": torch.tensor([[index % 4, (index + 1) % 4, (index + 2) % 4]]),
            "labels": torch.tensor([[index % 4, (index + 1) % 4, (index + 2) % 4]]),
        }
        for index in range(3)
    ]
    batch_plan = partition_repeated_seed_batches(pool, 2, 1, 1, seed=42)
    model = TinyPaperLanguageModel()
    optimizer_state = torch.optim.AdamW(model.parameters(), lr=1e-3).state_dict()
    contracts = resolve_method_contracts(["dacp"])
    provenance = _source_provenance()
    context = paper_script.WorkloadContext(
        checkpoint_path=root / "checkpoint.pt",
        checkpoint_sha256=_digest("checkpoint"),
        checkpoint=SimpleNamespace(step=workload.checkpoint_step),
        training_pool=pool,
        eval_batches=pool[:1],
        data_identity={},
    )

    job_path = root / f"{job.job_id}.json"
    job_template = paper_script._job_template(job, contracts, provenance, context)
    job_template["started_at"] = NOW
    store = JobResultStore.open(
        job_path,
        job_template,
        metric=workload.metric,
        resume=False,
    )
    store.ensure_batch_plan(
        42,
        {
            "selected_pool_indices": batch_plan.selected_pool_indices,
            **batch_plan.data_hashes(),
        },
    )
    record = run_method_trajectory(
        job,
        contracts[0],
        42,
        batch_plan,
        TinyPaperLanguageModel,
        model.state_dict(),
        optimizer_state,
        pool[:1],
        "cpu",
    )
    store.append_result(record)
    store.finalize(
        aggregate_job_results([record], workload.metric),
        finished_at=NOW,
    )

    suite_path = root / "suite_manifest.json"
    suite = SuiteResultStore.open(
        suite_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "started",
            "started_at": NOW,
            "provenance": provenance,
            "plan": {
                "jobs": [job.to_result_dict()],
                "methods": [contract.to_result_dict() for contract in contracts],
            },
            "planned_jobs": [job.job_id],
            "completed_jobs": {},
            "data_identities": {},
        },
        resume=False,
    )
    suite.record_completed_job(job.job_id, job_path)
    suite.finalize(finished_at=NOW)
    return suite_path, job_path


class TestPaperExperiments(unittest.TestCase):
    def test_repository_manifest_declares_exact_paper_workloads(self) -> None:
        manifest = load_paper_manifest(ROOT / "experiments/configs/paper_experiments.yaml")

        self.assertEqual(
            [workload.name for workload in manifest.workloads],
            [
                "gpt2_medium_wikitext103",
                "gpt2_large_wikitext103",
                "bert_large_mnli",
                "pythia_410m_alpaca",
                "pythia_1b_alpaca",
            ],
        )
        self.assertEqual(
            [contract.name for contract in manifest.methods],
            ["no_compression", "excp_style", "inshrinkerator_style", "dacp"],
        )
        self.assertEqual(
            manifest.claim_gates["full_end_to_end_baseline_claim"]["enabled"],
            False,
        )

    def test_full_claim_rejects_style_adapters(self) -> None:
        contracts = resolve_method_contracts(["excp_style", "inshrinkerator_style"])
        with self.assertRaisesRegex(ValueError, "style adapters"):
            validate_claim_gate(
                contracts,
                {
                    "baseline_owners": ["excp", "inshrinkerator"],
                    "required_baseline_fidelity": "full",
                },
            )
        contracts = resolve_method_contracts(["excp_style"])
        for owners in ("excp", ["excp", "excp"], [""]):
            with self.subTest(owners=owners), self.assertRaises(ValueError):
                validate_claim_gate(
                    contracts,
                    {
                        "baseline_owners": owners,
                        "required_baseline_fidelity": "style",
                    },
                )

    def test_main_rejects_methods_that_bypass_an_enabled_claim_gate(self) -> None:
        manifest = PaperManifest(
            path=Path("manifest.yaml"),
            schema_version=1,
            methods=tuple(resolve_method_contracts(["no_compression", "dacp"])),
            claim_gates={
                "matched_scoring_main_table": {
                    "baseline_owners": ["excp", "inshrinkerator"],
                    "required_baseline_fidelity": "style",
                }
            },
            workloads=(tiny_workload(),),
        )

        with (
            mock.patch.object(paper_script, "load_paper_manifest", return_value=manifest),
            self.assertRaisesRegex(ValueError, "enabled claim gate"),
        ):
            paper_script.main(["--methods", "dacp", "--dry-run"])

    def test_main_accepts_gate_satisfying_methods_and_gate_free_subsets(self) -> None:
        gated_manifest = PaperManifest(
            path=Path("manifest.yaml"),
            schema_version=1,
            methods=tuple(
                resolve_method_contracts(
                    ["no_compression", "excp_style", "inshrinkerator_style", "dacp"]
                )
            ),
            claim_gates={
                "matched_scoring_main_table": {
                    "baseline_owners": ["excp", "inshrinkerator"],
                    "required_baseline_fidelity": "style",
                }
            },
            workloads=(tiny_workload(),),
        )
        with mock.patch.object(paper_script, "load_paper_manifest", return_value=gated_manifest):
            paper_script.main(["--methods", "excp_style,inshrinkerator_style", "--dry-run"])

        ungated_manifest = PaperManifest(
            path=Path("manifest.yaml"),
            schema_version=1,
            methods=tuple(resolve_method_contracts(["dacp"])),
            claim_gates={},
            workloads=(tiny_workload(),),
        )
        with mock.patch.object(paper_script, "load_paper_manifest", return_value=ungated_manifest):
            paper_script.main(["--methods", "dacp", "--dry-run"])

    def test_repeated_partition_has_k_plus_one_train_segments(self) -> None:
        pool = [
            {"input_ids": torch.tensor([[index]]), "labels": torch.tensor([[index]])}
            for index in range(10)
        ]
        plan = partition_repeated_seed_batches(
            pool, total_steps=5, num_recoveries=2, hvp_batches=2, seed=42
        )

        self.assertEqual([len(segment) for segment in plan.training_segments], [2, 2, 1])
        self.assertEqual([len(batches) for batches in plan.scoring_batches], [2, 2])
        self.assertEqual(len(plan.selected_pool_indices), 9)
        self.assertEqual(len(set(plan.selected_pool_indices)), 9)
        self.assertEqual(len(plan.data_hashes()["training_segments"]), 3)

    def test_job_planner_only_allows_declared_slices(self) -> None:
        workload = tiny_workload()
        jobs = plan_jobs(
            [workload],
            prune_ratios=[0.5],
            recovery_counts=[3],
            seeds=[43],
        )
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].job_id, PaperJob(workload, 0.5, 3, (43,)).job_id)
        with self.assertRaisesRegex(ValueError, "not declared"):
            plan_jobs([workload], prune_ratios=[0.7])
        with self.assertRaisesRegex(ValueError, "Duplicate requested seeds"):
            plan_jobs([workload], seeds=[42, 42])
        with self.assertRaisesRegex(ValueError, "Unknown workloads"):
            plan_jobs([workload], workload_names=["missing"])
        with self.assertRaisesRegex(ValueError, "recovery count is not declared"):
            plan_jobs([workload], recovery_counts=[2])
        with self.assertRaisesRegex(ValueError, "seed is not declared"):
            plan_jobs([workload], seeds=[99])

        mismatched = [
            {"method": "dacp", "seed": 42, "final": {"perplexity": 1.0}},
            {"method": "excp_style", "seed": 43, "final": {"perplexity": 2.0}},
        ]
        with self.assertRaisesRegex(ValueError, "identical seed sets"):
            aggregate_job_results(mismatched, "perplexity")
        with self.assertRaisesRegex(ValueError, "empty result set"):
            aggregate_job_results([], "perplexity")

    def test_manifest_rejects_task_metric_and_non_finite_numeric_values(self) -> None:
        base = {
            "schema_version": 1,
            "methods": ["dacp"],
            "workloads": [
                {
                    "name": "tiny",
                    "model": "gpt2-medium",
                    "dataset": "wikitext103",
                    "checkpoint": "checkpoints/tiny.pt",
                    "data_dir": "data/wikitext103",
                    "metric": "perplexity",
                    "seeds": [42],
                    "prune_ratios": [0.5],
                    "recovery_counts": [1],
                    "total_steps": 2,
                    "hvp_batches": 1,
                    "eval_batches": 1,
                    "batch_size": 1,
                    "seq_length": 4,
                    "learning_rate": 1e-3,
                }
            ],
        }
        cases = (
            ("metric", "accuracy", "unsupported for task type"),
            ("learning_rate", float("nan"), "finite number"),
            ("learning_rate", 0.0, "must be positive"),
            ("seeds", [True], "invalid value"),
            ("seeds", [1.5], "invalid value"),
            ("prune_ratios", [float("inf")], "finite values"),
            ("name", "../escape", "contain only letters"),
            ("total_steps", 1, r"K\+1 training segments"),
        )
        for key, value, message in cases:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as temp_dir:
                payload = json.loads(json.dumps(base))
                payload["workloads"][0][key] = value
                path = Path(temp_dir) / "manifest.yaml"
                path.write_text(yaml.safe_dump(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_paper_manifest(path)

        for schema_version in (True, 2):
            with (
                self.subTest(schema_version=schema_version),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                payload = json.loads(json.dumps(base))
                payload["schema_version"] = schema_version
                path = Path(temp_dir) / "manifest.yaml"
                path.write_text(yaml.safe_dump(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    load_paper_manifest(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            payload = json.loads(json.dumps(base))
            payload["claim_gates"] = {
                "gate": {
                    "enabled": "false",
                    "baseline_owners": [],
                }
            }
            path = Path(temp_dir) / "manifest.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "enabled must be boolean"):
                load_paper_manifest(path)

    def test_no_compression_trajectory_runs_all_training_segments(self) -> None:
        workload = tiny_workload()
        job = PaperJob(workload, 0.5, 1, (42,))
        plan = partition_repeated_seed_batches(
            [
                {"input_ids": torch.tensor([[index]]), "labels": torch.tensor([[index]])}
                for index in range(5)
            ],
            total_steps=4,
            num_recoveries=1,
            hvp_batches=1,
            seed=42,
        )
        model = TinyPaperModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_calls = []
        scheduler_state = []

        def train_side_effect(model, optimizer, scheduler, batches, seed, device, task_type):
            train_calls.append((len(batches), seed, task_type))
            scheduler_state.append(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                    "t_max": scheduler.T_max,
                    "eta_min": scheduler.eta_min,
                    "last_epoch": scheduler.last_epoch,
                }
            )
            with torch.no_grad():
                model.weight.add_(len(batches))
            return {"steps": len(batches), "task_type": task_type}

        def evaluate_side_effect(model, batches, task_type, device):
            value = float(model.weight.item())
            return {"loss": value, "perplexity": value + 1.0}

        with (
            mock.patch(
                "experiments.lib.paper_runner.train_segment",
                side_effect=train_side_effect,
            ),
            mock.patch(
                "experiments.lib.paper_runner.evaluate_task",
                side_effect=evaluate_side_effect,
            ),
        ):
            result = run_method_trajectory(
                job,
                resolve_method_contracts(["no_compression"])[0],
                42,
                plan,
                TinyPaperModel,
                model.state_dict(),
                optimizer.state_dict(),
                [{"input_ids": torch.tensor([[0]]), "labels": torch.tensor([[0]])}],
                "cpu",
            )

        self.assertEqual(train_calls, [(2, 42, "lm"), (2, 1_000_042, "lm")])
        self.assertEqual(
            scheduler_state[0],
            {"lr": workload.learning_rate, "t_max": 4, "eta_min": 0.0, "last_epoch": 0},
        )
        self.assertEqual(len(result["cycles"]), 1)
        self.assertEqual(result["cycles"][0]["compression"], None)
        self.assertEqual(result["final"]["perplexity"], 5.0)
        json.dumps(result, allow_nan=False)

    def test_optimizer_build_does_not_share_checkpoint_state_tensors(self) -> None:
        source_model = torch.nn.Linear(2, 1)
        source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1e-3)
        source_model(torch.ones(1, 2)).sum().backward()
        source_optimizer.step()
        state = source_optimizer.state_dict()
        state_tensor = next(iter(state["state"].values()))["exp_avg"]
        before = state_tensor.clone()

        first_model = torch.nn.Linear(2, 1)
        second_model = torch.nn.Linear(2, 1)
        first = build_optimizer(first_model, state, learning_rate=1e-3)
        second = build_optimizer(second_model, state, learning_rate=1e-3)
        first_tensor = next(iter(first.state.values()))["exp_avg"]
        second_tensor = next(iter(second.state.values()))["exp_avg"]

        self.assertNotEqual(first_tensor.data_ptr(), state_tensor.data_ptr())
        self.assertNotEqual(second_tensor.data_ptr(), state_tensor.data_ptr())
        self.assertNotEqual(first_tensor.data_ptr(), second_tensor.data_ptr())
        first_model(torch.ones(1, 2)).sum().backward()
        first.step()
        self.assertTrue(torch.equal(state_tensor, before))
        self.assertTrue(torch.equal(second_tensor, before))

    def test_figures_require_completed_digest_verified_real_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fit_path = root / "fits.json"
            records = []
            for index in range(4):
                records.append(
                    {
                        "layer": f"block.{index}",
                        "n_params": 100 - index,
                        "weibull_ks": 0.1,
                        "gamma_ks": 0.2,
                        "lognormal_ks": 0.3,
                        "cdf": {
                            "x": [0.1, 0.2],
                            "empirical": [0.4, 0.8],
                            "weibull": [0.39, 0.79],
                            "gamma": [0.3, 0.7],
                            "lognormal": [0.35, 0.75],
                        },
                    }
                )
            fit_path.write_text(
                json.dumps({"schema_version": 1, "status": "complete", "records": records}),
                encoding="utf-8",
            )

            suite_path, job_path = _write_complete_dacp_suite(root)
            original_job = job_path.read_bytes()
            original_suite = suite_path.read_bytes()
            output_dir = root / "figures"

            distribution = generate_distribution_fit_figure(fit_path, output_dir)
            allocation = generate_allocation_figure(suite_path, output_dir)

            for output in (distribution, allocation):
                self.assertTrue(output.is_file())
                self.assertTrue(output.with_suffix(".png").is_file())
                self.assertTrue(output.with_suffix(".provenance.json").is_file())

            job_payload = json.loads(original_job)
            job_path.write_text(json.dumps({**job_payload, "tampered": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_completed_suite(suite_path)

            job_path.write_bytes(original_job)
            suite_path.write_bytes(original_suite)
            job_payload = json.loads(original_job)
            job_payload["method_contracts"][0]["fidelity"] = "full"
            job_payload["results"][0]["fidelity"] = "full"
            job_path.write_text(json.dumps(job_payload), encoding="utf-8")
            suite_payload = json.loads(original_suite)
            suite_payload["plan"]["methods"][0]["fidelity"] = "full"
            job_id = next(iter(suite_payload["completed_jobs"]))
            suite_payload["completed_jobs"][job_id]["sha256"] = hashlib.sha256(
                job_path.read_bytes()
            ).hexdigest()
            suite_path.write_text(json.dumps(suite_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "executable registry"):
                load_completed_suite(suite_path)

            job_path.write_bytes(original_job)
            suite_path.write_bytes(original_suite)
            job_payload = json.loads(original_job)
            job_payload["results"][0]["cycles"][0]["compression"]["mask"]["pruned"] = 0
            job_path.write_text(json.dumps(job_payload), encoding="utf-8")
            suite_payload = json.loads(suite_path.read_text(encoding="utf-8"))
            job_id = next(iter(suite_payload["completed_jobs"]))
            suite_payload["completed_jobs"][job_id]["sha256"] = hashlib.sha256(
                job_path.read_bytes()
            ).hexdigest()
            suite_path.write_text(json.dumps(suite_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected"):
                load_completed_suite(suite_path)

    def test_distribution_figure_rejects_duplicate_or_non_monotonic_cdf_evidence(self) -> None:
        base_record = {
            "layer": "block.0",
            "n_params": 100,
            "weibull_ks": 0.1,
            "gamma_ks": 0.2,
            "lognormal_ks": 0.3,
            "cdf": {
                "x": [0.1, 0.2],
                "empirical": [0.4, 0.8],
                "weibull": [0.39, 0.79],
                "gamma": [0.3, 0.7],
                "lognormal": [0.35, 0.75],
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fits.json"
            duplicates = [
                {**base_record, "layer": "block.0"},
                {**base_record, "layer": "block.0"},
                {**base_record, "layer": "block.2"},
                {**base_record, "layer": "block.3"},
            ]
            path.write_text(
                json.dumps({"schema_version": 1, "status": "complete", "records": duplicates}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate layer"):
                load_fit_records(path)

            malformed = [{**base_record, "layer": f"block.{index}"} for index in range(4)]
            malformed[0] = {
                **malformed[0],
                "cdf": {**malformed[0]["cdf"], "empirical": [0.8, 0.4]},
            }
            path.write_text(
                json.dumps({"schema_version": 1, "status": "complete", "records": malformed}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-decreasing"):
                load_fit_records(path)

            inconsistent = [
                {
                    **base_record,
                    "layer": f"block.{index}",
                    "weibull_ks": 0.0,
                    "gamma_ks": 0.2,
                    "lognormal_ks": 0.3,
                }
                for index in range(4)
            ]
            path.write_text(
                json.dumps({"schema_version": 1, "status": "complete", "records": inconsistent}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "CDF grid difference"):
                load_fit_records(path)

            incomplete = [
                {
                    **base_record,
                    "layer": f"block.{index}",
                    "weibull_ks": 0.1,
                    "gamma_ks": 0.2,
                    "lognormal_ks": 0.3,
                }
                for index in range(4)
            ]
            del incomplete[0]["cdf"]["gamma"]
            path.write_text(
                json.dumps({"schema_version": 1, "status": "complete", "records": incomplete}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing CDF competitors"):
                load_fit_records(path)

    def test_job_aggregate_preserves_paired_seed_deltas(self) -> None:
        records = [
            {"method": "dacp", "seed": 42, "final": {"perplexity": 10.0}},
            {"method": "excp_style", "seed": 42, "final": {"perplexity": 11.0}},
            {"method": "dacp", "seed": 43, "final": {"perplexity": 12.0}},
            {"method": "excp_style", "seed": 43, "final": {"perplexity": 11.5}},
        ]

        aggregate = aggregate_job_results(records, "perplexity")

        comparison = aggregate["paired_comparisons"]["dacp_vs_excp_style"]
        self.assertEqual(comparison["dacp_minus_baseline"], [-1.0, 0.5])
        self.assertEqual(comparison["wins"], 1)
        self.assertEqual(comparison["pairs"], 2)

    def test_dacp_mask_path_forwards_declared_layer_cap(self) -> None:
        scores = {"weight": torch.tensor([0.1, 0.2])}
        masks = {"weight": torch.tensor([True, False])}
        with (
            mock.patch.object(
                paper_runner,
                "compute_block_taylor_scores",
                return_value=(scores, {"score_kind": "taylor_hvp"}),
            ),
            mock.patch.object(
                paper_runner,
                "build_weibull_mask",
                return_value=(masks, {"target_pruned": 1}),
            ) as build_mask,
        ):
            actual_masks, _ = paper_runner._score_and_mask(
                resolve_method_contracts(["dacp"])[0],
                TinyPaperModel(),
                [{"input_ids": torch.tensor([[0]]), "labels": torch.tensor([[0]])}],
                [["weight"]],
                [["weight"]],
                {"weight": torch.tensor([0.1])},
                prune_ratio=0.5,
                max_layer_ratio=0.91,
                task_type="lm",
                model_family="gpt2",
                device="cpu",
                distributed_moments=False,
            )

        self.assertIs(actual_masks, masks)
        self.assertEqual(build_mask.call_args.kwargs["max_layer_ratio"], 0.91)

    def test_compressed_method_paths_execute_real_scoring_and_recovery(self) -> None:
        workload = PaperWorkload(
            **{
                **tiny_workload().__dict__,
                "total_steps": 2,
                "hvp_batches": 1,
                "train_pool_batches": 3,
                "recovery_counts": (1,),
            }
        )
        job = PaperJob(workload, 0.5, 1, (42,))
        pool = [
            {
                "input_ids": torch.tensor([[index % 4, (index + 1) % 4, (index + 2) % 4]]),
                "labels": torch.tensor([[index % 4, (index + 1) % 4, (index + 2) % 4]]),
            }
            for index in range(3)
        ]
        plan = partition_repeated_seed_batches(pool, 2, 1, 1, seed=42)
        template = TinyPaperLanguageModel()
        optimizer_state = torch.optim.AdamW(template.parameters(), lr=1e-3).state_dict()

        contracts = resolve_method_contracts(["excp_style", "inshrinkerator_style", "dacp"])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = paper_script.WorkloadContext(
                checkpoint_path=root / "checkpoint.pt",
                checkpoint_sha256=_digest("checkpoint"),
                checkpoint=SimpleNamespace(step=workload.checkpoint_step),
                training_pool=pool,
                eval_batches=pool[:1],
                data_identity={},
            )
            provenance = _source_provenance()
            job_template = paper_script._job_template(job, contracts, provenance, context)
            job_template["started_at"] = NOW
            store = JobResultStore.open(
                root / "job.json",
                job_template,
                metric=workload.metric,
                resume=False,
            )
            store.ensure_batch_plan(
                42,
                {
                    "selected_pool_indices": plan.selected_pool_indices,
                    **plan.data_hashes(),
                },
            )

            for contract in contracts:
                with self.subTest(method=contract.name):
                    result = run_method_trajectory(
                        job,
                        contract,
                        42,
                        plan,
                        TinyPaperLanguageModel,
                        template.state_dict(),
                        optimizer_state,
                        pool[:1],
                        "cpu",
                    )

                    self.assertEqual(len(result["cycles"]), 1)
                    mask = result["cycles"][0]["compression"]["mask"]
                    self.assertEqual(mask["eligible_parameters"], 16)
                    self.assertEqual(mask["pruned"], 8)
                    self.assertTrue(math.isfinite(result["final"]["perplexity"]))
                    store.append_result(result)

            records = list(paper_script.records_from(store))
            store.finalize(
                aggregate_job_results(records, workload.metric),
                finished_at=NOW,
            )
            reopened = JobResultStore.open(
                root / "job.json",
                job_template,
                metric=workload.metric,
                resume=True,
            )
            self.assertTrue(reopened.complete)

    def test_main_resume_skips_verified_completed_method(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = root / "checkpoint.pt"
            checkpoint_path.write_bytes(b"trusted checkpoint fixture")
            workload = PaperWorkload(
                **{
                    **tiny_workload().__dict__,
                    "checkpoint": checkpoint_path,
                    "data_dir": root,
                    "seeds": (42,),
                    "prune_ratios": (0.5, 0.7),
                    "recovery_counts": (1,),
                }
            )
            contracts = tuple(resolve_method_contracts(["no_compression", "dacp"]))
            manifest = PaperManifest(
                path=root / "manifest.yaml",
                schema_version=1,
                methods=contracts,
                claim_gates={},
                workloads=(workload,),
            )
            batches = [
                {
                    "input_ids": torch.tensor([[index, index + 1]]),
                    "labels": torch.tensor([[index, index + 1]]),
                }
                for index in range(8)
            ]
            calls = []
            checkpoint_loads = []
            data_loads = []

            def load_checkpoint(path):
                checkpoint_loads.append(path)
                return SimpleNamespace(model_state={}, optimizer_state={}, step=1000)

            def load_data(*args, **kwargs):
                data_loads.append((args, kwargs))
                return batches, batches[:1], "lm"

            def record_for(job, contract):
                calls.append(contract.name)
                if contract.name == "dacp" and calls.count("dacp") == 1:
                    raise RuntimeError("simulated interruption")
                learning_rates = [
                    job.workload.learning_rate
                    * (1.0 + math.cos(math.pi * step / job.workload.total_steps))
                    / 2.0
                    for step in range(job.workload.total_steps)
                ]
                training = {
                    "steps": 2,
                    "seconds": 0.01,
                    "train_losses": [1.0, 0.9],
                    "learning_rates": learning_rates[:2],
                    "task_type": "lm",
                }
                final_training = {
                    **training,
                    "learning_rates": learning_rates[2:],
                }
                before = {
                    "perplexity": 2.0,
                    "seconds": 0.01,
                    "batches": 1,
                    "examples": 1,
                }
                after = {
                    **before,
                    "perplexity": 2.0 if contract.name == "no_compression" else 2.1,
                }
                return {
                    "method": contract.name,
                    "owner": contract.owner,
                    "fidelity": contract.fidelity,
                    "internal_method": contract.internal_method,
                    "seed": 42,
                    "prune_ratio": job.prune_ratio,
                    "recovery_count": 1,
                    "cycles": [
                        {
                            "cycle": 1,
                            "training": training,
                            "before_recovery": before,
                            "after_recovery": after,
                            "compression": (
                                None
                                if contract.name == "no_compression"
                                else {
                                    "allocation": {
                                        "target_pruned": 1,
                                        "layer_sizes": [2],
                                        "eligible_parameters": 2,
                                        "target_eligible_sparsity": job.prune_ratio,
                                        "weibull_layer_counts": [1],
                                        "weibull_fits": [
                                            {
                                                "layer": 0,
                                                "valid": True,
                                                "count": 2,
                                                "mean": 1.0,
                                                "variance": 1.0,
                                                "cv_squared": 1.0,
                                                "maximum": 2.0,
                                                "shape": 1.0,
                                                "scale": 1.0,
                                                "zero_fraction": 0.0,
                                                "moment_reduction": {
                                                    "distributed": False,
                                                    "world_size": 1,
                                                    "backend": None,
                                                    "communicated_scalars_per_rank": 0,
                                                },
                                            }
                                        ],
                                        "weibull": {
                                            "threshold": math.log(2.0),
                                            "real_counts": [1.0],
                                            "capacities": [1],
                                            "fallback": None,
                                        },
                                    },
                                    "scoring": {
                                        "score_kind": "taylor_hvp",
                                        "total_seconds": 0.01,
                                        "layer_seconds": [0.01],
                                        "checksum_before": [1.0, 1.0],
                                        "checksum_after": [1.0, 1.0],
                                        "checksum_delta": [0.0, 0.0],
                                        "changed_parameter_versions": [],
                                        "optimizer_constructed": False,
                                        "model_mode": "eval",
                                        "hvp_batches": 1,
                                        "task_type": "lm",
                                        "model_family": "gpt2",
                                    },
                                    "mask": {
                                        "eligible_parameters": 2,
                                        "pruned": 1,
                                        "residual_magnitude_cost": 0.5,
                                        "layer_rates": [0.5],
                                    },
                                }
                            ),
                        }
                    ],
                    "final_training": final_training,
                    "final": {
                        "perplexity": 1.9,
                        "seconds": 0.01,
                        "batches": 1,
                        "examples": 1,
                    },
                    "wall_seconds": 0.1,
                }

            output_dir = root / "output"
            argv = ["--output-dir", str(output_dir), "--device", "cpu"]
            patches = (
                mock.patch.object(paper_script, "load_paper_manifest", return_value=manifest),
                mock.patch.object(
                    paper_script,
                    "_source_provenance",
                    return_value={
                        "manifest": str(manifest.path),
                        "manifest_sha256": _digest("manifest"),
                        "git_commit": "a" * 40,
                        "git_dirty": False,
                        "dirty_source_paths": [],
                        "source_file_count": 1,
                        "source_state_sha256": _digest("source"),
                    },
                ),
                mock.patch.object(
                    paper_script,
                    "load_training_checkpoint",
                    side_effect=load_checkpoint,
                ),
                mock.patch.object(
                    paper_script,
                    "get_data_loaders",
                    side_effect=load_data,
                ),
                mock.patch.object(
                    paper_script,
                    "run_method_trajectory",
                    side_effect=lambda job, contract, *args, **kwargs: record_for(job, contract),
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    paper_script.main(argv)
                paper_script.main([*argv, "--resume"])

            self.assertEqual(
                calls,
                ["no_compression", "dacp", "dacp", "no_compression", "dacp"],
            )
            self.assertEqual(len(checkpoint_loads), 2)
            self.assertEqual(len(data_loads), 2)
            job_id = PaperJob(workload, 0.5, 1, (42,)).job_id
            job_payload = json.loads((output_dir / f"{job_id}.json").read_text())
            self.assertEqual(job_payload["status"], "complete")
            self.assertEqual(
                [record["method"] for record in job_payload["results"]],
                ["no_compression", "dacp"],
            )
            suite = json.loads((output_dir / "suite_manifest.json").read_text())
            self.assertEqual(suite["status"], "complete")
            self.assertEqual(len(suite["completed_jobs"]), 2)
            self.assertEqual(job_payload["provenance"]["checkpoint_step"], 1000)
            self.assertEqual(
                job_payload["config"]["recovery_protocol"]["scheduler"],
                {
                    "name": "cosine_annealing",
                    "state": "fresh",
                    "scope": "matched_recovery_horizon",
                    "t_max": 4,
                    "eta_min": 0.0,
                },
            )

            checkpoint_path.write_bytes(b"changed trusted checkpoint fixture")
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaisesRegex(
                    ValueError, "cached data identity changed across suite jobs"
                ):
                    paper_script.main([*argv, "--resume"])

    def test_job_ids_are_path_safe_and_distinguish_close_ratios(self) -> None:
        workload = tiny_workload()
        first = PaperJob(workload, 0.5, 1, (42,)).job_id
        second = PaperJob(workload, 0.500000001, 1, (42,)).job_id

        self.assertNotEqual(first, second)
        self.assertNotIn("/", first)
        self.assertNotIn("\\", first)

    def test_paper_device_validation_fails_before_model_loading(self) -> None:
        self.assertEqual(paper_script._validate_device("cpu"), "cpu")
        for device in ("not-a-device", "mps"):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, "device|CPU"):
                paper_script._validate_device(device)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                paper_script._validate_device("cuda")

    def test_workload_context_rejects_wrong_checkpoint_step_and_short_data(self) -> None:
        workload = tiny_workload()
        batches = [
            {"input_ids": torch.tensor([[index]]), "labels": torch.tensor([[index]])}
            for index in range(workload.train_pool_batches)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = root / "checkpoint.pt"
            checkpoint_path.write_bytes(b"trusted fixture")
            workload = PaperWorkload(
                **{
                    **workload.__dict__,
                    "checkpoint": checkpoint_path,
                    "data_dir": root,
                }
            )
            common = {
                "checkpoint_root": None,
                "data_root": None,
                "declared_checkpoint_root": ROOT / "checkpoints",
                "declared_data_root": ROOT / "data",
                "expected_data_identity": None,
            }

            with (
                mock.patch.object(
                    paper_script,
                    "load_training_checkpoint",
                    return_value=SimpleNamespace(model_state={}, optimizer_state={}, step=999),
                ),
                self.assertRaisesRegex(ValueError, "manifest requires 1000"),
            ):
                paper_script._load_workload_context(workload, **common)

            with (
                mock.patch.object(
                    paper_script,
                    "load_training_checkpoint",
                    return_value=SimpleNamespace(model_state={}, optimizer_state={}, step=1000),
                ),
                mock.patch.object(
                    paper_script,
                    "get_data_loaders",
                    return_value=(batches[:-1], batches[:1], "lm"),
                ),
                self.assertRaisesRegex(RuntimeError, "training loader provided"),
            ):
                paper_script._load_workload_context(workload, **common)

    def test_source_provenance_tracks_sources_but_ignores_result_files(self) -> None:
        manifest_path = ROOT / "experiments/configs/paper_experiments.yaml"
        baseline = paper_script._source_provenance(manifest_path)

        result_path = ROOT / "results" / "provenance_test_output.json"
        source_path = ROOT / "provenance_test_source.py"
        try:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text("{}", encoding="utf-8")
            after_result = paper_script._source_provenance(manifest_path)
            self.assertEqual(
                after_result["source_state_sha256"],
                baseline["source_state_sha256"],
            )

            source_path.write_text("VALUE = 1\n", encoding="utf-8")
            after_source = paper_script._source_provenance(manifest_path)
            self.assertNotEqual(
                after_source["source_state_sha256"],
                baseline["source_state_sha256"],
            )
            self.assertIn("provenance_test_source.py", after_source["dirty_source_paths"])
        finally:
            result_path.unlink(missing_ok=True)
            source_path.unlink(missing_ok=True)

    def test_source_provenance_tracks_nested_sources_and_deletions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "experiments" / "configs" / "paper.yaml"
            nested_source = root / "experiments" / "lib" / "nested" / "source.py"
            manifest_path.parent.mkdir(parents=True)
            nested_source.parent.mkdir(parents=True)
            manifest_path.write_text("schema_version: 1\n", encoding="utf-8")
            nested_source.write_text("VALUE = 1\n", encoding="utf-8")

            def git(*arguments: str) -> None:
                subprocess.run(
                    ["git", *arguments],
                    cwd=root,
                    check=True,
                    capture_output=True,
                )

            git("init", "-q")
            git("config", "user.email", "tests@example.invalid")
            git("config", "user.name", "DACP Tests")
            git("add", ".")
            git("commit", "-q", "-m", "fixture")

            with mock.patch.object(paper_script, "ROOT", root):
                baseline = paper_script._source_provenance(manifest_path)
                nested_source.unlink()
                deleted = paper_script._source_provenance(manifest_path)

            self.assertEqual(baseline["source_file_count"], 2)
            self.assertEqual(deleted["source_file_count"], 1)
            self.assertNotEqual(
                baseline["source_state_sha256"],
                deleted["source_state_sha256"],
            )
            self.assertIn(
                "experiments/lib/nested/source.py",
                deleted["dirty_source_paths"],
            )

    def test_completed_workload_preflight_revalidates_only_finished_workloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = root / "checkpoint.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            workload_a = PaperWorkload(
                **{
                    **tiny_workload().__dict__,
                    "name": "workload_a",
                    "checkpoint": checkpoint_path,
                    "data_dir": root,
                    "prune_ratios": (0.5, 0.7),
                }
            )
            workload_b = PaperWorkload(
                **{
                    **tiny_workload().__dict__,
                    "name": "workload_b",
                    "checkpoint": checkpoint_path,
                    "data_dir": root,
                }
            )
            jobs = [
                *(PaperJob(workload_a, ratio, 1, (42,)) for ratio in workload_a.prune_ratios),
                PaperJob(workload_b, 0.5, 1, (42,)),
            ]

            class Suite:
                def __init__(self, completed, identities):
                    self.completed = completed
                    self.identities = identities

                def has_completed_job(self, job_id):
                    return job_id in self.completed

                def expected_data_identity(self, workload_name):
                    return self.identities[workload_name]

            identity = {
                "checkpoint_sha256": _digest("checkpoint"),
                "checkpoint_step": workload_a.checkpoint_step,
                "training_pool": {"count": 7, "sha256": _digest("training")},
                "evaluation_batches": {"count": 1, "sha256": _digest("evaluation")},
            }
            suite = Suite({job.job_id for job in jobs[:2]}, {"workload_a": identity})
            loaded = []

            def load_context(workload, **kwargs):
                loaded.append((workload.name, kwargs["expected_data_identity"]))
                return SimpleNamespace()

            with (
                mock.patch.object(paper_script, "_load_workload_context", side_effect=load_context),
                mock.patch.object(paper_script, "empty_device_cache") as empty_cache,
            ):
                paper_script._validate_completed_workload_contexts(
                    jobs,
                    suite,
                    checkpoint_root=None,
                    data_root=None,
                    declared_checkpoint_root=ROOT / "checkpoints",
                    declared_data_root=ROOT / "data",
                    device="cpu",
                )

            self.assertEqual(loaded, [("workload_a", identity)])
            empty_cache.assert_called_once_with("cpu")

            with mock.patch.object(
                paper_script,
                "_load_workload_context",
                side_effect=ValueError(
                    "workload_a: cached data identity changed across suite jobs"
                ),
            ):
                with self.assertRaisesRegex(ValueError, "data identity changed"):
                    paper_script._validate_completed_workload_contexts(
                        jobs,
                        suite,
                        checkpoint_root=None,
                        data_root=None,
                        declared_checkpoint_root=ROOT / "checkpoints",
                        declared_data_root=ROOT / "data",
                        device="cpu",
                    )

    def test_complete_suite_resume_revalidates_completed_workloads(self) -> None:
        workload = tiny_workload()
        manifest = PaperManifest(
            path=Path("manifest.yaml"),
            schema_version=1,
            methods=tuple(resolve_method_contracts(["dacp"])),
            claim_gates={},
            workloads=(workload,),
        )
        suite = SimpleNamespace(complete=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "suite_manifest.json").write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(paper_script, "load_paper_manifest", return_value=manifest),
                mock.patch.object(
                    paper_script, "_source_provenance", return_value=_source_provenance()
                ),
                mock.patch.object(paper_script, "SuiteResultStore") as store,
                mock.patch.object(
                    paper_script, "_validate_completed_workload_contexts"
                ) as preflight,
            ):
                store.open.return_value = suite
                paper_script.main(["--resume", "--device", "cpu", "--output-dir", str(output_dir)])

            preflight.assert_called_once()


if __name__ == "__main__":
    unittest.main()
