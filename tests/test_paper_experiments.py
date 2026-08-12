import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import experiments.lib.paper_runner as paper_runner
from experiments.lib.paper_baselines import (
    resolve_method_contracts,
    validate_claim_gate,
)
from experiments.lib.paper_manifest import PaperWorkload, load_paper_manifest
from experiments.lib.paper_runner import (
    PaperJob,
    aggregate_job_results,
    plan_jobs,
    run_method_trajectory,
)
from experiments.lib.residual_training import partition_repeated_seed_batches
from experiments.scripts.generate_paper_figures import (
    generate_allocation_figure,
    generate_distribution_fit_figure,
    load_completed_suite,
)


ROOT = Path(__file__).resolve().parents[1]


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


class TestPaperExperiments(unittest.TestCase):
    def test_repository_manifest_declares_exact_paper_workloads(self) -> None:
        manifest = load_paper_manifest(
            ROOT / "experiments/configs/paper_experiments.yaml"
        )

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
        self.assertEqual([job.job_id for job in jobs], ["tiny_p0p5_k3"])
        with self.assertRaisesRegex(ValueError, "not declared"):
            plan_jobs([workload], prune_ratios=[0.7])

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

        def train_side_effect(model, optimizer, scheduler, batches, seed, device, task_type):
            train_calls.append((len(batches), seed, task_type))
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
        self.assertEqual(len(result["cycles"]), 1)
        self.assertEqual(result["cycles"][0]["compression"], None)
        self.assertEqual(result["final"]["perplexity"], 5.0)
        json.dumps(result, allow_nan=False)

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
                json.dumps({"status": "complete", "records": records}),
                encoding="utf-8",
            )

            job_path = root / "job.json"
            job_payload = {
                "status": "complete",
                "config": {
                    "name": "bert_large_mnli",
                    "prune_ratio": 0.3,
                },
                "results": [
                    {
                        "method": "dacp",
                        "seed": 42,
                        "cycles": [
                            {
                                "compression": {
                                    "mask": {"layer_rates": [0.2, 0.4]}
                                }
                            }
                        ],
                    }
                ],
            }
            job_path.write_text(json.dumps(job_payload), encoding="utf-8")
            digest = hashlib.sha256(job_path.read_bytes()).hexdigest()
            suite_path = root / "suite_manifest.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "completed_jobs": {
                            "bert_job": {"path": str(job_path), "sha256": digest}
                        },
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "figures"

            distribution = generate_distribution_fit_figure(fit_path, output_dir)
            allocation = generate_allocation_figure(suite_path, output_dir)

            for output in (distribution, allocation):
                self.assertTrue(output.is_file())
                self.assertTrue(output.with_suffix(".png").is_file())
                self.assertTrue(output.with_suffix(".provenance.json").is_file())

            job_path.write_text(json.dumps({**job_payload, "tampered": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_completed_suite(suite_path)

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

        for contract in resolve_method_contracts(
            ["excp_style", "inshrinkerator_style", "dacp"]
        ):
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


if __name__ == "__main__":
    unittest.main()
