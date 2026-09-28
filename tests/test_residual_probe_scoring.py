import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.lib.paper_baselines import (
    METHOD_CONTRACTS,
    resolve_method_contracts,
    validate_claim_gate,
)
from experiments.lib.paper_manifest import PaperWorkload, load_paper_manifest
from experiments.lib.paper_results import JobResultStore
from experiments.lib.paper_runner import PaperJob, aggregate_job_results, run_method_trajectory
from experiments.lib.probe_coverage import weighted_token_ce
from experiments.lib.residual_probe_scoring import compute_probe_gn_scores, valid_shifted_logits
from experiments.lib.residual_protocol import PROBE_METHODS
from experiments.lib.residual_runtime import lm_loss
from experiments.lib.residual_scoring import eligible_layers, transformer_layers
from experiments.lib.residual_training import partition_repeated_seed_batches
from experiments.scripts import run_paper_experiments as paper_script

ROOT = Path(__file__).resolve().parents[1]
VOCAB = 50


def tiny_gpt2(seed: int = 0) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=16,
        n_embd=16,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(config).eval()


def lm_batch(tokens: list[int]) -> dict[str, torch.Tensor]:
    ids = torch.tensor([tokens], dtype=torch.long)
    return {"input_ids": ids, "labels": ids.clone()}


def residual(model: torch.nn.Module, layers, seed: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    named = dict(model.named_parameters())
    return {
        name: 0.05 * torch.randn(named[name].shape, generator=generator)
        for layer in layers
        for name in layer
    }


def explicit_terms(model, batches, names, delta):
    """Return (g, F d) for L = mean_b token_mean CE via explicit per-token vectors."""
    named = dict(model.named_parameters())
    params = [named[name] for name in names]
    probes = [delta[name].float() for name in names]
    gradient = [torch.zeros_like(p) for p in probes]
    curvature = [torch.zeros_like(p) for p in probes]
    for batch in batches:
        loss = lm_loss(model, batch, "cpu")
        for acc, value in zip(gradient, torch.autograd.grad(loss, params)):
            acc.add_(value / len(batches))
        logits, _ = valid_shifted_logits(model, batch, "cpu")
        count = logits.shape[0]
        for t in range(count):
            probs = torch.softmax(logits[t].detach(), dim=-1)
            log_probs = torch.log_softmax(logits[t], dim=-1)
            for y in range(logits.shape[1]):
                v = torch.autograd.grad(log_probs[y], params, retain_graph=True)
                dot = sum((vi * di).sum() for vi, di in zip(v, probes))
                scale = float(probs[y]) * float(dot) / (len(batches) * count)
                for acc, vi in zip(curvature, v):
                    acc.add_(vi, alpha=scale)
    return dict(zip(names, gradient)), dict(zip(names, curvature))


class ProbeScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = tiny_gpt2()
        self.layers = eligible_layers(self.model, "gpt2")
        self.blocks = transformer_layers(self.model, "gpt2")
        self.names = [name for layer in self.layers for name in layer]
        self.delta = residual(self.model, self.layers)

    def score(self, batches, **kwargs):
        return compute_probe_gn_scores(
            self.model,
            batches,
            self.layers,
            self.delta,
            "cpu",
            block_parameter_names=self.blocks,
            **kwargs,
        )

    def test_weighted_backward_equals_negative_weighted_logprob_gradients(self) -> None:
        batch = lm_batch([3, 7, 11, 2])
        named = dict(self.model.named_parameters())
        params = [named[name] for name in self.names]
        logits, _ = valid_shifted_logits(self.model, batch, "cpu")
        tokens = torch.tensor([5, 0, 49])
        weights = torch.tensor([0.7, -1.3, 0.2], dtype=torch.float64)
        w = torch.autograd.grad(
            weighted_token_ce(logits, tokens, weights), params, retain_graph=True
        )
        expected = [torch.zeros_like(p) for p in params]
        for t in range(3):
            log_prob = torch.log_softmax(logits[t], dim=-1)[tokens[t]]
            v = torch.autograd.grad(log_prob, params, retain_graph=True)
            for acc, vi in zip(expected, v):
                acc.add_(vi, alpha=-float(weights[t]))
        for actual, reference in zip(w, expected):
            torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-7)

    def test_full_enumeration_single_position_is_exact(self) -> None:
        batches = [lm_batch([4, 9])]
        g, fd = explicit_terms(self.model, batches, self.names, self.delta)
        for estimator, rounds in (("topk", VOCAB), ("headtail", VOCAB)):
            with self.subTest(estimator=estimator):
                scores, meta = self.score(batches, estimator=estimator, rounds=rounds, seed=3)
                self.assertEqual(meta["probe"]["total_positions"], 1)
                self.assertAlmostEqual(meta["probe"]["per_round"][-1]["mean_mass"], 1.0, 5)
                for name in self.names:
                    d = self.delta[name]
                    expected = (-g[name] * d + 0.5 * d * fd[name]).abs()
                    torch.testing.assert_close(scores[name], expected, rtol=2e-3, atol=1e-9)

    def test_sign_average_recovers_gauss_newton_with_batch_normalisation(self) -> None:
        batches = [lm_batch([1, 8, 30]), lm_batch([12, 6])]
        g, fd = explicit_terms(self.model, batches, self.names, self.delta)
        offsets = {0: 0, 1: 2}
        patterns = list(itertools.product((-1.0, 1.0), repeat=3))
        mean_u = {name: torch.zeros_like(self.delta[name]) for name in self.names}
        for pattern in patterns:
            signs_fn = lambda k, b, n, p=pattern: torch.tensor(p[offsets[b] : offsets[b] + n])
            components, meta = self.score(
                batches,
                estimator="topk",
                rounds=VOCAB,
                seed=0,
                signs_fn=signs_fn,
                return_components=True,
            )
            self.assertEqual(meta["probe"]["positions_per_batch"], [2, 1])
            for name in self.names:
                mean_u[name].add_(components["curvature"][name] / len(patterns))
                d = self.delta[name]
                torch.testing.assert_close(
                    components["gradient_term"][name], -g[name] * d, rtol=1e-4, atol=1e-10
                )
                torch.testing.assert_close(
                    components["score"][name],
                    (components["gradient_term"][name] + 0.5 * d * components["curvature"][name]).abs(),
                )
        for name in self.names:
            torch.testing.assert_close(mean_u[name], fd[name], rtol=2e-3, atol=1e-8)

    def test_sample_coverage_is_deduplicated_and_seeded(self) -> None:
        batches = [lm_batch([2, 3, 5, 7, 11]), lm_batch([13, 17, 19])]
        first, meta = self.score(batches, estimator="sample", rounds=8, seed=11)
        again, _ = self.score(batches, estimator="sample", rounds=8, seed=11)
        other, _ = self.score(batches, estimator="sample", rounds=8, seed=12)
        for name in self.names:
            torch.testing.assert_close(first[name], again[name], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(first[n], other[n]) for n in self.names))
        probe = meta["probe"]
        rounds = probe["per_round"]
        self.assertEqual(len(rounds), 8)
        self.assertEqual(rounds[0]["repeat_rate"], 0.0)
        masses = [record["mean_mass"] for record in rounds]
        self.assertEqual(masses, sorted(masses))
        self.assertLessEqual(masses[-1], 1.0)
        repeats = sum(record["positions"] - record["new_tokens"] for record in rounds)
        self.assertEqual(repeats, probe["cumulative_repeats"])
        self.assertEqual(probe["cumulative_draws"], 8 * 6)
        self.assertEqual(set(meta["phase_seconds"]), {
            "gradient",
            "probe_forward",
            "probe_selection",
            "probe_backward",
            "curvature_update",
            "score_assembly",
        })
        self.assertEqual(meta["checksum_delta"], [0.0, 0.0])

    def test_topk_rounds_cover_distinct_tokens(self) -> None:
        batches = [lm_batch([2, 3, 5, 7])]
        _, meta = self.score(batches, estimator="topk", rounds=4, seed=0)
        self.assertEqual(meta["probe"]["cumulative_repeats"], 0)
        _, head = self.score(batches, estimator="headtail", rounds=4, seed=0)
        self.assertEqual([r["kind"] for r in head["probe"]["per_round"]], ["head"] * 3 + ["tail"])
        self.assertIn("head_mean_mass", head["probe"]["per_round"][-1])

    def test_non_causal_task_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "causal-LM token cross-entropy"):
            self.score([lm_batch([1, 2])], estimator="topk", rounds=1, seed=0, task_type="cls")

    def test_rounds_beyond_vocabulary_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds vocabulary"):
            self.score([lm_batch([1, 2])], estimator="headtail", rounds=VOCAB + 1, seed=0)


class ProbeContractTest(unittest.TestCase):
    def test_probe_contracts_are_native_and_do_not_satisfy_baseline_gates(self) -> None:
        names = [*PROBE_METHODS, "first_order_exact_global"]
        contracts = resolve_method_contracts(names)
        self.assertTrue(all(c.owner == "dacp_probe" and c.fidelity == "native" for c in contracts))
        self.assertEqual(
            METHOD_CONTRACTS["first_order_exact_global"].internal_method,
            "residual_first_order_exact_global",
        )
        self.assertEqual(
            METHOD_CONTRACTS["inshrinkerator_style"].internal_method, "first_order_exact_global"
        )
        with self.assertRaisesRegex(ValueError, "missing baselines"):
            validate_claim_gate(
                contracts,
                {"baseline_owners": ["excp", "inshrinkerator"], "required_baseline_fidelity": "style"},
            )

    def test_manifest_loads_with_declared_selection(self) -> None:
        manifest = load_paper_manifest(ROOT / "experiments/configs/probe_small_models.yaml")
        self.assertEqual(
            [w.name for w in manifest.workloads],
            ["gpt2_medium_wikitext103", "pythia_410m_alpaca", "pythia_1b_alpaca"],
        )
        for workload in manifest.workloads:
            self.assertEqual(workload.task_type, "lm")
            self.assertEqual(workload.prune_ratios, (0.5, 0.9))
            self.assertEqual(workload.recovery_counts, (1,))
            self.assertEqual(workload.seeds, (42, 43, 44))

    def test_manifest_dry_run_cli(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "experiments/scripts/run_paper_experiments.py",
                "--manifest",
                "experiments/configs/probe_small_models.yaml",
                "--dry-run",
            ],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        plan = json.loads(completed.stdout.decode("utf-8"))
        self.assertEqual(len(plan["jobs"]), 6)
        self.assertEqual(
            [m["name"] for m in plan["methods"]],
            [
                "no_compression",
                "excp_style",
                "inshrinkerator_style",
                "first_order_exact_global",
                "taylor_exact_global",
                "dacp",
                "probe_topk_m1_exact_global",
                "probe_topk_m4_exact_global",
                "probe_sample_m1_exact_global",
                "probe_sample_m4_exact_global",
                "probe_headtail_m2_exact_global",
            ],
        )


class ProbeTrajectoryTest(unittest.TestCase):
    def test_probe_and_residual_first_order_records_pass_result_validation(self) -> None:
        workload = PaperWorkload(
            name="tiny",
            model="gpt2-medium",
            model_family="gpt2",
            dataset="wikitext103",
            task_type="lm",
            checkpoint=Path("checkpoint.pt"),
            checkpoint_step=1000,
            data_dir=Path("data"),
            metric="perplexity",
            seeds=(42,),
            prune_ratios=(0.5,),
            max_layer_ratio=0.95,
            recovery_counts=(1,),
            batch_size=1,
            seq_length=4,
            total_steps=2,
            hvp_batches=2,
            eval_batches=1,
            train_pool_batches=4,
            learning_rate=1e-3,
        )
        job = PaperJob(workload, 0.5, 1, (42,))
        pool = [lm_batch([(3 * i + j) % VOCAB for j in range(4)]) for i in range(4)]
        plan = partition_repeated_seed_batches(pool, 2, 1, 2, seed=42)
        model = tiny_gpt2()
        optimizer_state = torch.optim.AdamW(model.parameters(), lr=1e-3).state_dict()
        names = ["first_order_exact_global", "probe_headtail_m2_exact_global", "probe_sample_m1_exact_global"]
        contracts = resolve_method_contracts(names)
        context = paper_script.WorkloadContext(
            checkpoint_path=Path("checkpoint.pt"),
            checkpoint_sha256="0" * 64,
            checkpoint=SimpleNamespace(step=1000),
            training_pool=pool,
            eval_batches=pool[:1],
            data_identity={},
        )
        provenance = {
            "manifest": "probe.yaml",
            "manifest_sha256": "1" * 64,
            "git_commit": "a" * 40,
            "git_dirty": False,
            "dirty_source_paths": [],
            "source_file_count": 1,
            "source_state_sha256": "2" * 64,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobResultStore.open(
                Path(temp_dir) / "job.json",
                paper_script._job_template(job, contracts, provenance, context),
                metric="perplexity",
                resume=False,
            )
            store.ensure_batch_plan(
                42,
                {"selected_pool_indices": plan.selected_pool_indices, **plan.data_hashes()},
            )
            records = []
            for contract in contracts:
                record = run_method_trajectory(
                    job,
                    contract,
                    42,
                    plan,
                    tiny_gpt2,
                    model.state_dict(),
                    optimizer_state,
                    pool[:1],
                    "cpu",
                )
                store.append_result(record)
                records.append(record)
            store.finalize(aggregate_job_results(records, "perplexity"), finished_at=paper_script._utc_now())
        probe = records[1]["cycles"][0]["compression"]
        self.assertEqual(probe["score_kind"], "probe_gn_headtail")
        self.assertEqual(probe["allocation_kind"], "exact_global")
        self.assertEqual(probe["scoring"]["probe"]["seed"], 42)
        self.assertEqual(len(probe["scoring"]["probe"]["per_round"]), 2)
        self.assertEqual(records[0]["cycles"][0]["compression"]["score_probe"], "checkpoint_residual")


if __name__ == "__main__":
    unittest.main()
