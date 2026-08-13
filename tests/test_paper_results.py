import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.lib.paper_baselines import METHOD_CONTRACTS
from experiments.lib.paper_manifest import paper_job_id
from experiments.lib.paper_results import (
    SCHEMA_VERSION,
    JobResultStore,
    SuiteResultStore,
    records_from,
)
from experiments.lib.paper_runner import aggregate_job_results
from experiments.lib.residual_runtime import sha256_file, write_json

NOW = "2026-08-13T00:00:00+00:00"
BEFORE_NOW = "2026-08-12T23:59:59+00:00"
AFTER_NOW = "2026-08-13T00:00:01+00:00"


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


def _contract(name: str = "dacp") -> dict[str, str]:
    return METHOD_CONTRACTS[name].to_result_dict()


def _job_template() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "started_at": NOW,
        "provenance": {
            **_source_provenance(),
            "checkpoint": "checkpoint.pt",
            "checkpoint_sha256": _digest("checkpoint"),
            "checkpoint_step": 1000,
        },
        "config": {
            "name": "tiny",
            "model": "gpt2-medium",
            "model_family": "gpt2",
            "dataset": "wikitext103",
            "task_type": "lm",
            "checkpoint": "checkpoint.pt",
            "checkpoint_step": 1000,
            "data_dir": "data/wikitext103",
            "metric": "perplexity",
            "seeds": [42],
            "prune_ratios": [0.5],
            "prune_ratio": 0.5,
            "recovery_counts": [1],
            "recovery_count": 1,
            "total_steps": 2,
            "hvp_batches": 1,
            "eval_batches": 1,
            "train_pool_batches": 3,
            "max_layer_ratio": 0.95,
            "batch_size": 1,
            "seq_length": 2,
            "learning_rate": 1e-4,
            "recovery_protocol": {
                "optimizer_state": "restored_from_checkpoint",
                "learning_rate_source": "manifest",
                "learning_rate": 1e-4,
                "scheduler": {
                    "name": "cosine_annealing",
                    "state": "fresh",
                    "scope": "matched_recovery_horizon",
                    "t_max": 2,
                    "eta_min": 0.0,
                },
            },
        },
        "method_contracts": [_contract()],
        "training_pool": {"count": 3, "sha256": _digest("train")},
        "evaluation_batches": {"count": 1, "sha256": _digest("eval")},
        "batch_plans": {},
        "results": [],
    }


def _batch_plan() -> dict[str, object]:
    return {
        "selected_pool_indices": [0, 1, 2],
        "training_segments": [_digest("first"), _digest("second")],
        "scoring_batches": [_digest("score")],
    }


def _evaluation(perplexity: float) -> dict[str, object]:
    return {
        "perplexity": perplexity,
        "seconds": 0.01,
        "batches": 1,
        "examples": 1,
    }


def _record() -> dict[str, object]:
    contract = _contract()
    first_training = {
        "steps": 1,
        "seconds": 0.01,
        "train_losses": [1.0],
        "learning_rates": [1e-4],
        "task_type": "lm",
    }
    final_training = {
        **first_training,
        "learning_rates": [5e-5],
    }
    return {
        "method": contract["name"],
        "owner": contract["owner"],
        "fidelity": contract["fidelity"],
        "internal_method": contract["internal_method"],
        "seed": 42,
        "prune_ratio": 0.5,
        "recovery_count": 1,
        "cycles": [
            {
                "cycle": 1,
                "training": first_training,
                "before_recovery": _evaluation(2.0),
                "after_recovery": _evaluation(2.1),
                "compression": {
                    "allocation": {
                        "target_pruned": 1,
                        "layer_sizes": [2],
                        "eligible_parameters": 2,
                        "target_eligible_sparsity": 0.5,
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
                },
            }
        ],
        "final_training": final_training,
        "final": _evaluation(1.9),
        "wall_seconds": 0.1,
    }


def _aggregate(record: dict[str, object] | None = None) -> dict[str, object]:
    return aggregate_job_results([record or _record()], "perplexity")


def _job_id(payload: dict[str, object] | None = None) -> str:
    config = (payload or _job_template())["config"]
    return paper_job_id(config["name"], config["prune_ratio"], config["recovery_count"])


class TestPaperResultStores(unittest.TestCase):
    def test_job_store_rejects_inconsistent_lifecycle_fields(self) -> None:
        cases = []
        premature = _job_template()
        premature["finished_at"] = NOW
        cases.append(("premature", premature, "completion fields while started"))

        reversed_timeline = _job_template()
        reversed_timeline["status"] = "complete"
        reversed_timeline["finished_at"] = BEFORE_NOW
        reversed_timeline["batch_plans"] = {"42": _batch_plan()}
        reversed_timeline["results"] = [_record()]
        reversed_timeline["aggregate"] = _aggregate()
        cases.append(("reversed", reversed_timeline, "precedes started_at"))

        for name, payload, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "job.json"
                write_json(path, payload)
                with self.assertRaisesRegex(ValueError, message):
                    JobResultStore.open(
                        path,
                        _job_template(),
                        metric="perplexity",
                        resume=True,
                    )

    def test_job_finalize_rejects_finished_at_before_started_at(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.json"
            store = JobResultStore.open(
                path,
                _job_template(),
                metric="perplexity",
                resume=False,
            )
            store.ensure_batch_plan(42, _batch_plan())
            store.append_result(_record())
            stable = path.read_bytes()

            with self.assertRaisesRegex(ValueError, "precedes started_at"):
                store.finalize(_aggregate(), finished_at=BEFORE_NOW)

            self.assertFalse(store.complete)
            self.assertEqual(path.read_bytes(), stable)

    def test_job_store_resumes_only_complete_verified_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.json"
            template = _job_template()
            store = JobResultStore.open(
                path,
                template,
                metric="perplexity",
                resume=False,
            )
            store.ensure_batch_plan(42, _batch_plan())
            store.append_result(_record())

            resumed = JobResultStore.open(
                path,
                template,
                metric="perplexity",
                resume=True,
            )
            self.assertTrue(resumed.has_result(42, "dacp"))
            self.assertEqual(len(records_from(resumed)), 1)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                resumed.append_result(_record())

            resumed.finalize(_aggregate(), finished_at=NOW)
            completed = JobResultStore.open(
                path,
                template,
                metric="perplexity",
                resume=True,
            )
            self.assertTrue(completed.complete)

    def test_job_store_rejects_partial_tampered_or_incompatible_state(self) -> None:
        cases = []
        partial = _record()
        partial["cycles"] = []
        cases.append(("partial", partial, "incomplete"))
        non_finite = _record()
        non_finite["final"] = {"perplexity": float("inf")}
        cases.append(("non_finite", non_finite, "Non-finite JSON"))
        wrong_contract = _record()
        wrong_contract["owner"] = "other"
        cases.append(("wrong_contract", wrong_contract, "incompatible owner"))
        missing_training = _record()
        missing_training["final_training"] = {"steps": 1}
        cases.append(("missing_training", missing_training, "final training"))
        wrong_mask = _record()
        wrong_mask["cycles"][0]["compression"]["mask"]["pruned"] = 0
        cases.append(("wrong_mask", wrong_mask, "expected 1"))
        wrong_target = _record()
        wrong_target["cycles"][0]["compression"]["allocation"]["target_pruned"] = 0
        cases.append(("wrong_target", wrong_target, "target_pruned"))
        wrong_layer_sizes = _record()
        wrong_layer_sizes["cycles"][0]["compression"]["allocation"]["layer_sizes"] = [3]
        cases.append(("wrong_layer_sizes", wrong_layer_sizes, "layer sizes"))
        wrong_layer_rates = _record()
        wrong_layer_rates["cycles"][0]["compression"]["mask"]["layer_rates"] = [0.25]
        cases.append(("wrong_layer_rates", wrong_layer_rates, "exact layer prune count"))
        wrong_score_kind = _record()
        wrong_score_kind["cycles"][0]["compression"]["scoring"]["score_kind"] = "first_order"
        cases.append(("wrong_score_kind", wrong_score_kind, "scoring evidence"))
        wrong_scoring_batches = _record()
        wrong_scoring_batches["cycles"][0]["compression"]["scoring"]["hvp_batches"] = 2
        cases.append(("wrong_scoring_batches", wrong_scoring_batches, "scoring batch count"))
        changed_scoring_weights = _record()
        changed_scoring_weights["cycles"][0]["compression"]["scoring"]["checksum_after"] = [
            2.0,
            1.0,
        ]
        cases.append(("changed_scoring_weights", changed_scoring_weights, "changed model"))
        wrong_eval_batches = _record()
        wrong_eval_batches["cycles"][0]["after_recovery"]["batches"] = 2
        cases.append(("wrong_eval_batches", wrong_eval_batches, "evaluation batch count"))
        wrong_lr_trace = _record()
        wrong_lr_trace["final_training"]["learning_rates"] = [1e-4]
        cases.append(("wrong_lr_trace", wrong_lr_trace, "cosine schedule"))
        wrong_weibull_counts = _record()
        wrong_weibull_counts["cycles"][0]["compression"]["allocation"]["weibull_layer_counts"] = [0]
        cases.append(("wrong_weibull_counts", wrong_weibull_counts, "Weibull allocation"))
        wrong_weibull_shape = _record()
        wrong_weibull_shape["cycles"][0]["compression"]["allocation"]["weibull_fits"][0][
            "shape"
        ] = 0.0
        cases.append(
            ("wrong_weibull_shape", wrong_weibull_shape, "non-positive valid Weibull shape")
        )
        wrong_fit_count = _record()
        wrong_fit_count["cycles"][0]["compression"]["allocation"]["weibull_fits"][0]["count"] = 3
        cases.append(("wrong_fit_count", wrong_fit_count, "fit count does not match"))
        wrong_invalid_fit_count = _record()
        invalid_fit = wrong_invalid_fit_count["cycles"][0]["compression"]["allocation"]
        invalid_fit["weibull_fits"][0] = {
            "layer": 0,
            "valid": False,
            "reason": "non-positive maximum",
            "count": 3,
            "mean": 0.0,
            "variance": 0.0,
            "cv_squared": 0.0,
            "maximum": 0.0,
            "zero_fraction": 1.0,
            "moment_reduction": {
                "distributed": False,
                "world_size": 1,
                "backend": None,
                "communicated_scalars_per_rank": 0,
            },
        }
        invalid_fit["weibull"] = {
            "threshold": None,
            "fallback": "all Weibull fits invalid",
        }
        cases.append(
            ("wrong_invalid_fit_count", wrong_invalid_fit_count, "fit count does not match")
        )
        wrong_reduction = _record()
        wrong_reduction["cycles"][0]["compression"]["allocation"]["weibull_fits"][0][
            "moment_reduction"
        ]["world_size"] = 2
        cases.append(("wrong_reduction", wrong_reduction, "moment reduction"))
        wrong_capacity = _record()
        wrong_capacity["cycles"][0]["compression"]["allocation"]["weibull"]["capacities"] = [2]
        cases.append(("wrong_capacity", wrong_capacity, "threshold allocation evidence"))
        wrong_threshold = _record()
        wrong_threshold["cycles"][0]["compression"]["allocation"]["weibull"]["threshold"] = 0.4
        wrong_threshold["cycles"][0]["compression"]["allocation"]["weibull"]["real_counts"] = [
            2.0 * (1.0 - math.exp(-0.4))
        ]
        cases.append(("wrong_threshold", wrong_threshold, "does not solve the target budget"))

        for name, record, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "job.json"
                payload = _job_template()
                payload["batch_plans"] = {"42": _batch_plan()}
                payload["results"] = [record]
                if name == "non_finite":
                    path.write_text(json.dumps(payload), encoding="utf-8")
                else:
                    write_json(path, payload)
                with self.assertRaisesRegex(ValueError, message):
                    JobResultStore.open(
                        path,
                        _job_template(),
                        metric="perplexity",
                        resume=True,
                    )

    def test_job_store_validates_control_and_batch_identities(self) -> None:
        cases = []
        wrong_eval_count = _job_template()
        wrong_eval_count["evaluation_batches"]["count"] = 2
        cases.append(("eval_count", wrong_eval_count, "evaluation batch identity"))
        missing_eval_digest = _job_template()
        missing_eval_digest["evaluation_batches"]["sha256"] = ""
        cases.append(("eval_digest", missing_eval_digest, "evaluation batch digest"))
        wrong_pool_count = _job_template()
        wrong_pool_count["training_pool"]["count"] = 4
        cases.append(("pool_count", wrong_pool_count, "training pool count"))

        control_template = _job_template()
        control_template["method_contracts"] = [_contract("no_compression")]
        control = _record()
        control.update(
            {
                "method": "no_compression",
                "owner": "control",
                "fidelity": "native",
                "internal_method": "no_compression",
            }
        )
        control["cycles"][0]["compression"] = None
        control["cycles"][0]["after_recovery"] = _evaluation(2.2)
        control_template["batch_plans"] = {"42": _batch_plan()}
        control_template["results"] = [control]
        cases.append(("control", control_template, "preserve metrics"))

        for name, payload, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "job.json"
                write_json(path, payload)
                template = control_template if name == "control" else copy.deepcopy(payload)
                with self.assertRaisesRegex(ValueError, message):
                    JobResultStore.open(path, template, metric="perplexity", resume=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.json"
            payload = _job_template()
            payload["config"] = {**payload["config"], "prune_ratio": 0.7}
            write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "config does not match"):
                JobResultStore.open(
                    path,
                    _job_template(),
                    metric="perplexity",
                    resume=True,
                )

    def test_new_job_store_rejects_unexecutable_config_before_write(self) -> None:
        cases = []
        duplicate_seeds = _job_template()
        duplicate_seeds["config"]["seeds"] = [42, 42]
        cases.append(("duplicate_seeds", duplicate_seeds, "seeds"))
        short_pool = _job_template()
        short_pool["config"]["train_pool_batches"] = 2
        short_pool["training_pool"]["count"] = 2
        cases.append(("short_pool", short_pool, "train_pool_batches"))
        wrong_metric = _job_template()
        wrong_metric["config"]["metric"] = "accuracy"
        cases.append(("wrong_metric", wrong_metric, "metric"))
        wrong_protocol = _job_template()
        wrong_protocol["config"]["recovery_protocol"]["scheduler"]["t_max"] = 3
        cases.append(("wrong_protocol", wrong_protocol, "recovery_protocol"))
        undeclared_ratio = _job_template()
        undeclared_ratio["config"]["prune_ratio"] = 0.7
        cases.append(("undeclared_ratio", undeclared_ratio, "declared prune_ratios"))
        wrong_family = _job_template()
        wrong_family["config"]["model_family"] = "bert"
        cases.append(("wrong_family", wrong_family, "model_family"))
        uncovered_declared_ratio = _job_template()
        uncovered_declared_ratio["config"]["prune_ratios"] = [0.5, 0.99]
        cases.append(
            (
                "uncovered_declared_ratio",
                uncovered_declared_ratio,
                "max_layer_ratio must cover every declared",
            )
        )
        uncovered_declared_recovery = _job_template()
        uncovered_declared_recovery["config"]["recovery_counts"] = [1, 3]
        cases.append(
            (
                "uncovered_declared_recovery",
                uncovered_declared_recovery,
                "total_steps must cover every declared",
            )
        )

        for name, template, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "job.json"
                with self.assertRaisesRegex(ValueError, message):
                    JobResultStore.open(
                        path,
                        template,
                        metric=str(template["config"]["metric"]),
                        resume=False,
                    )
                self.assertFalse(path.exists())

    def test_new_job_store_rejects_inconsistent_provenance_before_write(self) -> None:
        cases = []
        false_clean = _job_template()
        false_clean["provenance"]["dirty_source_paths"] = ["experiments/lib/changed.py"]
        cases.append(("false_clean", false_clean, "git_dirty disagrees"))
        wrong_step = _job_template()
        wrong_step["provenance"]["checkpoint_step"] = 999
        cases.append(("wrong_step", wrong_step, "checkpoint_step does not match"))
        malformed_commit = _job_template()
        malformed_commit["provenance"]["git_commit"] = "not-a-commit"
        cases.append(("malformed_commit", malformed_commit, "Git object ID"))
        for name, template, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "job.json"
                with self.assertRaisesRegex(ValueError, message):
                    JobResultStore.open(
                        path,
                        template,
                        metric="perplexity",
                        resume=False,
                    )
                self.assertFalse(path.exists())

    def test_batch_plan_is_stable_across_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.json"
            store = JobResultStore.open(
                path,
                _job_template(),
                metric="perplexity",
                resume=False,
            )
            store.ensure_batch_plan(42, _batch_plan())
            different = {**_batch_plan(), "selected_pool_indices": [2, 1, 0]}
            with self.assertRaisesRegex(ValueError, "batch plan"):
                store.ensure_batch_plan(42, different)

    def test_store_mutations_commit_only_after_successful_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.json"
            store = JobResultStore.open(
                path,
                _job_template(),
                metric="perplexity",
                resume=False,
            )
            stable = path.read_bytes()
            with (
                mock.patch(
                    "experiments.lib.paper_results.write_json",
                    side_effect=OSError("disk full"),
                ),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                store.ensure_batch_plan(42, _batch_plan())
            self.assertEqual(store.payload["batch_plans"], {})
            self.assertEqual(path.read_bytes(), stable)

            store.ensure_batch_plan(42, _batch_plan())
            stable = path.read_bytes()
            with (
                mock.patch(
                    "experiments.lib.paper_results.write_json",
                    side_effect=OSError("disk full"),
                ),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                store.append_result(_record())
            self.assertFalse(store.has_result(42, "dacp"))
            self.assertEqual(store.payload["results"], [])
            self.assertEqual(path.read_bytes(), stable)

            store.append_result(_record())
            wrong_aggregate = copy.deepcopy(_aggregate())
            wrong_aggregate["methods"]["dacp"]["mean"] = 999.0
            with self.assertRaisesRegex(ValueError, "aggregate does not match"):
                store.finalize(wrong_aggregate, finished_at=NOW)
            self.assertFalse(store.complete)

            stable = path.read_bytes()
            with (
                mock.patch(
                    "experiments.lib.paper_results.write_json",
                    side_effect=OSError("disk full"),
                ),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                store.finalize(_aggregate(), finished_at=NOW)
            self.assertFalse(store.complete)
            self.assertEqual(path.read_bytes(), stable)

    def test_invalid_new_store_templates_do_not_leave_state_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_path = root / "job.json"
            bad_job = _job_template()
            bad_job["evaluation_batches"] = {"count": 0, "sha256": ""}
            with self.assertRaisesRegex(ValueError, "evaluation batch identity"):
                JobResultStore.open(
                    job_path,
                    bad_job,
                    metric="perplexity",
                    resume=False,
                )
            self.assertFalse(job_path.exists())

            suite_path = root / "suite_manifest.json"
            bad_suite = {
                "schema_version": SCHEMA_VERSION,
                "status": "started",
                "started_at": NOW,
                "provenance": _source_provenance(),
                "plan": {"jobs": [], "methods": []},
                "planned_jobs": ["missing-config"],
                "completed_jobs": {},
                "data_identities": {},
            }
            with self.assertRaisesRegex(ValueError, "invalid executable plan"):
                SuiteResultStore.open(suite_path, bad_suite, resume=False)
            self.assertFalse(suite_path.exists())

            forged_id = {
                "schema_version": SCHEMA_VERSION,
                "status": "started",
                "started_at": NOW,
                "provenance": _source_provenance(),
                "plan": {
                    "jobs": [_job_template()["config"]],
                    "methods": _job_template()["method_contracts"],
                },
                "planned_jobs": ["forged"],
                "completed_jobs": {},
                "data_identities": {},
            }
            with self.assertRaisesRegex(ValueError, "IDs do not match"):
                SuiteResultStore.open(suite_path, forged_id, resume=False)
            self.assertFalse(suite_path.exists())

    def test_completed_job_rejects_tampered_aggregate_and_batch_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = _job_template()
            payload["status"] = "complete"
            payload["finished_at"] = NOW
            payload["batch_plans"] = {"42": _batch_plan()}
            payload["results"] = [_record()]
            payload["aggregate"] = _aggregate()

            aggregate_path = root / "aggregate.json"
            aggregate_payload = copy.deepcopy(payload)
            aggregate_payload["aggregate"]["methods"]["dacp"]["mean"] = 999.0
            write_json(aggregate_path, aggregate_payload)
            with self.assertRaisesRegex(ValueError, "aggregate does not match"):
                JobResultStore.open(
                    aggregate_path,
                    _job_template(),
                    metric="perplexity",
                    resume=True,
                )

            plan_path = root / "plan.json"
            plan_payload = copy.deepcopy(payload)
            plan_payload["batch_plans"]["42"]["selected_pool_indices"] = [0, 0, 2]
            write_json(plan_path, plan_payload)
            with self.assertRaisesRegex(ValueError, "selected indices"):
                JobResultStore.open(
                    plan_path,
                    _job_template(),
                    metric="perplexity",
                    resume=True,
                )

    def test_suite_store_validates_job_digest_and_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_payload = _job_template()
            job_id = _job_id(job_payload)
            job_path = root / f"{job_id}.json"
            job_payload["status"] = "complete"
            job_payload["finished_at"] = NOW
            job_payload["batch_plans"] = {"42": _batch_plan()}
            job_payload["results"] = [_record()]
            job_payload["aggregate"] = _aggregate()
            write_json(job_path, job_payload)

            template = {
                "schema_version": SCHEMA_VERSION,
                "status": "started",
                "started_at": NOW,
                "provenance": _source_provenance(),
                "plan": {
                    "jobs": [job_payload["config"]],
                    "methods": job_payload["method_contracts"],
                },
                "planned_jobs": [job_id],
                "completed_jobs": {},
                "data_identities": {},
            }
            path = root / "suite_manifest.json"
            suite = SuiteResultStore.open(path, template, resume=False)
            suite.record_completed_job(job_id, job_path)
            suite.finalize(finished_at=NOW)
            self.assertTrue(suite.complete)

            resumed = SuiteResultStore.open(path, template, resume=True)
            resumed.validate_completed_jobs()
            self.assertTrue(resumed.has_completed_job(job_id))

            tampered = copy.deepcopy(job_payload)
            tampered["tampered"] = True
            write_json(job_path, tampered)
            self.assertNotEqual(
                sha256_file(job_path), suite.payload["completed_jobs"][job_id]["sha256"]
            )
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                resumed.validate_completed_jobs()

    def test_suite_rejects_jobs_outside_its_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_payload = _job_template()
            job_id = _job_id(job_payload)
            job_path = root / f"{job_id}.json"
            job_payload["status"] = "complete"
            job_payload["batch_plans"] = {"42": _batch_plan()}
            job_payload["results"] = [_record()]
            job_payload["aggregate"] = _aggregate()

            template = {
                "schema_version": SCHEMA_VERSION,
                "status": "started",
                "started_at": NOW,
                "provenance": _source_provenance(),
                "plan": {
                    "jobs": [job_payload["config"]],
                    "methods": job_payload["method_contracts"],
                },
                "planned_jobs": [job_id],
                "completed_jobs": {},
                "data_identities": {},
            }
            suite = SuiteResultStore.open(
                root / "suite_manifest.json",
                template,
                resume=False,
            )

            job_payload["started_at"] = BEFORE_NOW
            job_payload["finished_at"] = NOW
            write_json(job_path, job_payload)
            with self.assertRaisesRegex(ValueError, "started before its suite"):
                suite.record_completed_job(job_id, job_path)

            job_payload["started_at"] = NOW
            job_payload["finished_at"] = AFTER_NOW
            write_json(job_path, job_payload)
            suite.record_completed_job(job_id, job_path)
            with self.assertRaisesRegex(ValueError, "finished after its suite"):
                suite.finalize(finished_at=NOW)
            self.assertFalse(suite.complete)

    def test_suite_registration_commits_only_after_successful_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_payload = _job_template()
            job_id = _job_id(job_payload)
            job_path = root / f"{job_id}.json"
            job_payload["status"] = "complete"
            job_payload["finished_at"] = NOW
            job_payload["batch_plans"] = {"42": _batch_plan()}
            job_payload["results"] = [_record()]
            job_payload["aggregate"] = _aggregate()
            write_json(job_path, job_payload)
            template = {
                "schema_version": SCHEMA_VERSION,
                "status": "started",
                "started_at": NOW,
                "provenance": _source_provenance(),
                "plan": {
                    "jobs": [job_payload["config"]],
                    "methods": job_payload["method_contracts"],
                },
                "planned_jobs": [job_id],
                "completed_jobs": {},
                "data_identities": {},
            }
            suite_path = root / "suite_manifest.json"
            suite = SuiteResultStore.open(suite_path, template, resume=False)
            stable = suite_path.read_bytes()

            with (
                mock.patch(
                    "experiments.lib.paper_results.write_json",
                    side_effect=OSError("disk full"),
                ),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                suite.record_completed_job(job_id, job_path)

            self.assertEqual(suite.payload["completed_jobs"], {})
            self.assertEqual(suite.payload["data_identities"], {})
            self.assertEqual(suite_path.read_bytes(), stable)

            wrong_schema = copy.deepcopy(job_payload)
            wrong_schema["schema_version"] = SCHEMA_VERSION - 1
            write_json(job_path, wrong_schema)
            with self.assertRaisesRegex(ValueError, "schema does not match"):
                suite.record_completed_job(job_id, job_path)

    def test_suite_rejects_data_identity_drift_between_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_payload = _job_template()
            first_payload["status"] = "complete"
            first_payload["finished_at"] = NOW
            first_payload["batch_plans"] = {"42": _batch_plan()}
            first_payload["results"] = [_record()]
            first_payload["aggregate"] = _aggregate()
            first_id = _job_id(first_payload)
            first_path = root / f"{first_id}.json"
            write_json(first_path, first_payload)

            second_payload = copy.deepcopy(first_payload)
            second_payload["config"]["prune_ratios"] = [0.5, 0.7]
            second_payload["config"]["prune_ratio"] = 0.7
            second_payload["results"][0]["prune_ratio"] = 0.7
            second_payload["results"][0]["cycles"][0]["compression"]["allocation"][
                "target_eligible_sparsity"
            ] = 0.7
            second_payload["training_pool"] = {"count": 3, "sha256": _digest("changed")}
            second_id = _job_id(second_payload)
            second_path = root / f"{second_id}.json"
            write_json(second_path, second_payload)

            template = {
                "schema_version": SCHEMA_VERSION,
                "status": "started",
                "started_at": NOW,
                "provenance": _source_provenance(),
                "plan": {
                    "jobs": [first_payload["config"], second_payload["config"]],
                    "methods": first_payload["method_contracts"],
                },
                "planned_jobs": [first_id, second_id],
                "completed_jobs": {},
                "data_identities": {},
            }
            suite = SuiteResultStore.open(
                root / "suite_manifest.json",
                template,
                resume=False,
            )
            suite.record_completed_job(first_id, first_path)
            with self.assertRaisesRegex(ValueError, "data identity changed"):
                suite.record_completed_job(second_id, second_path)

    def test_existing_output_requires_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.json"
            write_json(path, _job_template())
            with self.assertRaisesRegex(FileExistsError, "--resume"):
                JobResultStore.open(
                    path,
                    _job_template(),
                    metric="perplexity",
                    resume=False,
                )

    def test_suite_rejects_false_completion_and_unknown_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = {
                "schema_version": SCHEMA_VERSION,
                "status": "started",
                "started_at": NOW,
                "provenance": _source_provenance(),
                "plan": {
                    "jobs": [_job_template()["config"]],
                    "methods": _job_template()["method_contracts"],
                },
                "planned_jobs": [_job_id()],
                "completed_jobs": {},
                "data_identities": {},
            }
            path = root / "suite_manifest.json"

            false_complete = copy.deepcopy(template)
            false_complete["status"] = "complete"
            false_complete["finished_at"] = NOW
            write_json(path, false_complete)
            with self.assertRaisesRegex(ValueError, "missing jobs"):
                SuiteResultStore.open(path, template, resume=True)

            unknown_identity = copy.deepcopy(template)
            unknown_identity["data_identities"] = {"other": {}}
            write_json(path, unknown_identity)
            with self.assertRaisesRegex(ValueError, "unplanned workload"):
                SuiteResultStore.open(path, template, resume=True)

    def test_suite_rejects_inconsistent_lifecycle_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = {
                "schema_version": SCHEMA_VERSION,
                "status": "started",
                "started_at": NOW,
                "provenance": _source_provenance(),
                "plan": {
                    "jobs": [_job_template()["config"]],
                    "methods": _job_template()["method_contracts"],
                },
                "planned_jobs": [_job_id()],
                "completed_jobs": {},
                "data_identities": {},
            }
            path = root / "suite_manifest.json"

            premature = copy.deepcopy(template)
            premature["finished_at"] = NOW
            write_json(path, premature)
            with self.assertRaisesRegex(ValueError, "completion fields while started"):
                SuiteResultStore.open(path, template, resume=True)

            reversed_timeline = copy.deepcopy(template)
            reversed_timeline["status"] = "complete"
            reversed_timeline["finished_at"] = BEFORE_NOW
            write_json(path, reversed_timeline)
            with self.assertRaisesRegex(ValueError, "precedes started_at"):
                SuiteResultStore.open(path, template, resume=True)

    def test_suite_finalize_rejects_finished_at_before_started_at(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "suite_manifest.json"
            template = {
                "schema_version": SCHEMA_VERSION,
                "status": "started",
                "started_at": NOW,
                "provenance": _source_provenance(),
                "plan": {
                    "jobs": [_job_template()["config"]],
                    "methods": _job_template()["method_contracts"],
                },
                "planned_jobs": [_job_id()],
                "completed_jobs": {},
                "data_identities": {},
            }
            suite = SuiteResultStore.open(path, template, resume=False)
            stable = path.read_bytes()

            with self.assertRaisesRegex(ValueError, "precedes started_at"):
                suite.finalize(finished_at=BEFORE_NOW)

            self.assertFalse(suite.complete)
            self.assertEqual(path.read_bytes(), stable)

    def test_store_snapshots_cannot_mutate_internal_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job = JobResultStore.open(
                root / "job.json",
                _job_template(),
                metric="perplexity",
                resume=False,
            )
            job_snapshot = job.payload
            job_snapshot["status"] = "complete"
            job_snapshot["batch_plans"]["42"] = _batch_plan()
            self.assertFalse(job.complete)
            self.assertEqual(job.payload["batch_plans"], {})

            records = records_from(job)
            self.assertIsInstance(records, tuple)

            suite_template = {
                "schema_version": SCHEMA_VERSION,
                "status": "started",
                "started_at": NOW,
                "provenance": _source_provenance(),
                "plan": {
                    "jobs": [_job_template()["config"]],
                    "methods": _job_template()["method_contracts"],
                },
                "planned_jobs": [_job_id()],
                "completed_jobs": {},
                "data_identities": {},
            }
            suite = SuiteResultStore.open(
                root / "suite_manifest.json",
                suite_template,
                resume=False,
            )
            suite_snapshot = suite.payload
            suite_snapshot["status"] = "complete"
            suite_snapshot["completed_jobs"]["forged"] = {}
            self.assertFalse(suite.complete)
            self.assertEqual(suite.payload["completed_jobs"], {})


if __name__ == "__main__":
    unittest.main()
