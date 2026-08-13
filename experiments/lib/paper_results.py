"""Atomic, resume-safe persistence for paper experiment suites."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from experiments.lib.paper_baselines import resolve_method_contracts
from experiments.lib.paper_manifest import paper_job_id
from experiments.lib.paper_result_validation import (
    SCHEMA_VERSION,
    _canonical_json_object,
    _expected_aggregate,
    _load_json,
    _require_equal,
    _require_list,
    _require_mapping,
    _require_provenance_compatible,
    _require_sha256,
    _validate_batch_plans,
    _validate_checkpoint_provenance,
    _validate_complete_record,
    _validate_evaluation_identity,
    _validate_job_config,
    _validate_lifecycle,
    _validate_nested_lifecycle,
)
from experiments.lib.residual_runtime import sha256_file, write_json


def _canonical_method_contracts(value: object) -> list[Mapping[str, object]]:
    """Validate paper-facing method metadata against the executable registry."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(contract, Mapping) for contract in value)
    ):
        raise ValueError("Paper result state has invalid method contracts")
    names = [contract.get("name") for contract in value]
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("Paper result state has invalid method contract names")
    canonical = [contract.to_result_dict() for contract in resolve_method_contracts(names)]
    if value != canonical:
        raise ValueError("Paper result method contracts do not match the executable registry")
    return value


def _planned_job_configs(
    payload: Mapping[str, object],
) -> tuple[dict[str, Mapping[str, object]], list]:
    """Validate and bind suite job identifiers to their executable configs."""
    planned = payload.get("planned_jobs")
    plan = payload.get("plan")
    if not isinstance(planned, list) or not isinstance(plan, Mapping):
        raise ValueError("Suite state has invalid planned jobs or plan")
    configs = plan.get("jobs")
    methods = plan.get("methods")
    if (
        not isinstance(configs, list)
        or len(configs) != len(planned)
        or not all(isinstance(config, Mapping) for config in configs)
        or not all(isinstance(job_id, str) and job_id for job_id in planned)
    ):
        raise ValueError("Suite state has an invalid executable plan")

    expected_ids = []
    for config in configs:
        _validate_job_config(config, context="Suite planned job config")
        try:
            expected_ids.append(
                paper_job_id(
                    config.get("name"),
                    config.get("prune_ratio"),
                    config.get("recovery_count"),
                )
            )
        except ValueError as exc:
            raise ValueError("Suite state has an invalid planned job config") from exc
    if planned != expected_ids:
        raise ValueError("Suite planned job IDs do not match their configs")
    if len(set(planned)) != len(planned):
        raise ValueError("Suite state has duplicate planned jobs")
    return dict(zip(planned, configs)), _canonical_method_contracts(methods)


@dataclass
class JobResultStore:
    """Mutable state for one job, persisted after every completed unit of work."""

    path: Path
    _payload: MutableMapping[str, object]
    metric: str
    _method_contracts: Mapping[str, Mapping[str, object]]
    _record_keys: set[tuple[int, str]]

    @classmethod
    def open(
        cls,
        path: Path,
        template: Mapping[str, object],
        *,
        metric: str,
        resume: bool,
    ) -> "JobResultStore":
        path = Path(path)
        template = _canonical_json_object(template)
        creating = not path.exists()
        if not creating:
            if not resume:
                raise FileExistsError(
                    f"Job output already exists: {path}. Use --resume or a new output directory."
                )
            payload: MutableMapping[str, object] = _load_json(path)
            for key in (
                "schema_version",
                "config",
                "method_contracts",
                "training_pool",
                "evaluation_batches",
            ):
                _require_equal(payload, key, template.get(key), context=f"Job state {path}")
        else:
            payload = dict(template)

        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Job state {path} has an unsupported schema version")
        _require_provenance_compatible(
            payload.get("provenance"),
            template.get("provenance"),
            keys=(
                "manifest_sha256",
                "source_state_sha256",
                "checkpoint_sha256",
                "checkpoint_step",
            ),
            context=f"Job state {path}",
        )
        _validate_lifecycle(
            payload,
            context=f"Job state {path}",
            completion_fields=("finished_at", "aggregate"),
        )

        config = payload.get("config")
        contracts = payload.get("method_contracts")
        records = payload.get("results")
        plans = payload.get("batch_plans")
        if not isinstance(config, Mapping):
            raise ValueError(f"Job state {path} has invalid config")
        _validate_job_config(config, context=f"Job state {path} config")
        if config.get("metric") != metric:
            raise ValueError(f"Job state {path} metric does not match the store")
        _validate_checkpoint_provenance(
            payload.get("provenance"),
            config,
            context=f"Job state {path} provenance",
        )
        contracts = _canonical_method_contracts(contracts)
        if not isinstance(records, list) or not all(
            isinstance(record, Mapping) for record in records
        ):
            raise ValueError(f"Job state {path} has invalid results")
        if not isinstance(plans, dict):
            raise ValueError(f"Job state {path} has invalid batch plans")

        expected_plan_keys = {str(seed) for seed in config.get("seeds", ())}
        unknown_plan_keys = set(plans) - expected_plan_keys
        if unknown_plan_keys:
            raise ValueError(
                f"Job state {path} has undeclared batch plans: {sorted(unknown_plan_keys)}"
            )
        if not all(isinstance(plan, Mapping) for plan in plans.values()):
            raise ValueError(f"Job state {path} has malformed batch plans")
        _validate_batch_plans(
            plans,
            config,
            payload.get("training_pool"),
            path=path,
        )
        _validate_evaluation_identity(payload.get("evaluation_batches"), config, path=path)

        contract_names = [str(contract["name"]) for contract in contracts]
        by_method = {name: contract for name, contract in zip(contract_names, contracts)}
        if len(by_method) != len(contracts):
            raise ValueError(f"Job state {path} has duplicate or invalid method contracts")
        record_keys: set[tuple[int, str]] = set()
        for record in records:
            record_key = _validate_complete_record(
                record,
                expected_config=config,
                method_contracts=by_method,
                metric=metric,
            )
            if record_key in record_keys:
                raise ValueError(f"Job state {path} has duplicate result {record_key}")
            if str(record_key[0]) not in plans:
                raise ValueError(f"Job state {path} has a result without its batch plan")
            record_keys.add(record_key)

        expected_total = len(config.get("seeds", ())) * len(contracts)
        if payload.get("status") == "complete" and len(record_keys) != expected_total:
            raise ValueError(f"Completed job state {path} is missing results")
        if payload.get("status") == "complete":
            expected_aggregate = _expected_aggregate(records, metric)
            _require_equal(
                payload,
                "aggregate",
                expected_aggregate,
                context=f"Completed job state {path}",
            )
        store = cls(path, payload, metric, by_method, record_keys)
        if creating:
            write_json(path, payload)
        return store

    @property
    def complete(self) -> bool:
        return self._payload.get("status") == "complete"

    @property
    def payload(self) -> MutableMapping[str, object]:
        """Return a detached snapshot so callers cannot bypass store validation."""
        return copy.deepcopy(self._payload)

    def ensure_batch_plan(self, seed: int, batch_plan: Mapping[str, object]) -> None:
        if self.complete:
            raise ValueError(f"Cannot modify completed job state {self.path}")
        plans = _require_mapping(
            self._payload.get("batch_plans"),
            context=f"Job state {self.path} batch_plans",
        )
        key = str(seed)
        existing = plans.get(key)
        if existing is not None and existing != batch_plan:
            raise ValueError(f"Stored batch plan for seed {seed} is incompatible")
        if existing is None:
            candidate = copy.deepcopy(self._payload)
            candidate_plans = _require_mapping(
                candidate.get("batch_plans"),
                context=f"Job state {self.path} batch_plans",
            )
            candidate_plans[key] = copy.deepcopy(batch_plan)
            config = candidate.get("config")
            if not isinstance(config, Mapping):
                raise ValueError(f"Job state {self.path} has invalid config")
            expected_plan_keys = {str(value) for value in config.get("seeds", ())}
            if key not in expected_plan_keys:
                raise ValueError(f"Cannot store a batch plan for undeclared seed {seed}")
            _validate_batch_plans(
                candidate_plans,
                config,
                candidate.get("training_pool"),
                path=self.path,
            )
            write_json(self.path, candidate)
            self._payload = candidate

    def has_result(self, seed: int, method: str) -> bool:
        return (seed, method) in self._record_keys

    def append_result(self, record: Mapping[str, object]) -> None:
        if self.complete:
            raise ValueError(f"Cannot modify completed job state {self.path}")
        config = self._payload["config"]
        if not isinstance(config, Mapping):
            raise ValueError(f"Job state {self.path} has invalid config")
        key = _validate_complete_record(
            record,
            expected_config=config,
            method_contracts=self._method_contracts,
            metric=self.metric,
        )
        if key in self._record_keys:
            raise ValueError(f"Refusing to append duplicate result {key}")
        plans = _require_mapping(
            self._payload.get("batch_plans"),
            context=f"Job state {self.path} batch_plans",
        )
        if str(key[0]) not in plans:
            raise ValueError(f"Refusing to append result {key} without its batch plan")
        candidate = copy.deepcopy(self._payload)
        records = _require_list(
            candidate.get("results"),
            context=f"Job state {self.path} results",
        )
        records.append(copy.deepcopy(record))
        write_json(self.path, candidate)
        self._payload = candidate
        self._record_keys.add(key)

    def finalize(
        self,
        aggregate: Mapping[str, object],
        *,
        finished_at: str,
    ) -> None:
        config = self._payload["config"]
        contracts = self._payload["method_contracts"]
        if not isinstance(config, Mapping) or not isinstance(contracts, list):
            raise ValueError(f"Job state {self.path} has invalid completion metadata")
        expected_total = len(config["seeds"]) * len(contracts)
        if len(self._record_keys) != expected_total:
            raise ValueError(
                f"Cannot finalize {self.path}: expected {expected_total} results, "
                f"found {len(self._record_keys)}"
            )
        records = _require_list(
            self._payload.get("results"),
            context=f"Job state {self.path} results",
        )
        validated_keys = {
            _validate_complete_record(
                record,
                expected_config=config,
                method_contracts=self._method_contracts,
                metric=self.metric,
            )
            for record in records
        }
        if validated_keys != self._record_keys:
            raise ValueError(f"Job state {self.path} records changed after validation")
        expected_aggregate = _expected_aggregate(records, self.metric)
        if aggregate != expected_aggregate:
            raise ValueError(f"Job state {self.path} aggregate does not match its results")
        if not isinstance(finished_at, str) or not finished_at:
            raise ValueError("finished_at must be a non-empty string")
        if self.complete:
            _require_equal(
                self._payload,
                "aggregate",
                expected_aggregate,
                context=f"Job state {self.path}",
            )
            return
        candidate = copy.deepcopy(self._payload)
        candidate["aggregate"] = dict(expected_aggregate)
        candidate["status"] = "complete"
        candidate["finished_at"] = finished_at
        _validate_lifecycle(
            candidate,
            context=f"Job state {self.path}",
            completion_fields=("finished_at", "aggregate"),
        )
        write_json(self.path, candidate)
        self._payload = candidate


@dataclass
class SuiteResultStore:
    """Top-level suite state and digest registry for completed jobs."""

    path: Path
    _payload: MutableMapping[str, object]

    @classmethod
    def open(
        cls,
        path: Path,
        template: Mapping[str, object],
        *,
        resume: bool,
    ) -> "SuiteResultStore":
        path = Path(path)
        template = _canonical_json_object(template)
        creating = not path.exists()
        if not creating:
            if not resume:
                raise FileExistsError(
                    f"Suite output already exists: {path}. Use --resume or a new output directory."
                )
            payload: MutableMapping[str, object] = _load_json(path)
            for key in ("schema_version", "plan", "planned_jobs"):
                _require_equal(payload, key, template.get(key), context=f"Suite state {path}")
            if not isinstance(payload.get("completed_jobs"), dict):
                raise ValueError(f"Suite state {path} has invalid completed_jobs")
            if not isinstance(payload.get("data_identities"), dict):
                raise ValueError(f"Suite state {path} has invalid data_identities")
        else:
            if resume:
                raise FileNotFoundError(f"Cannot resume missing suite state: {path}")
            payload = dict(template)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Suite state {path} has an unsupported schema version")
        _require_provenance_compatible(
            payload.get("provenance"),
            template.get("provenance"),
            keys=("manifest_sha256", "source_state_sha256"),
            context=f"Suite state {path}",
        )
        _validate_lifecycle(
            payload,
            context=f"Suite state {path}",
            completion_fields=("finished_at",),
        )
        store = cls(path, payload)
        store.validate_completed_jobs()
        if creating:
            write_json(path, payload)
        return store

    @property
    def complete(self) -> bool:
        return self._payload.get("status") == "complete"

    @property
    def payload(self) -> MutableMapping[str, object]:
        """Return a detached snapshot so callers cannot mutate suite state in place."""
        return copy.deepcopy(self._payload)

    def has_completed_job(self, job_id: str) -> bool:
        completed = _require_mapping(
            self._payload.get("completed_jobs"),
            context=f"Suite state {self.path} completed_jobs",
        )
        return job_id in completed

    def expected_data_identity(self, workload_name: str) -> Mapping[str, object] | None:
        identities = _require_mapping(
            self._payload.get("data_identities"),
            context=f"Suite state {self.path} data_identities",
        )
        identity = identities.get(workload_name)
        if identity is not None and not isinstance(identity, Mapping):
            raise ValueError(f"Suite workload {workload_name} has invalid data identity")
        return copy.deepcopy(identity)

    def record_completed_job(self, job_id: str, job_path: Path) -> None:
        job_path = Path(job_path).resolve()
        if job_path.parent != self.path.parent.resolve() or not job_path.is_file():
            raise ValueError(f"Completed suite job {job_id} points to an invalid path")
        if job_path.name != f"{job_id}.json":
            raise ValueError(f"Completed suite job {job_id} must use {job_id}.json")
        planned_configs, planned_methods = _planned_job_configs(self._payload)
        if job_id not in planned_configs:
            raise ValueError(f"Cannot register unplanned suite job {job_id}")
        expected_config = planned_configs[job_id]
        job_payload = _load_json(job_path)
        if job_payload.get("status") != "complete":
            raise ValueError(f"Cannot register incomplete job: {job_path}")
        if job_payload.get("schema_version") != self._payload.get("schema_version"):
            raise ValueError(f"Completed job {job_id} schema does not match the suite")
        config = job_payload.get("config")
        if not isinstance(config, Mapping):
            raise ValueError(f"Completed job {job_id} has invalid config")
        if config != expected_config:
            raise ValueError(f"Completed job {job_id} config does not match the suite plan")
        if job_payload.get("method_contracts") != planned_methods:
            raise ValueError(f"Completed job {job_id} methods do not match the suite plan")
        suite_provenance = self._payload.get("provenance")
        job_provenance = job_payload.get("provenance")
        if not isinstance(suite_provenance, Mapping) or not isinstance(job_provenance, Mapping):
            raise ValueError(f"Completed job {job_id} has invalid provenance")
        if any(job_provenance.get(key) != value for key, value in suite_provenance.items()):
            raise ValueError(f"Completed job {job_id} provenance does not match the suite")
        metric = config.get("metric")
        if not isinstance(metric, str) or not metric:
            raise ValueError(f"Completed job {job_id} has no declared metric")
        JobResultStore.open(job_path, job_payload, metric=metric, resume=True)
        _validate_nested_lifecycle(
            self._payload,
            job_payload,
            context=f"Completed job {job_id}",
        )
        workload_name = config.get("name")
        if not isinstance(workload_name, str) or not workload_name:
            raise ValueError(f"Completed job {job_id} has invalid workload name")
        identity = {
            "checkpoint_sha256": job_provenance.get("checkpoint_sha256"),
            "checkpoint_step": job_provenance.get("checkpoint_step"),
            "training_pool": job_payload.get("training_pool"),
            "evaluation_batches": job_payload.get("evaluation_batches"),
        }
        candidate = copy.deepcopy(self._payload)
        identities = _require_mapping(
            candidate.get("data_identities"),
            context=f"Suite state {self.path} data_identities",
        )
        existing_identity = identities.get(workload_name)
        if existing_identity is not None and existing_identity != identity:
            raise ValueError(f"Suite workload {workload_name} data identity changed")
        if existing_identity is None:
            identities[workload_name] = identity
        entry = {"path": job_path.name, "sha256": sha256_file(job_path)}
        completed = _require_mapping(
            candidate.get("completed_jobs"),
            context=f"Suite state {self.path} completed_jobs",
        )
        existing = completed.get(job_id)
        if existing is not None and existing != entry:
            raise ValueError(f"Completed suite job {job_id} has a digest mismatch")
        if existing is None:
            completed[job_id] = entry
            write_json(self.path, candidate)
            self._payload = candidate

    def validate_completed_jobs(self) -> None:
        completed = self._payload.get("completed_jobs")
        provenance = self._payload.get("provenance")
        if not isinstance(completed, dict) or not isinstance(provenance, Mapping):
            raise ValueError("Suite state has invalid completed jobs or provenance")
        expected_configs, planned_methods = _planned_job_configs(self._payload)
        planned = list(expected_configs)
        planned_configs = list(expected_configs.values())
        identities = self._payload.get("data_identities")
        if not isinstance(identities, dict):
            raise ValueError("Suite state has invalid data identities")
        planned_workloads = {
            config.get("name")
            for config in planned_configs
            if isinstance(config.get("name"), str) and config.get("name")
        }
        unknown_identities = set(identities) - planned_workloads
        if unknown_identities:
            raise ValueError(
                "Suite state contains unplanned workload identities: "
                f"{sorted(unknown_identities)}"
            )
        unknown = set(completed) - set(planned)
        if unknown:
            raise ValueError(f"Suite state contains unplanned completed jobs: {sorted(unknown)}")
        for job_id, raw_entry in completed.items():
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"Suite job {job_id} has invalid digest metadata")
            relative_path = raw_entry.get("path")
            digest = raw_entry.get("sha256")
            if not isinstance(relative_path, str) or not isinstance(digest, str):
                raise ValueError(f"Suite job {job_id} has invalid digest metadata")
            _require_sha256(digest, context=f"Suite job {job_id} digest")
            if relative_path != f"{job_id}.json":
                raise ValueError(f"Suite job {job_id} must use {job_id}.json")
            job_path = (self.path.parent / relative_path).resolve()
            if job_path.parent != self.path.parent.resolve() or not job_path.is_file():
                raise ValueError(f"Suite job {job_id} points to an invalid path")
            if sha256_file(job_path) != digest:
                raise ValueError(f"Suite job {job_id} has a digest mismatch")
            job_payload = _load_json(job_path)
            if job_payload.get("status") != "complete":
                raise ValueError(f"Suite job {job_id} is not complete")
            if job_payload.get("config") != expected_configs[job_id]:
                raise ValueError(f"Suite job {job_id} config does not match the plan")
            if job_payload.get("method_contracts") != planned_methods:
                raise ValueError(f"Suite job {job_id} methods do not match the plan")
            workload_name = expected_configs[job_id].get("name")
            job_provenance = job_payload.get("provenance")
            job_identity = {
                "checkpoint_sha256": (
                    job_provenance.get("checkpoint_sha256")
                    if isinstance(job_provenance, Mapping)
                    else None
                ),
                "checkpoint_step": (
                    job_provenance.get("checkpoint_step")
                    if isinstance(job_provenance, Mapping)
                    else None
                ),
                "training_pool": job_payload.get("training_pool"),
                "evaluation_batches": job_payload.get("evaluation_batches"),
            }
            if not isinstance(workload_name, str) or identities.get(workload_name) != job_identity:
                raise ValueError(f"Suite job {job_id} data identity does not match the suite")
            if not isinstance(job_provenance, Mapping) or any(
                job_provenance.get(key) != value for key, value in provenance.items()
            ):
                raise ValueError(f"Suite job {job_id} provenance does not match the suite")
            metric = expected_configs[job_id].get("metric")
            if not isinstance(metric, str) or not metric:
                raise ValueError(f"Suite job {job_id} has no declared metric")
            expected_job_template = {
                **job_payload,
                "schema_version": self._payload.get("schema_version"),
                "provenance": {
                    **provenance,
                    "checkpoint_sha256": job_provenance.get("checkpoint_sha256"),
                    "checkpoint_step": job_provenance.get("checkpoint_step"),
                },
                "config": expected_configs[job_id],
                "method_contracts": planned_methods,
                "training_pool": job_identity["training_pool"],
                "evaluation_batches": job_identity["evaluation_batches"],
            }
            JobResultStore.open(
                job_path,
                expected_job_template,
                metric=metric,
                resume=True,
            )
            _validate_nested_lifecycle(
                self._payload,
                job_payload,
                context=f"Suite job {job_id}",
            )
        if self.complete and set(completed) != set(planned):
            missing = sorted(set(planned) - set(completed))
            raise ValueError(f"Completed suite state is missing jobs: {missing}")

    def _validate_completed_job_lifecycles(
        self,
        suite_payload: Mapping[str, object],
    ) -> None:
        completed = _require_mapping(
            suite_payload.get("completed_jobs"),
            context=f"Suite state {self.path} completed_jobs",
        )
        for job_id, raw_entry in completed.items():
            if not isinstance(raw_entry, Mapping) or not isinstance(raw_entry.get("path"), str):
                raise ValueError(f"Suite job {job_id} has invalid digest metadata")
            job_payload = _load_json(self.path.parent / raw_entry["path"])
            _validate_nested_lifecycle(
                suite_payload,
                job_payload,
                context=f"Suite job {job_id}",
            )

    def finalize(self, *, finished_at: str) -> None:
        if not isinstance(finished_at, str) or not finished_at:
            raise ValueError("finished_at must be a non-empty string")
        candidate = copy.deepcopy(self._payload)
        candidate["status"] = "complete"
        candidate["finished_at"] = finished_at
        _validate_lifecycle(
            candidate,
            context=f"Suite state {self.path}",
            completion_fields=("finished_at",),
        )
        self.validate_completed_jobs()
        self._validate_completed_job_lifecycles(candidate)
        planned = self._payload["planned_jobs"]
        completed = self._payload["completed_jobs"]
        if not isinstance(planned, list) or not isinstance(completed, dict):
            raise ValueError("Suite state has invalid planned/completed jobs")
        if set(completed) != set(planned):
            missing = sorted(set(planned) - set(completed))
            raise ValueError(f"Cannot finalize suite; missing jobs: {missing}")
        if self.complete:
            return
        write_json(self.path, candidate)
        self._payload = candidate


def records_from(store: JobResultStore) -> Sequence[Mapping[str, object]]:
    records = _require_list(
        store._payload.get("results"),
        context=f"Job state {store.path} results",
    )
    return tuple(copy.deepcopy(record) for record in records)


__all__ = [
    "JobResultStore",
    "SCHEMA_VERSION",
    "SuiteResultStore",
    "records_from",
]
