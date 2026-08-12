"""Run declaration-driven, paired residual-recovery experiments for the paper."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.data import cache_batches, get_data_loaders  # noqa: E402
from experiments.lib.models import load_model  # noqa: E402
from experiments.lib.paper_manifest import load_paper_manifest  # noqa: E402
from experiments.lib.paper_runner import (  # noqa: E402
    aggregate_job_results,
    plan_jobs,
    run_method_trajectory,
)
from experiments.lib.residual_runtime import (  # noqa: E402
    batch_hash,
    configure_hf_offline,
    load_training_checkpoint,
    sha256_file,
    set_seed,
    write_json,
)
from experiments.lib.residual_training import (  # noqa: E402
    partition_repeated_seed_batches,
)


def _csv(values: str | None, cast) -> tuple:
    if values is None:
        return ()
    return tuple(cast(value.strip()) for value in values.split(",") if value.strip())


def _source_provenance(manifest_path: Path) -> dict[str, object]:
    command = ["git", "status", "--porcelain", "--untracked-files=no"]
    status = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "git_commit": commit.stdout.strip(),
        "git_dirty": bool(status.stdout.strip()),
    }


def _remap(path: Path, declared_root: Path, override: Path | None) -> Path:
    if override is None:
        return path
    try:
        relative = path.relative_to(declared_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Cannot remap {path}; it is outside {declared_root}") from exc
    return (override.resolve() / relative).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments/configs/paper_experiments.yaml",
    )
    parser.add_argument("--workload", help="Comma-separated workload names")
    parser.add_argument("--prune-ratio", help="Comma-separated declared ratios")
    parser.add_argument("--recoveries", help="Comma-separated declared K values")
    parser.add_argument("--seeds", help="Comma-separated declared seeds")
    parser.add_argument("--methods", help="Comma-separated manifest method names")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/paper_runs"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = load_paper_manifest(args.manifest)
    jobs = plan_jobs(
        manifest.workloads,
        workload_names=_csv(args.workload, str),
        prune_ratios=_csv(args.prune_ratio, float),
        recovery_counts=_csv(args.recoveries, int),
        seeds=_csv(args.seeds, int),
    )
    contracts = list(manifest.methods)
    if args.methods:
        requested = _csv(args.methods, str)
        by_name = {contract.name: contract for contract in contracts}
        unknown = set(requested) - set(by_name)
        if unknown:
            raise ValueError(f"Methods are not declared in the manifest: {sorted(unknown)}")
        contracts = [by_name[name] for name in requested]

    plan = {
        "jobs": [job.to_result_dict() for job in jobs],
        "methods": [contract.to_result_dict() for contract in contracts],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, default=str))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    configure_hf_offline()
    provenance = _source_provenance(manifest.path)
    suite_path = args.output_dir / "suite_manifest.json"
    suite = {
        "schema_version": 1,
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "plan": plan,
        "planned_jobs": [job.job_id for job in jobs],
        "completed_jobs": {},
    }
    write_json(suite_path, suite)

    declared_checkpoint_root = ROOT / "checkpoints"
    declared_data_root = ROOT / "data"
    for job in jobs:
        workload = job.workload
        checkpoint_path = _remap(
            workload.checkpoint, declared_checkpoint_root, args.checkpoint_root
        )
        data_dir = _remap(workload.data_dir, declared_data_root, args.data_root)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        checkpoint = load_training_checkpoint(checkpoint_path)
        set_seed(0)
        train_loader, val_loader, task_type = get_data_loaders(
            workload.model,
            workload.dataset,
            workload.batch_size,
            workload.seq_length,
            data_dir=str(data_dir),
        )
        if task_type != workload.task_type:
            raise RuntimeError("Data loader task type does not match the manifest")
        training_pool = cache_batches(
            train_loader, workload.train_pool_batches, workload.task_type
        )
        eval_batches = cache_batches(
            val_loader, workload.eval_batches, workload.task_type
        )
        if len(training_pool) < workload.total_steps + job.recovery_count * workload.hvp_batches:
            raise RuntimeError("Training loader did not provide the declared batch pool")

        model_factory = lambda workload=workload: load_model(
            workload.model,
            pretrained=False,
            checkpoint_path=None,
            device="cpu",
            dataset_name=workload.dataset,
        )[0]
        job_path = args.output_dir / f"{job.job_id}.json"
        result = {
            "schema_version": 1,
            "status": "started",
            "provenance": {
                **provenance,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            },
            "config": job.to_result_dict(),
            "method_contracts": [
                contract.to_result_dict() for contract in contracts
            ],
            "evaluation_batches": {
                "count": len(eval_batches),
                "sha256": batch_hash(eval_batches),
            },
            "batch_plans": {},
            "results": [],
        }
        write_json(job_path, result)
        for seed in job.seeds:
            batch_plan = partition_repeated_seed_batches(
                training_pool,
                workload.total_steps,
                job.recovery_count,
                workload.hvp_batches,
                seed,
            )
            result["batch_plans"][str(seed)] = {
                "selected_pool_indices": batch_plan.selected_pool_indices,
                **batch_plan.data_hashes(),
            }
            for contract in contracts:
                record = run_method_trajectory(
                    job,
                    contract,
                    seed,
                    batch_plan,
                    model_factory,
                    checkpoint.model_state,
                    checkpoint.optimizer_state,
                    eval_batches,
                    args.device,
                )
                result["results"].append(record)
                write_json(job_path, result)
        result["aggregate"] = aggregate_job_results(
            result["results"], workload.metric
        )
        result["status"] = "complete"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(job_path, result)
        suite["completed_jobs"][job.job_id] = {
            "path": job_path.name,
            "sha256": sha256_file(job_path),
        }
        write_json(suite_path, suite)

    suite["status"] = "complete"
    suite["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(suite_path, suite)


if __name__ == "__main__":
    main()
