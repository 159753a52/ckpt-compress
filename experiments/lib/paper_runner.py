"""Core execution engine for matched paper residual-recovery jobs."""

from __future__ import annotations

import gc
import math
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import torch

from experiments.lib.paper_baselines import MethodContract
from experiments.lib.paper_manifest import PaperWorkload, paper_job_id
from experiments.lib.residual_masks import global_mask, layer_rates, mask_metrics, restore_with_mask
from experiments.lib.residual_methods import build_weibull_mask
from experiments.lib.residual_runtime import empty_device_cache, evaluate_task
from experiments.lib.residual_scoring import (
    compute_block_first_order_scores,
    compute_block_taylor_scores,
    eligible_layers,
    transformer_layers,
)
from experiments.lib.residual_training import (
    RepeatedSeedBatchPlan,
    build_optimizer,
    clone_model_state_to_cpu,
    train_segment,
)

RECOVERY_SCHEDULER_NAME = "cosine_annealing"


def recovery_protocol(workload: PaperWorkload) -> dict[str, object]:
    """Describe the matched recovery schedule applied after the checkpoint."""
    return {
        "optimizer_state": "restored_from_checkpoint",
        "learning_rate_source": "manifest",
        "learning_rate": workload.learning_rate,
        "scheduler": {
            "name": RECOVERY_SCHEDULER_NAME,
            "state": "fresh",
            "scope": "matched_recovery_horizon",
            "t_max": workload.total_steps,
            "eta_min": 0.0,
        },
    }


@dataclass(frozen=True)
class PaperJob:
    workload: PaperWorkload
    prune_ratio: float
    recovery_count: int
    seeds: tuple[int, ...]

    @property
    def job_id(self) -> str:
        return paper_job_id(self.workload.name, self.prune_ratio, self.recovery_count)

    def to_result_dict(self) -> dict[str, object]:
        result = asdict(self.workload)
        result.update(
            {
                "checkpoint": str(self.workload.checkpoint),
                "data_dir": str(self.workload.data_dir),
                "prune_ratio": self.prune_ratio,
                "recovery_count": self.recovery_count,
                "seeds": list(self.seeds),
                "recovery_protocol": recovery_protocol(self.workload),
            }
        )
        return result


def plan_jobs(
    workloads: Sequence[PaperWorkload],
    *,
    workload_names: Sequence[str] = (),
    prune_ratios: Sequence[float] = (),
    recovery_counts: Sequence[int] = (),
    seeds: Sequence[int] = (),
) -> list[PaperJob]:
    """Expand workload declarations into deterministic executable jobs."""
    for label, values in (
        ("workloads", workload_names),
        ("prune ratios", prune_ratios),
        ("recovery counts", recovery_counts),
        ("seeds", seeds),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate requested {label} are not allowed")
    selected_names = set(workload_names)
    unknown = selected_names - {workload.name for workload in workloads}
    if unknown:
        raise ValueError(f"Unknown workloads: {sorted(unknown)}")
    jobs = []
    for workload in workloads:
        if selected_names and workload.name not in selected_names:
            continue
        ratios = tuple(prune_ratios) or workload.prune_ratios
        recoveries = tuple(recovery_counts) or workload.recovery_counts
        selected_seeds = tuple(seeds) or workload.seeds
        if not set(ratios).issubset(workload.prune_ratios):
            raise ValueError(f"{workload.name}: requested prune ratio is not declared")
        if not set(recoveries).issubset(workload.recovery_counts):
            raise ValueError(f"{workload.name}: requested recovery count is not declared")
        if not set(selected_seeds).issubset(workload.seeds):
            raise ValueError(f"{workload.name}: requested seed is not declared")
        for ratio in ratios:
            for count in recoveries:
                jobs.append(PaperJob(workload, ratio, count, selected_seeds))
    if not jobs:
        raise ValueError("Experiment selection produced no jobs")
    job_ids = [job.job_id for job in jobs]
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("Experiment selection produced colliding job identifiers")
    return jobs


def _build_global_mask(
    layers: Sequence[Sequence[str]],
    scores: Mapping[str, torch.Tensor],
    prune_ratio: float,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    names = [name for layer in layers for name in layer]
    sizes = [sum(scores[name].numel() for name in layer) for layer in layers]
    target = math.floor(prune_ratio * sum(sizes))
    masks = global_mask(names, scores, target)
    return masks, {
        "allocation": "exact_global",
        "layer_sizes": sizes,
        "target_pruned": target,
    }


def _score_and_mask(
    contract: MethodContract,
    model: torch.nn.Module,
    scoring_batches: Sequence[Mapping[str, torch.Tensor]],
    layers: Sequence[Sequence[str]],
    blocks: Sequence[Sequence[str]],
    delta: Mapping[str, torch.Tensor],
    prune_ratio: float,
    max_layer_ratio: float,
    task_type: str,
    model_family: str,
    device: str,
    distributed_moments: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if contract.name == "excp_style":
        magnitude = {name: value.abs() for name, value in delta.items()}
        masks, allocation = _build_global_mask(layers, magnitude, prune_ratio)
        return masks, {
            "score_kind": "residual_magnitude",
            "scoring_batches": 0,
            "allocation": allocation,
        }
    if contract.name == "inshrinkerator_style":
        named_parameters = dict(model.named_parameters())
        current_weights = {name: named_parameters[name].detach() for name in delta}
        scores, scoring = compute_block_first_order_scores(
            model,
            scoring_batches,
            layers,
            current_weights,
            device,
            task_type=task_type,
            model_family=model_family,
            block_parameter_names=blocks,
        )
        masks, allocation = _build_global_mask(layers, scores, prune_ratio)
        return masks, {
            "scoring": scoring,
            "allocation": allocation,
            "score_probe": "current_full_weight",
            "application_scope": "matched_checkpoint_residual",
        }
    if contract.name == "dacp":
        scores, scoring = compute_block_taylor_scores(
            model,
            scoring_batches,
            layers,
            delta,
            device,
            return_components=False,
            task_type=task_type,
            model_family=model_family,
            block_parameter_names=blocks,
        )
        masks, allocation = build_weibull_mask(
            layers,
            scores,
            prune_ratio,
            max_layer_ratio=max_layer_ratio,
            distributed_moments=distributed_moments,
        )
        return masks, {
            "scoring": scoring,
            "allocation": allocation,
        }
    raise ValueError(f"Method {contract.name} does not define a compression mask")


def run_method_trajectory(
    job: PaperJob,
    contract: MethodContract,
    seed: int,
    plan: RepeatedSeedBatchPlan,
    model_factory: Callable[[], torch.nn.Module],
    initial_model_state: Mapping[str, torch.Tensor],
    initial_optimizer_state: Mapping,
    eval_batches: Sequence[Mapping[str, torch.Tensor]],
    device: str,
    *,
    distributed_moments: bool = False,
) -> dict[str, object]:
    """Run one method on one paired seed through all K recovery cycles."""
    workload = job.workload
    model = None
    optimizer = None
    scheduler = None
    try:
        model = model_factory()
        model.load_state_dict(initial_model_state, strict=True)
        model.to(device)
        optimizer = build_optimizer(
            model,
            initial_optimizer_state,
            learning_rate=workload.learning_rate,
        )
        scheduler_config = recovery_protocol(workload)["scheduler"]
        if not isinstance(scheduler_config, Mapping):
            raise RuntimeError("Invalid recovery scheduler protocol")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_config["t_max"]),
            eta_min=float(scheduler_config["eta_min"]),
        )
        reconstructed_state = initial_model_state
        cycles = []
        final_training = None
        started = time.perf_counter()

        for segment_index, training_batches in enumerate(plan.training_segments):
            training = train_segment(
                model,
                optimizer,
                scheduler,
                training_batches,
                seed + segment_index * 1_000_000,
                device,
                task_type=workload.task_type,
            )
            if segment_index == job.recovery_count:
                final_training = training
                break

            current_state = clone_model_state_to_cpu(model)
            before = evaluate_task(model, eval_batches, workload.task_type, device)
            cycle: dict[str, object] = {
                "cycle": segment_index + 1,
                "training": training,
                "before_recovery": before,
            }
            if contract.name == "no_compression":
                reconstructed_state = current_state
                cycle["after_recovery"] = before
                cycle["compression"] = None
            else:
                layers = eligible_layers(model, workload.model_family)
                blocks = transformer_layers(model, workload.model_family)
                eligible_names = [name for layer in layers for name in layer]
                delta = {
                    name: current_state[name].detach().float()
                    - reconstructed_state[name].detach().float()
                    for name in eligible_names
                }
                masks, compression = _score_and_mask(
                    contract,
                    model,
                    plan.scoring_batches[segment_index],
                    layers,
                    blocks,
                    delta,
                    job.prune_ratio,
                    workload.max_layer_ratio,
                    workload.task_type,
                    workload.model_family,
                    device,
                    distributed_moments,
                )
                restore_with_mask(model, current_state, reconstructed_state, masks, device)
                reconstructed_state = clone_model_state_to_cpu(model)
                mask_stats = mask_metrics(
                    masks,
                    {name: values.abs() for name, values in delta.items()},
                )
                mask_stats["residual_magnitude_cost"] = mask_stats.pop("proxy_cost")
                compression["mask"] = {
                    **mask_stats,
                    "eligible_parameters": sum(mask.numel() for mask in masks.values()),
                    "layer_rates": layer_rates(layers, masks),
                }
                cycle["compression"] = compression
                cycle["after_recovery"] = evaluate_task(
                    model, eval_batches, workload.task_type, device
                )
                del delta, masks
            cycles.append(cycle)
            del current_state

        final = evaluate_task(model, eval_batches, workload.task_type, device)
        return {
            "method": contract.name,
            "owner": contract.owner,
            "fidelity": contract.fidelity,
            "internal_method": contract.internal_method,
            "seed": seed,
            "prune_ratio": job.prune_ratio,
            "recovery_count": job.recovery_count,
            "cycles": cycles,
            "final_training": final_training,
            "final": final,
            "wall_seconds": time.perf_counter() - started,
        }
    finally:
        scheduler = None
        optimizer = None
        model = None
        gc.collect()
        empty_device_cache(device)


def aggregate_job_results(
    records: Sequence[Mapping[str, object]],
    metric: str,
) -> dict[str, object]:
    """Aggregate final metrics without assuming lower- or higher-is-better."""
    metric_directions = {
        "loss": "lower",
        "perplexity": "lower",
        "accuracy": "higher",
        "pearson": "higher",
    }
    if metric not in metric_directions:
        raise ValueError(f"Unsupported aggregate metric: {metric}")
    if not records:
        raise ValueError("Cannot aggregate an empty result set")
    method_summaries: dict[str, object] = {}
    paired_comparisons: dict[str, object] = {}
    methods = list(dict.fromkeys(str(record["method"]) for record in records))
    by_method_seed: dict[str, dict[int, float]] = {}
    for method in methods:
        method_records = [record for record in records if record["method"] == method]
        seeded_values = {}
        for record in method_records:
            seed = record.get("seed")
            final = record.get("final")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(f"Invalid aggregate seed for method {method}: {seed!r}")
            if not isinstance(final, Mapping) or metric not in final:
                raise ValueError(f"Missing final metric {metric} for method {method}")
            raw_value = final[metric]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"Invalid final metric {metric} for method {method}")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"Non-finite final metric {metric} for method {method}")
            if seed in seeded_values:
                raise ValueError(f"Duplicate seed records for method {method}")
            seeded_values[seed] = value
        values = [seeded_values[seed] for seed in sorted(seeded_values)]
        if not values:
            raise ValueError(f"No values for method {method} and metric {metric}")
        by_method_seed[method] = seeded_values
        method_summaries[method] = {
            "metric": metric,
            "seeds": sorted(seeded_values),
            "values": values,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
        }
    seed_sets = {tuple(sorted(values)) for values in by_method_seed.values()}
    if len(seed_sets) != 1:
        raise ValueError("Matched methods must have identical seed sets")
    direction = metric_directions[metric]
    if "dacp" in by_method_seed:
        for baseline in ("excp_style", "inshrinkerator_style"):
            if baseline not in by_method_seed:
                continue
            dacp_values = by_method_seed["dacp"]
            baseline_values = by_method_seed[baseline]
            if set(dacp_values) != set(baseline_values):
                raise ValueError(f"Paired comparison has mismatched seeds: {baseline}")
            seeds = sorted(dacp_values)
            deltas = [dacp_values[seed] - baseline_values[seed] for seed in seeds]
            wins = sum(delta < 0 if direction == "lower" else delta > 0 for delta in deltas)
            paired_comparisons[f"dacp_vs_{baseline}"] = {
                "metric": metric,
                "direction": direction,
                "seeds": seeds,
                "dacp_minus_baseline": deltas,
                "mean_delta": statistics.mean(deltas),
                "wins": wins,
                "pairs": len(deltas),
            }
    return {"methods": method_summaries, "paired_comparisons": paired_comparisons}


__all__ = [
    "PaperJob",
    "aggregate_job_results",
    "plan_jobs",
    "run_method_trajectory",
]
