"""Pure full-weight pruning assembly and exact-budget evidence helpers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from baselines.inshrinkerator.per_type_allocation import PerTypeAllocation
from dacp.pruning.masks import exact_keep_mask
from dacp.pruning.param_schema import infer_layer_type
from experiments.lib.residual_budget import bounded_largest_remainder_counts
from experiments.lib.residual_methods import build_weibull_mask

DACP_FULLWEIGHT_METHOD_ID = "dacp_fullweight"
INSHRINKERATOR_STYLE_FULLWEIGHT_METHOD_ID = "inshrinkerator_style_fullweight"
FULLWEIGHT_METHOD_IDS = (
    DACP_FULLWEIGHT_METHOD_ID,
    INSHRINKERATOR_STYLE_FULLWEIGHT_METHOD_ID,
)


@dataclass(frozen=True)
class FullWeightMethodContract:
    method_id: str
    fidelity: str
    score_kind: str
    allocation_kind: str
    description: str

    def to_result_dict(self) -> dict[str, str]:
        return {
            "method_id": self.method_id,
            "fidelity": self.fidelity,
            "score_kind": self.score_kind,
            "allocation_kind": self.allocation_kind,
            "description": self.description,
        }


FULLWEIGHT_METHOD_CONTRACTS = {
    DACP_FULLWEIGHT_METHOD_ID: FullWeightMethodContract(
        method_id=DACP_FULLWEIGHT_METHOD_ID,
        fidelity="native",
        score_kind="taylor_hvp",
        allocation_kind="weibull_moment",
        description=(
            "Full-weight Taylor HVP scoring with Weibull moment allocation and direct "
            "zeroing; no residual reconstruction, quantization, or encoding."
        ),
    ),
    INSHRINKERATOR_STYLE_FULLWEIGHT_METHOD_ID: FullWeightMethodContract(
        method_id=INSHRINKERATOR_STYLE_FULLWEIGHT_METHOD_ID,
        fidelity="style",
        score_kind="searched_magnitude_or_first_order",
        allocation_kind="per_type",
        description=(
            "Inshrinkerator-style searched full-weight magnitude/first-order pruning "
            "with direct zeroing; excludes protection, quantization, and encoding."
        ),
    ),
}


@dataclass(frozen=True)
class FullWeightMaskAssembly:
    """One exact-budget full-weight mask and its auditable allocation evidence."""

    method_id: str
    fidelity: str
    score_kind: str
    allocation_kind: str
    target_pruned: int
    actual_pruned: int
    layer_counts: tuple[int, ...]
    layer_rates: tuple[float, ...]
    mask_fingerprint: str
    masks: Mapping[str, torch.Tensor]
    evidence: Mapping[str, object]

    def to_result_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "fidelity": self.fidelity,
            "score_kind": self.score_kind,
            "allocation_kind": self.allocation_kind,
            "target_pruned": self.target_pruned,
            "actual_pruned": self.actual_pruned,
            "actual_global_ratio": self.actual_pruned / sum(self.evidence["layer_sizes"]),
            "layer_counts": list(self.layer_counts),
            "layer_rates": list(self.layer_rates),
            "mask_fingerprint": self.mask_fingerprint,
            **dict(self.evidence),
        }


def _validate_ratio(prune_ratio: object) -> float:
    if isinstance(prune_ratio, bool) or not isinstance(prune_ratio, (int, float)):
        raise TypeError("prune_ratio must be a real number")
    ratio = float(prune_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError("prune_ratio must be finite and in (0, 1)")
    return ratio


def _eligible_names(layers: Sequence[Sequence[str]]) -> list[str]:
    names = [name for layer in layers for name in layer]
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Eligible layers must contain non-empty parameter names")
    if len(names) != len(set(names)):
        raise ValueError("Eligible full-weight parameter names must be unique")
    if any(not layer for layer in layers):
        raise ValueError("Eligible layers must be non-empty")
    return names


def _validate_parameter_scores(
    parameters: Mapping[str, torch.Tensor],
    scores: Mapping[str, torch.Tensor],
    names: Sequence[str],
    *,
    label: str,
) -> None:
    if set(parameters) != set(names):
        raise ValueError(f"{label} parameters do not match the eligible parameter set")
    if set(scores) != set(names):
        raise ValueError(f"{label} scores do not match the eligible parameter set")
    for name in names:
        parameter = parameters[name]
        score = scores[name]
        if not isinstance(parameter, torch.Tensor) or not parameter.is_floating_point():
            raise TypeError(f"Full-weight parameter {name!r} must be a real tensor")
        if not isinstance(score, torch.Tensor) or not score.is_floating_point():
            raise TypeError(f"{label} score {name!r} must be a real tensor")
        if parameter.shape != score.shape:
            raise ValueError(f"{label} score shape for {name!r} does not match its parameter")
        if not torch.isfinite(parameter).all().item() or not torch.isfinite(score).all().item():
            raise ValueError(f"Full-weight parameters and {label} scores must be finite")


def target_pruned_count(parameters: Mapping[str, torch.Tensor], prune_ratio: float) -> int:
    """Return the formal floor budget over the eligible full-weight parameters."""
    ratio = _validate_ratio(prune_ratio)
    eligible = sum(value.numel() for value in parameters.values())
    if eligible < 1:
        raise ValueError("Eligible full-weight parameters must contain elements")
    return math.floor(ratio * eligible)


def _mask_fingerprint(masks: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(masks):
        mask = masks[name].detach().cpu().contiguous()
        if mask.dtype != torch.bool:
            raise TypeError(f"Mask for {name!r} must have bool dtype")
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(mask.shape)).encode("ascii"))
        digest.update(mask.numpy().tobytes())
    return digest.hexdigest()


def _layer_evidence(
    layers: Sequence[Sequence[str]],
    masks: Mapping[str, torch.Tensor],
    scores: Mapping[str, torch.Tensor],
) -> tuple[tuple[int, ...], tuple[float, ...], int]:
    counts = []
    sizes = []
    for layer in layers:
        size = sum(scores[name].numel() for name in layer)
        count = sum(int((~masks[name]).sum().item()) for name in layer)
        sizes.append(size)
        counts.append(count)
    actual = sum(counts)
    rates = tuple(count / size for count, size in zip(counts, sizes))
    return tuple(counts), rates, actual


def _make_assembly(
    *,
    method_id: str,
    fidelity: str,
    score_kind: str,
    allocation_kind: str,
    target: int,
    layers: Sequence[Sequence[str]],
    scores: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    evidence: Mapping[str, object],
) -> FullWeightMaskAssembly:
    names = _eligible_names(layers)
    if set(masks) != set(names):
        raise ValueError("Full-weight masks do not match the eligible parameter set")
    for name in names:
        if masks[name].shape != scores[name].shape or masks[name].dtype != torch.bool:
            raise ValueError(f"Mask for {name!r} does not match its score tensor")
    layer_counts, layer_rates, actual = _layer_evidence(layers, masks, scores)
    if actual != target:
        raise ValueError(f"Full-weight mask pruned {actual}; expected exact target {target}")
    return FullWeightMaskAssembly(
        method_id=method_id,
        fidelity=fidelity,
        score_kind=score_kind,
        allocation_kind=allocation_kind,
        target_pruned=target,
        actual_pruned=actual,
        layer_counts=layer_counts,
        layer_rates=layer_rates,
        mask_fingerprint=_mask_fingerprint(masks),
        masks={name: mask.detach().cpu().clone() for name, mask in masks.items()},
        evidence={
            "layer_sizes": [sum(scores[name].numel() for name in layer) for layer in layers],
            "eligible_parameters": sum(score.numel() for score in scores.values()),
            **dict(evidence),
        },
    )


def build_dacp_fullweight_mask(
    parameters: Mapping[str, torch.Tensor],
    layers: Sequence[Sequence[str]],
    taylor_scores: Mapping[str, torch.Tensor],
    prune_ratio: float,
    *,
    max_layer_ratio: float = 1.0,
) -> FullWeightMaskAssembly:
    """Assemble DACP's full-weight Taylor-HVP plus Weibull mask.

    ``taylor_scores`` must already have been computed with the current full
    parameter values as the HVP probe. This function only assembles the mask.
    """
    names = _eligible_names(layers)
    _validate_parameter_scores(parameters, taylor_scores, names, label="Taylor HVP")
    target = target_pruned_count(parameters, prune_ratio)
    if isinstance(max_layer_ratio, bool) or not isinstance(max_layer_ratio, (int, float)):
        raise TypeError("max_layer_ratio must be a real number")
    if not math.isfinite(float(max_layer_ratio)) or not 0.0 < float(max_layer_ratio) <= 1.0:
        raise ValueError("max_layer_ratio must be finite and in (0, 1]")
    masks, allocation = build_weibull_mask(
        layers,
        taylor_scores,
        prune_ratio,
        max_layer_ratio=float(max_layer_ratio),
    )
    if allocation.get("target_pruned") != target:
        raise ValueError("DACP Weibull allocation target does not match the full-weight budget")
    return _make_assembly(
        method_id=DACP_FULLWEIGHT_METHOD_ID,
        fidelity="native",
        score_kind="taylor_hvp",
        allocation_kind="weibull_moment",
        target=target,
        layers=layers,
        scores=taylor_scores,
        masks=masks,
        evidence={
            "probe": "current_full_weight_theta",
            "weibull_fits": allocation["weibull_fits"],
            "weibull_layer_counts": allocation["weibull_layer_counts"],
            "weibull": allocation["weibull"],
            "allocation_provenance": {
                "allocator": "build_weibull_mask",
                "direct_zeroing": True,
                "residual_reconstruction": False,
            },
        },
    )


def _validated_search_selection(
    search_result: Mapping[str, object]
) -> tuple[str, dict[str, float]]:
    best_metric = search_result.get("best_metric")
    if best_metric not in {"magnitude", "sensitivity"}:
        raise ValueError("Search result best_metric must be 'magnitude' or 'sensitivity'")
    raw_ratios = search_result.get("per_type_ratios")
    if not isinstance(raw_ratios, Mapping) or not raw_ratios:
        raise ValueError("Search result per_type_ratios must be a non-empty object")
    ratios: dict[str, float] = {}
    for layer_type, raw_ratio in raw_ratios.items():
        if not isinstance(layer_type, str) or not layer_type:
            raise ValueError("Search result per_type_ratios keys must be non-empty strings")
        if isinstance(raw_ratio, bool) or not isinstance(raw_ratio, (int, float)):
            raise ValueError(f"Search result ratio for {layer_type!r} must be numeric")
        ratio = float(raw_ratio)
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError(f"Search result ratio for {layer_type!r} must be finite in [0, 1]")
        ratios[layer_type] = ratio
    actual_global_ratio = search_result.get("actual_global_ratio")
    if (
        isinstance(actual_global_ratio, bool)
        or not isinstance(actual_global_ratio, (int, float))
        or not math.isfinite(float(actual_global_ratio))
        or not 0.0 <= float(actual_global_ratio) <= 1.0
    ):
        raise ValueError("Search result actual_global_ratio must be finite in [0, 1]")
    return str(best_metric), ratios


def build_inshrinkerator_style_fullweight_mask(
    parameters: Mapping[str, torch.Tensor],
    layers: Sequence[Sequence[str]],
    score: Mapping[str, torch.Tensor],
    search_result: Mapping[str, object],
    prune_ratio: float,
    *,
    model_family: str = "gpt2",
) -> FullWeightMaskAssembly:
    """Project searched per-type ratios onto an exact full-weight pruning budget."""
    names = _eligible_names(layers)
    _validate_parameter_scores(parameters, score, names, label="Inshrinkerator-style")
    best_metric, per_type_ratios = _validated_search_selection(search_result)
    target = target_pruned_count(parameters, prune_ratio)
    allocator = PerTypeAllocation(
        per_type_ratios=per_type_ratios,
        source_global_ratio=float(search_result["actual_global_ratio"]),
        model_family=model_family,
    )
    proposed_ratios = allocator.allocate(dict(score), prune_ratio)
    real_counts = [proposed_ratios[name] * score[name].numel() for name in names]
    capacities = [score[name].numel() for name in names]
    counts = bounded_largest_remainder_counts(
        real_counts,
        target,
        [0] * len(names),
        capacities,
    )
    masks = {name: exact_keep_mask(score[name], count) for name, count in zip(names, counts)}
    type_counts: dict[str, int] = {}
    type_sizes: dict[str, int] = {}
    for name, count in zip(names, counts):
        layer_type = infer_layer_type(name, score[name], model_family)
        if layer_type == "skip":
            raise ValueError(f"Eligible full-weight parameter {name!r} maps to skip")
        type_counts[layer_type] = type_counts.get(layer_type, 0) + count
        type_sizes[layer_type] = type_sizes.get(layer_type, 0) + score[name].numel()
    type_rates = {
        layer_type: type_counts[layer_type] / type_sizes[layer_type]
        for layer_type in sorted(type_counts)
    }
    return _make_assembly(
        method_id=INSHRINKERATOR_STYLE_FULLWEIGHT_METHOD_ID,
        fidelity="style",
        score_kind="magnitude" if best_metric == "magnitude" else "first_order",
        allocation_kind="per_type",
        target=target,
        layers=layers,
        scores=score,
        masks=masks,
        evidence={
            "search_best_metric": best_metric,
            "search_per_type_ratios": per_type_ratios,
            "per_parameter_proposed_ratios": proposed_ratios,
            "per_type_counts": type_counts,
            "per_type_rates": type_rates,
            "allocation_provenance": {
                "allocator": "PerTypeAllocation",
                "projection": "bounded_largest_remainder_counts",
                "direct_zeroing": True,
                "residual_reconstruction": False,
                "protection_applied": False,
            },
        },
    )


def assemble_fullweight_comparison(
    parameters: Mapping[str, torch.Tensor],
    layers: Sequence[Sequence[str]],
    score_families: Mapping[str, Mapping[str, torch.Tensor]],
    search_result: Mapping[str, object],
    prune_ratio: float,
    *,
    model_family: str = "gpt2",
    max_layer_ratio: float = 1.0,
) -> tuple[FullWeightMaskAssembly, FullWeightMaskAssembly]:
    """Build both methods from one immutable full-weight score/evidence bundle."""
    best_metric, _ = _validated_search_selection(search_result)
    required = {"magnitude", "first_order", "taylor_hvp"}
    missing = sorted(required.difference(score_families))
    if missing:
        raise ValueError(f"Full-weight score families are missing: {missing}")
    selected_key = "magnitude" if best_metric == "magnitude" else "first_order"
    dacp = build_dacp_fullweight_mask(
        parameters,
        layers,
        score_families["taylor_hvp"],
        prune_ratio,
        max_layer_ratio=max_layer_ratio,
    )
    inshrinkerator = build_inshrinkerator_style_fullweight_mask(
        parameters,
        layers,
        score_families[selected_key],
        search_result,
        prune_ratio,
        model_family=model_family,
    )
    return dacp, inshrinkerator


def apply_zero_masks_to_state(
    state: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return a cloned state with selected full weights directly set to zero."""
    result = {name: value.detach().clone() for name, value in state.items()}
    for name, mask in masks.items():
        if name not in result:
            raise ValueError(f"Mask refers to missing state parameter {name!r}")
        if mask.dtype != torch.bool or mask.shape != result[name].shape:
            raise ValueError(f"Mask for {name!r} does not match the state tensor")
        result[name] = result[name] * mask.to(device=result[name].device, dtype=result[name].dtype)
    return result


def validate_search_result_provenance(
    payload: Mapping[str, object],
    *,
    model: str,
    dataset: str,
    seed: int,
    checkpoint: Path,
    checkpoint_sha256: str,
    source_digest: str,
    scoring_batch_sha256: str,
    evaluation_batch_sha256: str,
    scoring_batch_count: int,
    evaluation_batch_count: int,
    protect_fraction: float,
) -> None:
    """Fail closed unless a search result is tied to the formal comparison inputs."""
    if not isinstance(payload, Mapping):
        raise ValueError("Search JSON must contain an object")
    required = {
        "schema_version",
        "status",
        "model",
        "dataset",
        "seed",
        "checkpoint",
        "checkpoint_sha256",
        "source_digest",
        "scoring_batch_sha256",
        "evaluation_batch_sha256",
        "scoring_batch_count",
        "evaluation_batch_count",
        "best_metric",
        "per_type_ratios",
        "actual_global_ratio",
        "quality_drop",
        "protect_fraction",
    }
    missing = sorted(key for key in required if key not in payload)
    if missing:
        raise ValueError(f"Search JSON is missing provenance fields: {missing}")
    if (
        isinstance(payload["schema_version"], bool)
        or payload["schema_version"] != 1
        or payload["status"] != "complete"
    ):
        raise ValueError("Search JSON must have schema_version=1 and status=complete")
    if payload["model"] != model or payload["dataset"] != dataset or payload["seed"] != seed:
        raise ValueError("Search JSON model, dataset, or seed does not match the comparison")
    if Path(str(payload["checkpoint"])).resolve() != checkpoint.resolve():
        raise ValueError("Search JSON checkpoint path does not match the comparison")
    if payload["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Search JSON checkpoint digest does not match the comparison")
    if payload["source_digest"] != source_digest:
        raise ValueError("Search JSON source digest does not match the comparison")
    if payload["scoring_batch_sha256"] != scoring_batch_sha256:
        raise ValueError("Search JSON scoring batch digest does not match the comparison")
    if payload["evaluation_batch_sha256"] != evaluation_batch_sha256:
        raise ValueError("Search JSON evaluation batch digest does not match the comparison")
    if payload["scoring_batch_count"] != scoring_batch_count:
        raise ValueError("Search JSON scoring batch count does not match the comparison")
    if payload["evaluation_batch_count"] != evaluation_batch_count:
        raise ValueError("Search JSON evaluation batch count does not match the comparison")
    if payload["protect_fraction"] != protect_fraction:
        raise ValueError("Search JSON protect_fraction does not match the comparison")
    _validated_search_selection(payload)
    quality_drop = payload["quality_drop"]
    if (
        isinstance(quality_drop, bool)
        or not isinstance(quality_drop, (int, float))
        or not math.isfinite(float(quality_drop))
    ):
        raise ValueError("Search JSON quality_drop must be finite")


__all__ = [
    "DACP_FULLWEIGHT_METHOD_ID",
    "FULLWEIGHT_METHOD_CONTRACTS",
    "FULLWEIGHT_METHOD_IDS",
    "FullWeightMaskAssembly",
    "FullWeightMethodContract",
    "INSHRINKERATOR_STYLE_FULLWEIGHT_METHOD_ID",
    "apply_zero_masks_to_state",
    "assemble_fullweight_comparison",
    "build_dacp_fullweight_mask",
    "build_inshrinkerator_style_fullweight_mask",
    "target_pruned_count",
    "validate_search_result_provenance",
]
