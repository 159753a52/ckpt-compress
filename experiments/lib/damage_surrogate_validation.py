"""Validate mask-level residual score costs against immediate loss changes."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
import torch.nn as nn
from scipy.stats import kendalltau, spearmanr

from experiments.lib.residual_runtime import task_loss
from experiments.lib.residual_scoring import model_checksum

TensorDict = Mapping[str, torch.Tensor]
LossFunction = Callable[[nn.Module, Mapping[str, torch.Tensor], str], torch.Tensor]


@dataclass(frozen=True)
class MaskCandidate:
    """One named candidate mask and its provenance label."""

    id: str
    source: str
    mask: TensorDict
    score_family: str | None = None


@dataclass
class _ModelSnapshot:
    state: dict[str, torch.Tensor]
    modes: dict[str, bool]
    requires_grad: dict[str, bool]
    grad_refs: dict[str, torch.Tensor | None]
    grad_values: dict[str, torch.Tensor | None]
    versions: dict[str, int]
    checksum: tuple[float, float]


def _validate_layers(
    model: nn.Module,
    eligible_layers: Sequence[Sequence[str]],
) -> list[list[str]]:
    if not isinstance(eligible_layers, Sequence) or not eligible_layers:
        raise ValueError("eligible_layers must contain at least one layer")
    named_parameters = dict(model.named_parameters())
    normalized: list[list[str]] = []
    names: list[str] = []
    for index, layer in enumerate(eligible_layers):
        if isinstance(layer, (str, bytes)) or not isinstance(layer, Sequence) or not layer:
            raise ValueError(f"Eligible layer {index} must be a non-empty sequence")
        if not all(isinstance(name, str) and name for name in layer):
            raise TypeError(f"Eligible layer {index} names must be non-empty strings")
        normalized_layer = list(layer)
        if len(set(normalized_layer)) != len(normalized_layer):
            raise ValueError(f"Eligible layer {index} contains duplicate parameter names")
        missing = sorted(set(normalized_layer).difference(named_parameters))
        if missing:
            raise ValueError(f"Eligible parameters are missing from model: {missing}")
        normalized.append(normalized_layer)
        names.extend(normalized_layer)
    if len(names) != len(set(names)):
        raise ValueError("Eligible parameter names must be unique across layers")
    return normalized


def _validate_finite_real_tensor(value: object, label: str, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} for {name!r} must be a tensor")
    if (
        value.layout != torch.strided
        or value.is_quantized
        or bool(getattr(value, "is_nested", False))
    ):
        raise TypeError(f"{label} for {name!r} must be a dense tensor")
    if not value.is_floating_point() or value.is_complex():
        raise TypeError(f"{label} for {name!r} must be real floating point")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{label} for {name!r} must contain only finite values")
    return value


def _normalize_score_families(
    score_families: Mapping[str, TensorDict],
    names: Sequence[str],
) -> dict[str, dict[str, torch.Tensor]]:
    if not isinstance(score_families, Mapping) or not score_families:
        raise ValueError("score_families must contain at least one family")
    expected = set(names)
    normalized: dict[str, dict[str, torch.Tensor]] = {}
    for family, scores in score_families.items():
        if not isinstance(family, str) or not family:
            raise TypeError("Score family names must be non-empty strings")
        if not isinstance(scores, Mapping):
            raise TypeError(f"Scores for family {family!r} must be a mapping")
        missing = sorted(expected.difference(scores))
        extra = sorted(set(scores).difference(expected))
        if missing or extra:
            raise ValueError(
                f"Scores for family {family!r} must match eligible parameters: "
                f"missing={missing}, extra={extra}"
            )
        family_scores: dict[str, torch.Tensor] = {}
        for name in names:
            score = _validate_finite_real_tensor(scores[name], "Score", name)
            if score.numel() == 0:
                raise ValueError(f"Score for {name!r} must contain at least one element")
            if (score < 0).any().item():
                raise ValueError(f"Score for {name!r} must be non-negative")
            family_scores[name] = score.detach().clone()
        normalized[family] = family_scores
    return normalized


def _normalize_candidates(
    candidate_masks: Mapping[str, object] | Sequence[object],
) -> list[MaskCandidate]:
    if isinstance(candidate_masks, Mapping):
        if candidate_masks and all(
            isinstance(value, torch.Tensor) for value in candidate_masks.values()
        ):
            raw_candidates: list[object] = [candidate_masks]
        else:
            raw_candidates = [(key, value) for key, value in candidate_masks.items()]
    elif isinstance(candidate_masks, Sequence) and not isinstance(candidate_masks, (str, bytes)):
        raw_candidates = list(candidate_masks)
    else:
        raise TypeError("candidate_masks must be a mapping or a sequence")
    if not raw_candidates:
        raise ValueError("candidate_masks must contain at least one mask")

    candidates: list[MaskCandidate] = []
    for index, raw in enumerate(raw_candidates):
        default_id = f"mask_{index:03d}"
        if isinstance(raw, MaskCandidate):
            candidate = raw
        elif isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], str):
            key, value = raw
            if isinstance(value, MaskCandidate):
                candidate = MaskCandidate(
                    id=value.id or key,
                    source=value.source,
                    mask=value.mask,
                    score_family=value.score_family,
                )
            elif isinstance(value, Mapping) and "mask" in value:
                candidate = MaskCandidate(
                    id=str(value.get("id", key)),
                    source=str(value.get("source", key)),
                    mask=value["mask"],
                    score_family=value.get("score_family"),
                )
            else:
                candidate = MaskCandidate(id=key, source=key, mask=value)
        elif isinstance(raw, Mapping) and "mask" in raw:
            candidate = MaskCandidate(
                id=str(raw.get("id", default_id)),
                source=str(raw.get("source", raw.get("id", default_id))),
                mask=raw["mask"],
                score_family=raw.get("score_family"),
            )
        else:
            candidate = MaskCandidate(id=default_id, source=default_id, mask=raw)  # type: ignore[arg-type]
        if not isinstance(candidate.id, str) or not candidate.id:
            raise ValueError("Candidate mask ids must be non-empty strings")
        if not isinstance(candidate.source, str) or not candidate.source:
            raise ValueError(f"Candidate {candidate.id!r} source must be non-empty")
        if candidate.score_family is not None and (
            not isinstance(candidate.score_family, str) or not candidate.score_family
        ):
            raise ValueError(f"Candidate {candidate.id!r} score_family must be non-empty")
        candidates.append(candidate)
    ids = [candidate.id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate mask ids must be unique")
    return candidates


def _validate_mask(
    mask: TensorDict,
    names: Sequence[str],
    candidate_id: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(mask, Mapping):
        raise TypeError(f"Candidate {candidate_id!r} mask must be a mapping")
    expected = set(names)
    missing = sorted(expected.difference(mask))
    extra = sorted(set(mask).difference(expected))
    if missing or extra:
        raise ValueError(
            f"Candidate {candidate_id!r} mask must match eligible parameters: "
            f"missing={missing}, extra={extra}"
        )
    normalized: dict[str, torch.Tensor] = {}
    for name in names:
        value = mask[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Candidate {candidate_id!r} mask for {name!r} must be a tensor")
        if value.dtype != torch.bool:
            raise TypeError(f"Candidate {candidate_id!r} mask for {name!r} must be boolean")
        if (
            value.layout != torch.strided
            or value.is_quantized
            or bool(getattr(value, "is_nested", False))
        ):
            raise TypeError(f"Candidate {candidate_id!r} mask for {name!r} must be dense")
        if value.numel() == 0:
            raise ValueError(f"Candidate {candidate_id!r} mask for {name!r} is empty")
        normalized[name] = value.detach().clone()
    return normalized


def _mask_fingerprint(mask: Mapping[str, torch.Tensor], names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        value = mask[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _snapshot_model(model: nn.Module) -> _ModelSnapshot:
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    modes = {name: module.training for name, module in model.named_modules()}
    requires_grad: dict[str, bool] = {}
    grad_refs: dict[str, torch.Tensor | None] = {}
    grad_values: dict[str, torch.Tensor | None] = {}
    versions: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        requires_grad[name] = parameter.requires_grad
        grad_refs[name] = parameter.grad
        grad_values[name] = None if parameter.grad is None else parameter.grad.detach().clone()
        versions[name] = parameter._version
    return _ModelSnapshot(
        state=state,
        modes=modes,
        requires_grad=requires_grad,
        grad_refs=grad_refs,
        grad_values=grad_values,
        versions=versions,
        checksum=model_checksum(model),
    )


def _copy_state_preserving_versions(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    current = model.state_dict()
    if set(current) != set(state):
        raise ValueError("Model state keys changed during damage validation")
    with torch.no_grad():
        for name, target in current.items():
            source = state[name]
            if target.shape != source.shape:
                raise ValueError(f"Model state shape changed for {name!r}")
            # Keep temporary probes out of autograd's version-counter history.
            target.data.copy_(source.to(device=target.device, dtype=target.dtype))


def _restore_snapshot(model: nn.Module, snapshot: _ModelSnapshot) -> None:
    _copy_state_preserving_versions(model, snapshot.state)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(snapshot.requires_grad[name])
        original_gradient = snapshot.grad_refs[name]
        original_values = snapshot.grad_values[name]
        if original_gradient is None:
            parameter.grad = None
        else:
            assert original_values is not None
            with torch.no_grad():
                original_gradient.copy_(original_values)
            parameter.grad = original_gradient
    for name, module in model.named_modules():
        module.training = snapshot.modes[name]


def _assert_snapshot(model: nn.Module, snapshot: _ModelSnapshot) -> None:
    current = model.state_dict()
    for name, expected in snapshot.state.items():
        if name not in current or not torch.equal(current[name], expected):
            raise RuntimeError(f"Damage validation changed model state for {name!r}")
    checksum = model_checksum(model)
    if checksum != snapshot.checksum:
        raise RuntimeError(
            f"Damage validation changed model checksum: before={snapshot.checksum}, after={checksum}"
        )
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    if versions != snapshot.versions:
        changed = [name for name in versions if versions[name] != snapshot.versions[name]]
        raise RuntimeError(f"Damage validation changed parameter versions: {changed[:5]}")
    modes = {name: module.training for name, module in model.named_modules()}
    if modes != snapshot.modes:
        raise RuntimeError("Damage validation did not restore module training modes")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad != snapshot.requires_grad[name]:
            raise RuntimeError(f"Damage validation did not restore requires_grad for {name!r}")
        if parameter.grad is not snapshot.grad_refs[name]:
            raise RuntimeError(f"Damage validation did not restore the original grad for {name!r}")
        expected = snapshot.grad_values[name]
        if expected is not None and not torch.equal(parameter.grad, expected):
            raise RuntimeError(f"Damage validation changed the original grad for {name!r}")


def _evaluate_loss(
    model: nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    loss_fn: LossFunction,
    device: str,
) -> float:
    if not batches:
        raise ValueError("evaluation batches must contain at least one batch")
    modes = {name: module.training for name, module in model.named_modules()}
    model.eval()
    values: list[float] = []
    try:
        with torch.no_grad():
            for index, batch in enumerate(batches):
                if not isinstance(batch, Mapping):
                    raise TypeError(f"Evaluation batch {index} must be a mapping")
                loss = loss_fn(model, batch, device)
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise TypeError("Loss function must return a scalar torch.Tensor")
                value = float(loss.detach().float().cpu().item())
                if not math.isfinite(value):
                    raise ValueError(f"Evaluation loss for batch {index} must be finite")
                values.append(value)
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
    average = sum(values) / len(values)
    if not math.isfinite(average):
        raise ValueError("Evaluation loss must be finite")
    return average


def _correlation(predicted: Sequence[float], actual: Sequence[float]) -> dict[str, float | int]:
    if len(predicted) < 2:
        raise ValueError("At least two masks are required for rank correlation")
    if len(set(predicted)) < 2 or len(set(actual)) < 2:
        raise ValueError("Rank correlation requires variation in both costs and losses")
    spearman = float(spearmanr(predicted, actual).statistic)
    kendall = float(kendalltau(predicted, actual).statistic)
    if not math.isfinite(spearman) or not math.isfinite(kendall):
        raise ValueError("Rank correlations must be finite")
    return {"spearman": spearman, "kendall": kendall, "mask_count": len(predicted)}


def validate_damage_surrogate(
    model: nn.Module,
    residual_delta: TensorDict,
    eligible_layers: Sequence[Sequence[str]],
    evaluation_batches: Sequence[Mapping[str, torch.Tensor]],
    candidate_masks: Mapping[str, object] | Sequence[object],
    score_families: Mapping[str, TensorDict],
    *,
    previous_reconstructed: TensorDict | None = None,
    device: str | None = None,
    task_type: str = "lm",
    loss_fn: LossFunction | None = None,
    default_score_family: str | None = None,
) -> dict[str, object]:
    """Measure score-cost and immediate-loss rank agreement over equal-budget masks."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    layers = _validate_layers(model, eligible_layers)
    names = [name for layer in layers for name in layer]
    named_parameters = dict(model.named_parameters())
    if not isinstance(residual_delta, Mapping):
        raise TypeError("residual_delta must be a mapping")
    delta: dict[str, torch.Tensor] = {}
    for name in names:
        if name not in residual_delta:
            raise ValueError(f"Residual delta is missing eligible parameter {name!r}")
        value = _validate_finite_real_tensor(residual_delta[name], "Residual delta", name)
        if value.shape != named_parameters[name].shape:
            raise ValueError(
                f"Residual delta shape for {name!r} must match the parameter: "
                f"{tuple(value.shape)} != {tuple(named_parameters[name].shape)}"
            )
        delta[name] = value.detach().float().cpu().clone()

    previous: dict[str, torch.Tensor] = {}
    current_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for name in names:
        if previous_reconstructed is None:
            previous[name] = named_parameters[name].detach().float().cpu() - delta[name]
        else:
            if name not in previous_reconstructed:
                raise ValueError(f"previous_reconstructed is missing eligible parameter {name!r}")
            value = _validate_finite_real_tensor(
                previous_reconstructed[name],
                "Previous reconstructed parameter",
                name,
            )
            if value.shape != named_parameters[name].shape:
                raise ValueError(
                    f"Previous reconstructed shape for {name!r} must match the parameter"
                )
            previous[name] = value.detach().float().cpu().clone()

    scores = _normalize_score_families(score_families, names)
    for family, family_scores in scores.items():
        for name in names:
            if family_scores[name].shape != named_parameters[name].shape:
                raise ValueError(
                    f"Score shape for family {family!r}, parameter {name!r} must match the parameter"
                )
    candidates = _normalize_candidates(candidate_masks)
    validated_candidates: list[tuple[MaskCandidate, dict[str, torch.Tensor], int, str]] = []
    fingerprints: set[str] = set()
    expected_pruned: int | None = None
    for candidate in candidates:
        mask = _validate_mask(candidate.mask, names, candidate.id)
        for name in names:
            if mask[name].shape != named_parameters[name].shape:
                raise ValueError(
                    f"Candidate {candidate.id!r} mask shape for {name!r} must match the parameter"
                )
        fingerprint = _mask_fingerprint(mask, names)
        if fingerprint in fingerprints:
            raise ValueError(f"Duplicate candidate mask: {candidate.id!r}")
        fingerprints.add(fingerprint)
        pruned = sum(int((~mask[name]).sum().item()) for name in names)
        if expected_pruned is None:
            expected_pruned = pruned
        elif pruned != expected_pruned:
            raise ValueError(
                f"Candidate masks must have one prune count: {expected_pruned} != {pruned}"
            )
        validated_candidates.append((candidate, mask, pruned, fingerprint))
    assert expected_pruned is not None

    if default_score_family is None:
        default_score_family = "taylor" if "taylor" in scores else next(iter(scores))
    if default_score_family not in scores:
        raise ValueError(f"Unknown default score family: {default_score_family!r}")
    if loss_fn is None:
        if not isinstance(task_type, str) or not task_type:
            raise ValueError("task_type must be a non-empty string")

        def loss_fn(current_model, batch, current_device):
            return task_loss(current_model, batch, task_type, current_device)

    if device is None:
        try:
            device = str(next(model.parameters()).device)
        except StopIteration as exc:
            raise ValueError("model must contain at least one parameter") from exc
    snapshot = _snapshot_model(model)
    base_loss = _evaluate_loss(model, evaluation_batches, loss_fn, device)
    _assert_snapshot(model, snapshot)

    records: list[dict[str, object]] = []
    try:
        for candidate, mask, pruned, fingerprint in validated_candidates:
            restored_state = {name: value.clone() for name, value in current_state.items()}
            for name in names:
                keep = mask[name].detach().cpu()
                restored = previous[name] + keep.to(dtype=delta[name].dtype) * delta[name]
                restored_state[name] = restored.to(dtype=current_state[name].dtype)
            _copy_state_preserving_versions(model, restored_state)
            masked_loss = _evaluate_loss(model, evaluation_batches, loss_fn, device)
            actual_delta_loss = masked_loss - base_loss
            if not math.isfinite(actual_delta_loss):
                raise ValueError(f"Actual loss delta for {candidate.id!r} must be finite")
            predicted_costs = {
                family: sum(
                    scores[family][name][~mask[name]].detach().float().sum().item()
                    for name in names
                )
                for family in scores
            }
            if any(not math.isfinite(value) or value < 0 for value in predicted_costs.values()):
                raise ValueError(
                    f"Predicted cost for {candidate.id!r} must be finite and non-negative"
                )
            selected_family = candidate.score_family
            if selected_family is None or selected_family not in predicted_costs:
                selected_family = (
                    candidate.source
                    if candidate.source in predicted_costs
                    else default_score_family
                )
            records.append(
                {
                    "id": candidate.id,
                    "source": candidate.source,
                    "pruned_count": pruned,
                    "predicted_cost": float(predicted_costs[selected_family]),
                    "predicted_costs": {
                        family: float(value) for family, value in predicted_costs.items()
                    },
                    "base_loss": float(base_loss),
                    "masked_loss": float(masked_loss),
                    "actual_delta_loss": float(actual_delta_loss),
                    "prediction_family": selected_family,
                    "mask_fingerprint": fingerprint,
                }
            )
            _restore_snapshot(model, snapshot)
            _assert_snapshot(model, snapshot)
    finally:
        _restore_snapshot(model, snapshot)
        _assert_snapshot(model, snapshot)

    correlations = {}
    actual_values = [float(record["actual_delta_loss"]) for record in records]
    for family in scores:
        predicted_values = [
            float(record["predicted_costs"][family]) for record in records  # type: ignore[index]
        ]
        correlations[family] = _correlation(predicted_values, actual_values)
    return {
        "records": records,
        "correlations": correlations,
        "score_families": list(scores),
        "default_score_family": default_score_family,
        "pruned_count": expected_pruned,
        "mask_count": len(records),
    }


evaluate_damage_surrogate = validate_damage_surrogate


__all__ = [
    "MaskCandidate",
    "evaluate_damage_surrogate",
    "validate_damage_surrogate",
]
