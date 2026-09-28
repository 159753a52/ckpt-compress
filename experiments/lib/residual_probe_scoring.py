"""Coverage-tracked Gauss--Newton probe scores for small causal-LM residual runs.

Objective (identical to the Taylor and first-order scorers):

    L = (1/B) sum_b (1/N_b) sum_{t in b} CE(z_t, y_t^true)

over the ``B`` scoring batches, where ``N_b`` counts the valid shifted positions
of batch ``b`` (labels different from the ignore index after applying the
attention mask).  The Gauss--Newton matrix of ``L`` is

    F = (1/B) sum_b (1/N_b) sum_{t in b} sum_y p_ty v_ty v_ty^T,
    v_ty = J_t^T (e_y - p_t).

Each probe round ``k`` selects one token per position and one Rademacher sign
``xi_t``; a single ordinary backward of ``sum_t c_t CE(z_t, y_t)`` over all
scoring batches yields ``w_k = -sum_t c_t v_{t y_t}``.  The curvature product is
``u = sum_k w_k (w_k . d)`` with the inner product over every eligible tensor
(full Gauss--Newton, not block diagonal), and the score is

    s = | -g * d + 0.5 * d * u |

with ``g`` the true-label gradient of ``L``.

Estimators (``normalizer = sqrt(B * N_b)``):

- ``sample``: ``y_t ~ p_t``; a token already drawn at the same position gets
  weight 0; ``c_t = xi_t sqrt(p_ty) / normalizer``.  ``E_xi[sum_k w_k w_k^T]``
  equals the covered operator ``F_S``.
- ``topk``: round ``k`` uses the k-th most probable token; same weight.
- ``headtail``: ``m - 1`` deterministic head rounds, then one tail token drawn
  from ``p_t`` restricted to the complement of the head (renormalised) with
  ``c_t = xi_t sqrt(1 - q_head) / normalizer``; unbiased for ``F``.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.lib.losses import (
    CAUSAL_LM_IGNORE_INDEX,
    extract_logits,
    move_batch_to_device,
)
from experiments.lib.probe_coverage import (
    CoverageTracker,
    coverage_probe_weights,
    rademacher,
    weighted_token_ce,
)
from experiments.lib.residual_masks import TensorDict
from experiments.lib.residual_protocol import PROBE_ESTIMATORS
from experiments.lib.residual_runtime import empty_device_cache, lm_loss
from experiments.lib.residual_scoring import (
    _restore_scoring_state,
    _snapshot_scoring_state,
    _structural_blocks_for_scoring,
    _validate_output_score,
    _validate_scoring_inputs,
    model_checksum,
)

SignsFn = Callable[[int, int, int], torch.Tensor]


def valid_shifted_logits(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``[N, V]`` logits and true labels at the valid causal-LM positions.

    The valid set matches :func:`experiments.lib.losses.causal_lm_loss`.
    """
    batch_on_device = move_batch_to_device(batch, device)
    input_ids = batch_on_device["input_ids"]
    labels = batch_on_device["labels"]
    attention_mask = batch_on_device.get("attention_mask")
    outputs = model(
        input_ids, **({"attention_mask": attention_mask} if attention_mask is not None else {})
    )
    logits = extract_logits(outputs)
    if logits.shape[:-1] != input_ids.shape:
        raise ValueError("causal-LM logits must match the input batch and sequence dimensions")
    effective_labels = labels.clone()
    if attention_mask is not None:
        effective_labels.masked_fill_(attention_mask == 0, CAUSAL_LM_IGNORE_INDEX)
    shift_labels = effective_labels[..., 1:]
    valid = shift_labels != CAUSAL_LM_IGNORE_INDEX
    if not bool(valid.any().item()):
        raise ValueError("causal-LM batch contains no supervised tokens")
    return logits[..., :-1, :][valid], shift_labels[valid]


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _lower_quantile(values: torch.Tensor, fraction: float) -> float:
    flat = values.detach().double().reshape(-1).cpu()
    index = max(1, math.ceil(fraction * flat.numel()))
    return float(flat.kthvalue(index).values)


def compute_probe_gn_scores(
    model: nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    layers: Sequence[Sequence[str]],
    delta: Mapping[str, torch.Tensor],
    device: str,
    *,
    estimator: str,
    rounds: int,
    seed: int,
    task_type: str = "lm",
    model_family: str = "gpt2",
    block_parameter_names: Sequence[Sequence[str]] | None = None,
    signs_fn: SignsFn | None = None,
    return_components: bool = False,
) -> Tuple[TensorDict | Dict[str, TensorDict], Dict[str, object]]:
    """Score residual coordinates with a coverage-tracked Gauss--Newton probe.

    ``signs_fn(round, batch, positions)`` replaces the Rademacher signs; it is
    intended for deterministic verification of the estimator identity.
    """
    if task_type != "lm":
        raise ValueError(
            "Probe Gauss-Newton scoring requires a causal-LM token cross-entropy "
            f"objective; task_type={task_type!r} is not supported"
        )
    if estimator not in PROBE_ESTIMATORS:
        raise ValueError(f"Unknown probe estimator {estimator!r}; expected {PROBE_ESTIMATORS}")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError(f"rounds must be a positive integer, got {rounds!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if not batches:
        raise ValueError("batches must contain at least one scoring batch")

    named_params = dict(model.named_parameters())
    structural_blocks = _structural_blocks_for_scoring(
        model, layers, model_family, block_parameter_names
    )
    _validate_scoring_inputs(named_params, layers, structural_blocks, delta, device)
    eligible_names = [name for layer in layers for name in layer]
    original_modes, original_requires_grad, original_gradients = _snapshot_scoring_state(model)
    versions_before = {name: p._version for name, p in named_params.items()}
    checksum_before = model_checksum(model)
    head_rounds = rounds if estimator == "topk" else rounds - 1 if estimator == "headtail" else 0
    phase_seconds = {
        "gradient": 0.0,
        "probe_forward": 0.0,
        "probe_selection": 0.0,
        "probe_backward": 0.0,
        "curvature_update": 0.0,
        "score_assembly": 0.0,
    }
    per_round: List[Dict[str, object]] = []
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    batch_count = len(batches)
    started = time.perf_counter()
    model.eval()
    try:
        eligible_set = set(eligible_names)
        for name, parameter in named_params.items():
            parameter.requires_grad_(name in eligible_set)
        params = [named_params[name] for name in eligible_names]
        probes = [delta[name].to(device, dtype=torch.float32) for name in eligible_names]

        _sync(device)
        phase_started = time.perf_counter()
        gradient = [torch.zeros_like(probe) for probe in probes]
        for batch in batches:
            with sdpa_kernel(SDPBackend.MATH):
                loss = lm_loss(model, batch, device)
                grads = torch.autograd.grad(loss, params, allow_unused=True)
            for name, accumulator, value in zip(eligible_names, gradient, grads):
                if value is None:
                    raise RuntimeError(f"Probe gradient is missing for {name!r}")
                accumulator.add_(value.detach().float(), alpha=1.0 / batch_count)
            del loss, grads
        _sync(device)
        phase_seconds["gradient"] += time.perf_counter() - phase_started

        curvature = [torch.zeros_like(probe) for probe in probes]
        trackers: List[CoverageTracker] = []
        head_tokens: List[torch.Tensor | None] = []
        positions: List[int] = []
        vocab_size = None
        for round_index in range(rounds):
            round_started = time.perf_counter()
            direction = [torch.zeros_like(probe) for probe in probes]
            new_tokens = 0
            round_positions = 0
            weight_sq = 0.0
            head_mass: List[torch.Tensor] = []
            for batch_index, batch in enumerate(batches):
                _sync(device)
                t0 = time.perf_counter()
                with sdpa_kernel(SDPBackend.MATH):
                    logits, _ = valid_shifted_logits(model, batch, device)
                _sync(device)
                t1 = time.perf_counter()
                with torch.no_grad():
                    probs = torch.softmax(logits.detach().float(), dim=-1)
                    count = int(probs.shape[0])
                    if round_index == 0:
                        vocab_size = int(probs.shape[1])
                        if head_rounds > vocab_size or (
                            estimator == "headtail" and head_rounds >= vocab_size
                        ):
                            raise ValueError(
                                f"{estimator} probe with {rounds} rounds exceeds vocabulary "
                                f"size {vocab_size}"
                            )
                        positions.append(count)
                        trackers.append(CoverageTracker(count, rounds, device=probs.device))
                        head_tokens.append(
                            probs.topk(head_rounds, dim=-1).indices if head_rounds else None
                        )
                    elif count != positions[batch_index]:
                        raise RuntimeError("Scoring batch positions changed across probe rounds")
                    normalizer = batch_count * count
                    if signs_fn is None:
                        signs = rademacher(count, generator, probs.device)
                    else:
                        signs = signs_fn(round_index, batch_index, count).to(
                            probs.device, torch.float64
                        )
                        if signs.shape != (count,):
                            raise ValueError("signs_fn must return one sign per position")
                    tracker = trackers[batch_index]
                    if estimator == "sample":
                        tokens = torch.multinomial(probs, 1, generator=generator).reshape(-1)
                        token_probs = probs.gather(1, tokens[:, None]).reshape(-1)
                        is_new = tracker.update(tokens, token_probs)
                        weights = coverage_probe_weights(token_probs, is_new, signs, normalizer)
                    elif round_index < head_rounds:
                        heads = head_tokens[batch_index]
                        assert heads is not None
                        tokens = heads[:, round_index]
                        token_probs = probs.gather(1, tokens[:, None]).reshape(-1)
                        is_new = tracker.update(tokens, token_probs)
                        weights = coverage_probe_weights(token_probs, is_new, signs, normalizer)
                    else:
                        tail = probs.clone()
                        heads = head_tokens[batch_index]
                        if heads is not None:
                            head_mass.append(probs.gather(1, heads).double().sum(dim=1))
                            tail.scatter_(1, heads, 0.0)
                        tail_mass = tail.double().sum(dim=1)
                        degenerate = tail_mass <= 0
                        if bool(degenerate.any().item()):
                            fallback = torch.ones_like(tail[degenerate])
                            if heads is not None:
                                fallback.scatter_(1, heads[degenerate], 0.0)
                            tail[degenerate] = fallback
                        tokens = torch.multinomial(tail, 1, generator=generator).reshape(-1)
                        token_probs = probs.gather(1, tokens[:, None]).reshape(-1)
                        is_new = tracker.update(tokens, token_probs)
                        tail_scale = torch.where(
                            degenerate, torch.zeros_like(tail_mass), tail_mass.clamp_min(0.0)
                        ).sqrt()
                        weights = signs.double() * tail_scale / math.sqrt(normalizer)
                    new_tokens += int(is_new.sum().item())
                    round_positions += count
                    weight_sq += float(weights.square().sum().item())
                _sync(device)
                t2 = time.perf_counter()
                probe_loss = weighted_token_ce(logits, tokens, weights)
                grads = torch.autograd.grad(probe_loss, params, allow_unused=True)
                for name, accumulator, value in zip(eligible_names, direction, grads):
                    if value is None:
                        raise RuntimeError(f"Probe direction is missing for {name!r}")
                    accumulator.add_(value.detach().float())
                del logits, probs, probe_loss, grads
                _sync(device)
                t3 = time.perf_counter()
                phase_seconds["probe_forward"] += t1 - t0
                phase_seconds["probe_selection"] += t2 - t1
                phase_seconds["probe_backward"] += t3 - t2

            t4 = time.perf_counter()
            with torch.no_grad():
                alpha = torch.zeros((), dtype=torch.float64, device=probes[0].device)
                for value, probe in zip(direction, probes):
                    alpha += (value.double() * probe.double()).sum()
                alpha_value = float(alpha.item())
                for accumulator, value in zip(curvature, direction):
                    accumulator.add_(value, alpha=alpha_value)
            del direction
            _sync(device)
            phase_seconds["curvature_update"] += time.perf_counter() - t4

            mass = torch.cat([tracker.mass for tracker in trackers])
            record: Dict[str, object] = {
                "round": round_index + 1,
                "kind": (
                    "sample"
                    if estimator == "sample"
                    else "head" if round_index < head_rounds else "tail"
                ),
                "mean_mass": float(mass.mean().item()),
                "p05_mass": _lower_quantile(mass, 0.05),
                "min_mass": float(mass.min().item()),
                "new_tokens": new_tokens,
                "positions": round_positions,
                "repeat_rate": 1.0 - new_tokens / round_positions,
                "alpha": alpha_value,
                "weight_sq_sum": weight_sq,
                "round_seconds": time.perf_counter() - round_started,
            }
            if head_mass:
                head = torch.cat(head_mass)
                record["head_mean_mass"] = float(head.mean().item())
                record["head_p05_mass"] = _lower_quantile(head, 0.05)
            per_round.append(record)

        t5 = time.perf_counter()
        scores: TensorDict = {}
        gradient_terms: TensorDict = {}
        curvature_terms: TensorDict = {}
        with torch.no_grad():
            for name, grad_value, u_value, probe in zip(
                eligible_names, gradient, curvature, probes
            ):
                signed = -grad_value * probe + 0.5 * probe * u_value
                score = signed.abs().cpu()
                _validate_output_score(named_params[name], score, name)
                scores[name] = score
                if return_components:
                    gradient_terms[name] = (-grad_value * probe).cpu()
                    curvature_terms[name] = u_value.cpu()
        phase_seconds["score_assembly"] += time.perf_counter() - t5
        del gradient, curvature, probes
        total_draws = sum(tracker.draws for tracker in trackers)
        total_repeats = sum(tracker.repeats for tracker in trackers)
    finally:
        _restore_scoring_state(
            model,
            original_modes,
            original_requires_grad,
            original_gradients,
        )
        empty_device_cache(device)

    versions_after = {name: p._version for name, p in named_params.items()}
    checksum_after = model_checksum(model)
    changed_versions = [
        name for name in versions_before if versions_before[name] != versions_after[name]
    ]
    checksum_delta = [after - before for before, after in zip(checksum_before, checksum_after)]
    if changed_versions or checksum_delta != [0.0, 0.0]:
        raise RuntimeError(
            f"Scoring mutated model weights: versions={changed_versions[:5]}, "
            f"checksum_delta={checksum_delta}"
        )
    meta: Dict[str, object] = {
        "score_kind": f"probe_gn_{estimator}",
        "total_seconds": time.perf_counter() - started,
        "phase_seconds": phase_seconds,
        "checksum_before": list(checksum_before),
        "checksum_after": list(checksum_after),
        "checksum_delta": checksum_delta,
        "changed_parameter_versions": changed_versions,
        "optimizer_constructed": False,
        "model_mode": "eval",
        "scoring_batches": batch_count,
        "task_type": task_type,
        "model_family": model_family,
        "probe": {
            "estimator": estimator,
            "rounds": rounds,
            "head_rounds": head_rounds,
            "seed": seed,
            "signs": "rademacher" if signs_fn is None else "override",
            "curvature": "full_gauss_newton",
            "normalization": "batch_mean_of_token_mean",
            "positions_per_batch": positions,
            "total_positions": sum(positions),
            "vocab_size": vocab_size,
            "cumulative_draws": total_draws,
            "cumulative_repeats": total_repeats,
            "repeat_rate": total_repeats / max(1, total_draws),
            "per_round": per_round,
        },
    }
    if return_components:
        return {"score": scores, "gradient_term": gradient_terms, "curvature": curvature_terms}, meta
    return scores, meta


__all__ = ["compute_probe_gn_scores", "valid_shifted_logits"]
