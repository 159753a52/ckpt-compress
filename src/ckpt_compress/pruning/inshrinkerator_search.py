"""Inshrinkerator-like per-layer-type pruning search.

Reproduces the core pruning allocation idea from the Inshrinkerator paper:
- Group parameters by layer type (attn/mlp/conv/fc)
- All layers of the same type share a single quantile threshold
- Two-phase grid search for optimal per-type pruning ratios
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .param_schema import get_prunable_types

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Search hyperparameters, model/dataset agnostic."""
    coarse_candidates: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4]
    )
    fine_step: float = 0.1
    fine_radius: int = 1
    metrics: List[str] = field(
        default_factory=lambda: ['magnitude', 'sensitivity']
    )
    epsilons: List[float] = field(
        default_factory=lambda: [0.01, 0.05]
    )
    max_type_ratio: float = 0.95


@dataclass
class SearchResult:
    """Result of a single search run (fixed epsilon)."""
    epsilon: float
    best_metric: Optional[str]
    per_type_ratios: Dict[str, float]
    actual_global_ratio: float
    baseline_metrics: Dict[str, float]
    pruned_metrics: Dict[str, float]
    quality_drop_pct: float
    search_log: List[Dict]


def apply_pruning_per_type(
    model: nn.Module,
    scores: Dict[str, torch.Tensor],
    per_type_ratios: Dict[str, float],
    type_map: Dict[str, str],
    device: str = 'cpu',
) -> Tuple[nn.Module, Dict[str, torch.Tensor], Dict[str, float], float]:
    """Prune model using per-layer-type quantile thresholds.

    All scores within the same type are concatenated, and a single
    quantile threshold is computed for that type.

    Returns:
        (model, masks, actual_ratio_by_type, actual_global_ratio)
    """
    type_scores: Dict[str, List[torch.Tensor]] = {}
    for name, score in scores.items():
        lt = type_map.get(name, 'skip')
        if lt == 'skip' or lt not in per_type_ratios:
            continue
        type_scores.setdefault(lt, []).append(score.flatten())

    type_thresholds: Dict[str, float] = {}
    for lt, score_list in type_scores.items():
        ratio = per_type_ratios[lt]
        if ratio <= 0:
            type_thresholds[lt] = -1.0
            continue
        cat = torch.cat(score_list)
        k = max(1, int(len(cat) * ratio))
        k = min(k, len(cat))
        type_thresholds[lt] = torch.kthvalue(cat, k).values.item()

    masks: Dict[str, torch.Tensor] = {}
    model_params = dict(model.named_parameters())
    total_pruned = 0
    total_params = 0
    type_pruned: Dict[str, int] = {}
    type_total: Dict[str, int] = {}

    for name, score in scores.items():
        lt = type_map.get(name, 'skip')
        if lt == 'skip' or lt not in type_thresholds:
            continue
        if name not in model_params:
            continue

        threshold = type_thresholds[lt]
        if threshold < 0:
            mask = torch.ones_like(score)
        else:
            mask = (score > threshold).float()

        masks[name] = mask
        model_params[name].data.mul_(mask.to(device))

        n_pruned = int((mask == 0).sum().item())
        n_total = mask.numel()
        total_pruned += n_pruned
        total_params += n_total
        type_pruned[lt] = type_pruned.get(lt, 0) + n_pruned
        type_total[lt] = type_total.get(lt, 0) + n_total

    actual_global = total_pruned / total_params if total_params > 0 else 0.0
    actual_by_type = {}
    for lt in type_total:
        if type_total[lt] > 0:
            actual_by_type[lt] = type_pruned.get(lt, 0) / type_total[lt]
    return model, masks, actual_by_type, actual_global


def estimate_global_ratio(
    per_type_ratios: Dict[str, float],
    type_param_counts: Dict[str, int],
) -> float:
    """Estimate global pruning ratio from per-type ratios and param counts."""
    total_pruned = sum(
        type_param_counts.get(lt, 0) * r
        for lt, r in per_type_ratios.items()
    )
    total_params = sum(type_param_counts.values())
    return total_pruned / total_params if total_params > 0 else 0.0


def _count_params_by_type(
    scores: Dict[str, torch.Tensor],
    type_map: Dict[str, str],
    prunable_types: Sequence[str],
) -> Dict[str, int]:
    counts: Dict[str, int] = {lt: 0 for lt in prunable_types}
    for name, score in scores.items():
        lt = type_map.get(name, 'skip')
        if lt in counts:
            counts[lt] += score.numel()
    return counts


def _generate_candidates(
    prunable_types: List[str],
    candidates_per_type: List[float],
    max_ratio: float,
) -> List[Dict[str, float]]:
    clipped = [min(c, max_ratio) for c in candidates_per_type]
    combos = list(product(clipped, repeat=len(prunable_types)))
    return [dict(zip(prunable_types, combo)) for combo in combos]


def _refine_candidates(
    best: Dict[str, float],
    prunable_types: List[str],
    fine_step: float,
    fine_radius: int,
    max_ratio: float,
) -> List[Dict[str, float]]:
    """Generate fine-grid candidates around the best solution."""
    per_type_options: Dict[str, List[float]] = {}
    for lt in prunable_types:
        center = best.get(lt, 0.0)
        options = []
        for delta in range(-fine_radius, fine_radius + 1):
            val = round(center + delta * fine_step, 4)
            val = max(0.0, min(val, max_ratio))
            if val not in options:
                options.append(val)
        per_type_options[lt] = sorted(options)

    combos = list(product(*[per_type_options[lt] for lt in prunable_types]))
    return [dict(zip(prunable_types, c)) for c in combos]


def _metric_name_to_score_key(metric_name: str) -> str:
    return {'magnitude': 'magnitude', 'sensitivity': 'first-order'}.get(
        metric_name, metric_name
    )


def search_best_config(
    model_factory: Callable[[], nn.Module],
    scores_by_metric: Dict[str, Dict[str, torch.Tensor]],
    type_map: Dict[str, str],
    cached_eval: List,
    task_type: str,
    device: str,
    evaluate_fn: Callable,
    quality_drop_fn: Callable,
    epsilon: float,
    config: Optional[SearchConfig] = None,
    model_family: str = 'gpt2',
) -> SearchResult:
    """Two-phase search for optimal per-type pruning ratios.

    Args:
        model_factory: callable returning a fresh model copy
        scores_by_metric: {'magnitude': {name: score}, 'first-order': {name: score}}
        type_map: {param_name: layer_type}
        cached_eval: cached evaluation batches
        task_type: 'lm' | 'cls' | 'cv' | 'reg'
        device: torch device string
        evaluate_fn: evaluate(model, cached_eval, task_type, device) -> metrics dict
        quality_drop_fn: compute_quality_drop(baseline, pruned, task_type) -> float
        epsilon: quality constraint threshold
        config: search configuration
        model_family: model family name for type rules
    """
    if config is None:
        config = SearchConfig()

    prunable_types = get_prunable_types(model_family)

    first_metric_key = list(scores_by_metric.keys())[0]
    type_counts = _count_params_by_type(
        scores_by_metric[first_metric_key], type_map, prunable_types
    )
    active_types = [lt for lt in prunable_types if type_counts.get(lt, 0) > 0]

    # Evaluate baseline (unpruned)
    baseline_model = model_factory()
    baseline_model = baseline_model.to(device)
    baseline_metrics = evaluate_fn(baseline_model, cached_eval, task_type, device)
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    search_log: List[Dict] = []
    best_global_ratio = -1.0
    best_result: Optional[Dict] = None
    best_metric_name: Optional[str] = None

    def _eval_candidate(
        scores: Dict[str, torch.Tensor],
        cand: Dict[str, float],
        phase: str,
        metric_name: str,
    ) -> Tuple[float, float, Dict[str, float]]:
        model_copy = model_factory()
        model_copy = model_copy.to(device)
        _, _, _, actual_global = apply_pruning_per_type(
            model_copy, scores, cand, type_map, device
        )
        metrics = evaluate_fn(model_copy, cached_eval, task_type, device)
        drop = quality_drop_fn(baseline_metrics, metrics, task_type)

        search_log.append({
            'phase': phase,
            'metric': metric_name,
            'per_type_ratios': dict(cand),
            'est_global_ratio': estimate_global_ratio(cand, type_counts),
            'actual_global_ratio': actual_global,
            'drop': drop,
            'feasible': drop <= epsilon,
        })
        logger.info(
            "%s %s | %s | global=%.3f | drop=%.4f | %s",
            phase, metric_name, cand, actual_global, drop,
            "OK" if drop <= epsilon else "FAIL",
        )

        del model_copy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metrics_float = {k: float(v) for k, v in metrics.items()}
        return actual_global, drop, metrics_float

    # Phase 1: coarse grid
    for metric_name in config.metrics:
        metric_key = _metric_name_to_score_key(metric_name)
        if metric_key not in scores_by_metric:
            continue
        scores = scores_by_metric[metric_key]

        candidates = _generate_candidates(
            active_types, config.coarse_candidates, config.max_type_ratio
        )
        candidates.sort(
            key=lambda c: estimate_global_ratio(c, type_counts), reverse=True
        )

        for cand in candidates:
            est = estimate_global_ratio(cand, type_counts)
            if est <= best_global_ratio:
                continue

            actual_global, drop, metrics = _eval_candidate(
                scores, cand, 'coarse', metric_name
            )
            if drop <= epsilon and actual_global > best_global_ratio:
                best_global_ratio = actual_global
                best_result = {
                    'per_type_ratios': dict(cand),
                    'actual_global_ratio': actual_global,
                    'pruned_metrics': metrics,
                    'drop': drop,
                }
                best_metric_name = metric_name

    # Phase 2: fine grid around best
    if best_result is not None and best_metric_name is not None:
        metric_key = _metric_name_to_score_key(best_metric_name)
        scores = scores_by_metric[metric_key]

        fine_cands = _refine_candidates(
            best_result['per_type_ratios'],
            active_types,
            config.fine_step,
            config.fine_radius,
            config.max_type_ratio,
        )
        fine_cands.sort(
            key=lambda c: estimate_global_ratio(c, type_counts), reverse=True
        )

        for cand in fine_cands:
            est = estimate_global_ratio(cand, type_counts)
            if est <= best_global_ratio:
                continue

            actual_global, drop, metrics = _eval_candidate(
                scores, cand, 'fine', best_metric_name
            )
            if drop <= epsilon and actual_global > best_global_ratio:
                best_global_ratio = actual_global
                best_result = {
                    'per_type_ratios': dict(cand),
                    'actual_global_ratio': actual_global,
                    'pruned_metrics': metrics,
                    'drop': drop,
                }

    # Build result
    baseline_float = {k: float(v) for k, v in baseline_metrics.items()}

    if best_result is None:
        return SearchResult(
            epsilon=epsilon,
            best_metric=None,
            per_type_ratios={lt: 0.0 for lt in active_types},
            actual_global_ratio=0.0,
            baseline_metrics=baseline_float,
            pruned_metrics=baseline_float,
            quality_drop_pct=0.0,
            search_log=search_log,
        )

    return SearchResult(
        epsilon=epsilon,
        best_metric=best_metric_name,
        per_type_ratios=best_result['per_type_ratios'],
        actual_global_ratio=best_result['actual_global_ratio'],
        baseline_metrics=baseline_float,
        pruned_metrics=best_result['pruned_metrics'],
        quality_drop_pct=best_result['drop'] * 100,
        search_log=search_log,
    )


def save_search_result(result: SearchResult, output_dir: str, name: str) -> str:
    """Save search result as JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    data = asdict(result)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    return str(path)


def load_search_result(json_path: str) -> SearchResult:
    """Load search result from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    return SearchResult(**data)
