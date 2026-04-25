"""
Inshrinkerator-like per-layer-type pruning ratio search.

Searches for the best per-type (attn/mlp/...) pruning ratios under a
quality constraint, using magnitude and sensitivity metrics.

Usage:
    python experiments/scripts/run_inshrinkerator_like_search.py \
        --model gpt2-small --dataset wikitext103 \
        --epsilons 0.01,0.05 \
        --num_steps 50 --eval_batches 20 \
        --device cuda

    python experiments/scripts/run_inshrinkerator_like_search.py \
        --model bert-large --dataset sst2 \
        --checkpoint checkpoints/bert_large_sst2_1000steps/checkpoint_step_1000_final.pt \
        --epsilons 0.01,0.05 \
        --device cuda
"""

import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"

import sys
from pathlib import Path
import argparse
import copy
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.lib.models import load_model
from experiments.lib.data import get_data_loaders, cache_batches
from experiments.lib.evaluation import evaluate, compute_quality_drop
from experiments.lib.importance_compare.scoring import compute_scores_by_method
from dacp.pruning.param_schema import build_type_map
from baselines.inshrinkerator.per_type_search import (
    SearchConfig,
    search_best_config,
    save_search_result,
)


def _get_model_family(model_name: str) -> str:
    if 'gpt2' in model_name:
        return 'gpt2'
    elif 'bert' in model_name:
        return 'bert'
    elif 'resnet' in model_name:
        return 'resnet'
    return 'gpt2'


def main():
    parser = argparse.ArgumentParser(
        description='Inshrinkerator-like per-layer-type pruning search'
    )
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--epsilons', type=str, default='0.01,0.05')
    parser.add_argument('--coarse_candidates', type=str, default='0.0,0.2,0.4')
    parser.add_argument('--fine_step', type=float, default=0.1)
    parser.add_argument('--fine_radius', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--hvp_batches', type=int, default=4)
    parser.add_argument('--eval_batches', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--output_dir', type=str,
        default='results/paper_results/inshrinkerator_search',
    )
    args = parser.parse_args()

    epsilons = [float(e) for e in args.epsilons.split(',')]
    coarse = [float(c) for c in args.coarse_candidates.split(',')]
    model_family = _get_model_family(args.model)

    print("=" * 70)
    print("Inshrinkerator-like Per-Layer-Type Pruning Search")
    print("=" * 70)
    print(f"Model: {args.model} | Dataset: {args.dataset}")
    print(f"Epsilons: {epsilons}")
    print(f"Coarse candidates: {coarse}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # 1. Load model
    print("\n[1/5] Loading model...")
    model, model_family = load_model(
        args.model,
        pretrained=True,
        checkpoint_path=args.checkpoint,
        device='cpu',
        dataset_name=args.dataset,
    )

    # 2. Load data
    print(f"[2/5] Loading data ({args.dataset})...")
    train_loader, val_loader, task_type = get_data_loaders(
        args.model, args.dataset, args.batch_size, args.seq_length,
    )
    cached_train = cache_batches(train_loader, args.num_steps, task_type)
    cached_eval = cache_batches(val_loader, args.eval_batches, task_type)

    # 3. Compute importance scores (magnitude + first-order)
    print("[3/5] Computing importance scores...")
    model_for_scoring = copy.deepcopy(model).to(args.device)
    scores_by_metric = compute_scores_by_method(
        model_for_scoring,
        cached_train,
        task_type,
        methods=['magnitude', 'first-order'],
        alpha=args.alpha,
        hvp_batches=args.hvp_batches,
        model_family=model_family,
    )
    del model_for_scoring
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Build type map
    print("[4/5] Building type map...")
    weights = {n: p.detach().cpu() for n, p in model.named_parameters()}
    type_map = build_type_map(weights, model_family)

    # Print type distribution
    from collections import Counter
    type_counts = Counter(type_map.values())
    for lt, cnt in sorted(type_counts.items()):
        print(f"  {lt}: {cnt} params")

    # 5. Search
    print("[5/5] Running search...")
    search_cfg = SearchConfig(
        coarse_candidates=coarse,
        fine_step=args.fine_step,
        fine_radius=args.fine_radius,
        metrics=['magnitude', 'sensitivity'],
        epsilons=epsilons,
    )

    def model_factory():
        return copy.deepcopy(model)

    for eps in epsilons:
        print(f"\n--- epsilon = {eps} ---")
        result = search_best_config(
            model_factory=model_factory,
            scores_by_metric=scores_by_metric,
            type_map=type_map,
            cached_eval=cached_eval,
            task_type=task_type,
            device=args.device,
            evaluate_fn=evaluate,
            quality_drop_fn=compute_quality_drop,
            epsilon=eps,
            config=search_cfg,
            model_family=model_family,
        )

        print(f"  Best metric: {result.best_metric}")
        print(f"  Per-type ratios: {result.per_type_ratios}")
        print(f"  Actual global ratio: {result.actual_global_ratio:.4f}")
        print(f"  Quality drop: {result.quality_drop_pct:.2f}%")
        print(f"  Baseline: {result.baseline_metrics}")
        print(f"  Pruned:   {result.pruned_metrics}")

        name = f"search_{args.model}_{args.dataset}_eps{eps}"
        path = save_search_result(result, args.output_dir, name)
        print(f"  Saved: {path}")

    print("\nSearch complete!")


if __name__ == '__main__':
    main()
