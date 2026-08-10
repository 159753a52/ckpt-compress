"""
Table 1: 固定剪枝比例下的质量对比（论文核心实验）

对比方法:
  - magnitude + uniform
  - first-order + uniform (Inshrinkerator-style)
  - residual-magnitude + uniform (ExCP-style)
  - second-order-hvp + uniform (Ours, ablation)
  - second-order-hvp + gamma-adaptive (Ours, full)

运行示例:
    # GPT-2 Small (快速验证)
    python experiments/scripts/run_table1.py \
        --model gpt2-small --dataset wikitext103 \
        --prune_ratios 0.1,0.2,0.3,0.4 \
        --num_steps 50 --eval_batches 10 \
        --device cuda

    # GPT-2 Medium (论文实验)
    python experiments/scripts/run_table1.py \
        --model gpt2-medium --dataset wikitext103 \
        --checkpoint checkpoints/gpt2_medium_wikitext103/final.pt \
        --prune_ratios 0.1,0.2,0.3,0.4 \
        --num_steps 100 --eval_batches 20 \
        --device cuda

    # BERT-Large on SST-2
    python experiments/scripts/run_table1.py \
        --model bert-large --dataset sst2 \
        --checkpoint checkpoints/bert_large_sst2_1000steps/final.pt \
        --prune_ratios 0.1,0.2,0.3,0.4 \
        --device cuda

    # ResNet18 on CIFAR-10
    python experiments/scripts/run_table1.py \
        --model resnet18 --dataset cifar10 \
        --prune_ratios 0.1,0.2,0.3,0.4 \
        --device cuda
"""

import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import gc
import sys
from pathlib import Path
import argparse
import copy
import torch

# 项目路径
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.lib.models import load_model
from experiments.lib.data import get_data_loaders, cache_batches
from experiments.lib.evaluation import evaluate
from experiments.lib.results import save_results, print_results_table
from experiments.lib.importance_compare.scoring import compute_scores_by_method
from dacp.pruning import apply_pruning
from dacp.pruning.allocation import get_allocation_strategy
from dacp.pruning.importance import combine_scores_2d_with_protection, apply_magnitude_protection
from baselines.inshrinkerator.per_type_allocation import PerTypeAllocation


# 要对比的方法组合
METHODS = [
    {'importance': 'magnitude',          'allocation': 'uniform'},
    {'importance': 'magnitude',          'allocation': 'weibull-adaptive'},
    {'importance': 'first-order',        'allocation': 'uniform'},
    {'importance': 'first-order',        'allocation': 'weibull-adaptive'},
    {'importance': 'residual-magnitude', 'allocation': 'uniform'},
    {'importance': 'second-order-hvp',   'allocation': 'uniform'},
    {'importance': 'second-order-hvp',   'allocation': 'weibull-adaptive'},
]


def run_single_method(model_factory, scores, method_config, prune_ratios, 
                      cached_eval, task_type, device, baseline_metrics,
                      allocation_scores=None):
    """对单个方法运行所有剪枝率的实验。
    
    Args:
        allocation_scores: 若非 None，用此得分做层间分配（Weibull），
                           而 scores 用于层内剪枝决策。实现2D解耦。
    """
    importance = method_config['importance']
    allocation = method_config['allocation']
    method_name = f"{importance}+{allocation}"

    allocator = get_allocation_strategy(allocation)
    # 层间分配用 allocation_scores（若提供），否则用 scores
    alloc_input = allocation_scores if allocation_scores is not None else scores
    results = []
    
    for ratio in prune_ratios:
        # 复制模型
        model_copy = model_factory()
        model_copy = model_copy.to(device)
        
        # 分配剪枝率（用 allocation_scores 或 scores）
        layer_ratios = allocator.allocate(alloc_input, ratio)
        
        # 剪枝
        _, masks, actual_ratio = apply_pruning(model_copy, scores, layer_ratios, device=device)
        
        # 评估
        metrics = evaluate(model_copy, cached_eval, task_type, device)
        
        result = {
            'method': method_name,
            'importance': importance,
            'allocation': allocation,
            'target_ratio': ratio,
            'actual_ratio': actual_ratio,
        }
        
        # 添加指标
        for k, v in metrics.items():
            result[k] = v
        
        # 计算相对变化
        if 'loss' in baseline_metrics:
            result['loss_increase'] = metrics['loss'] - baseline_metrics['loss']
            result['loss_increase_pct'] = (result['loss_increase'] / baseline_metrics['loss']) * 100
        if 'perplexity' in baseline_metrics:
            result['baseline_ppl'] = baseline_metrics['perplexity']
        if 'accuracy' in baseline_metrics:
            result['accuracy_drop'] = baseline_metrics['accuracy'] - metrics.get('accuracy', 0)
            result['baseline_acc'] = baseline_metrics['accuracy']
        
        results.append(result)
        
        print(f"  {method_name} | ratio={ratio:.0%} (actual={actual_ratio:.1%}) | "
              + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        
        # 释放内存
        del model_copy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def _run_per_type_method(model_factory, scores, allocator, method_config,
                        prune_ratios, cached_eval, task_type, device,
                        baseline_metrics):
    """Run per-type allocation method across all prune ratios."""
    importance = method_config['importance']
    method_name = f"{importance}+per-type"

    results = []
    for ratio in prune_ratios:
        model_copy = model_factory()
        model_copy = model_copy.to(device)

        layer_ratios = allocator.allocate(scores, ratio)
        _, masks, actual_ratio = apply_pruning(model_copy, scores, layer_ratios, device=device)
        metrics = evaluate(model_copy, cached_eval, task_type, device)

        result = {
            'method': method_name,
            'importance': importance,
            'allocation': 'per-type',
            'target_ratio': ratio,
            'actual_ratio': actual_ratio,
        }
        for k, v in metrics.items():
            result[k] = v
        if 'loss' in baseline_metrics:
            result['loss_increase'] = metrics['loss'] - baseline_metrics['loss']
            result['loss_increase_pct'] = (result['loss_increase'] / baseline_metrics['loss']) * 100
        if 'perplexity' in baseline_metrics:
            result['baseline_ppl'] = baseline_metrics['perplexity']
        if 'accuracy' in baseline_metrics:
            result['accuracy_drop'] = baseline_metrics['accuracy'] - metrics.get('accuracy', 0)
            result['baseline_acc'] = baseline_metrics['accuracy']

        results.append(result)
        print(f"  {method_name} | ratio={ratio:.0%} (actual={actual_ratio:.1%}) | "
              + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        del model_copy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description='Table 1: 固定剪枝比例下的质量对比')
    parser.add_argument('--model', type=str, required=True, help='模型名称')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
    parser.add_argument('--checkpoint', type=str, default=None, help='微调检查点路径')
    parser.add_argument('--reference_checkpoint', type=str, default=None, 
                        help='参考检查点路径（ExCP 残差剪枝用）')
    parser.add_argument('--prune_ratios', type=str, default='0.1,0.2,0.3,0.4')
    parser.add_argument('--methods', type=str, default=None,
                        help='指定方法（逗号分隔），默认全部运行')
    parser.add_argument('--alpha', type=str, default='0.1', help='二阶项权重（支持逗号分隔多值扫描，如 0.05,0.1,0.2）')
    parser.add_argument('--num_steps', type=int, default=100, help='梯度累积步数')
    parser.add_argument('--hvp_batches', type=int, default=8)
    parser.add_argument('--hvp_mode', type=str, default='block', choices=['full', 'block'])
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--grad_batches_first_order', type=int, default=None,
                        help='给 first-order 方法单独指定梯度 batch 数。'
                             '默认 None 表示与 --hvp_batches 相同。')
    parser.add_argument('--eval_batches', type=int, default=20, help='评估批次数')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='对一阶/二阶项做 per-tensor 尺度归一化')
    parser.add_argument('--per_type_json', type=str, default=None,
                        help='Inshrinkerator-like 搜索结果 JSON 路径（per-type 方法用）')
    parser.add_argument('--scoring_mode', type=str, default='1d',
                        choices=['1d', '2d', 'protected'],
                        help='得分模式：1d=仅damage score（默认），'
                             '2d=magnitude+damage rank组合，'
                             'protected=magnitude保护+damage score')
    parser.add_argument('--protection_ratio', type=float, default=0.001,
                        help='Protection比例（2d/protected模式），默认0.1%%')
    parser.add_argument('--output_dir', type=str, default='results/paper_results/table1')
    args = parser.parse_args()
    
    prune_ratios = [float(r) for r in args.prune_ratios.split(',')]
    alpha_values = [float(a) for a in args.alpha.split(',')]
    
    print("=" * 70)
    print("Table 1: 固定剪枝比例下的质量对比")
    print("=" * 70)
    print(f"模型: {args.model} | 数据集: {args.dataset}")
    print(f"剪枝率: {prune_ratios}")
    print(f"alpha 值: {alpha_values}")
    print(f"设备: {args.device}")
    print("=" * 70)
    
    # 1. 加载模型和数据
    print("\n[1/6] 加载模型...")
    model, model_family = load_model(
        args.model,
        pretrained=True,
        checkpoint_path=args.checkpoint,
        device='cpu',
        dataset_name=args.dataset,
    )

    print(f"[2/6] 加载数据 ({args.dataset})...")
    train_loader, val_loader, task_type = get_data_loaders(
        args.model, args.dataset, args.batch_size, args.seq_length)
    
    # 2. 缓存批次
    print(f"[3/6] 缓存批次 (train={args.num_steps}, eval={args.eval_batches})...")
    cached_train = cache_batches(train_loader, args.num_steps, task_type)
    cached_eval = cache_batches(val_loader, args.eval_batches, task_type)
    
    # 3. 确定要运行的方法
    methods_to_run = list(METHODS)

    per_type_allocator = None
    if args.per_type_json is not None:
        model_family = 'gpt2'
        if 'bert' in args.model:
            model_family = 'bert'
        elif 'resnet' in args.model:
            model_family = 'resnet'
        per_type_allocator = PerTypeAllocation.from_json(
            args.per_type_json, model_family=model_family
        )
        methods_to_run.append({'importance': 'per-type-auto', 'allocation': 'per-type'})

    if args.methods:
        selected = set(args.methods.split(','))
        normalized = set()
        for s in selected:
            if s.startswith('second-order+'):
                s = s.replace('second-order+', 'second-order-hvp+', 1)
            normalized.add(s)
        selected = normalized
        methods_to_run = [m for m in methods_to_run if f"{m['importance']}+{m['allocation']}" in selected]
    
    # 4. 加载参考权重（ExCP 残差剪枝用）
    reference_weights = None
    need_residual = any(m['importance'] == 'residual-magnitude' for m in methods_to_run)
    if args.reference_checkpoint:
        ref_model, _ = load_model(
            args.model,
            pretrained=True,
            checkpoint_path=args.reference_checkpoint,
            device='cpu',
            dataset_name=args.dataset,
        )
        reference_weights = {name: p.detach().cpu() for name, p in ref_model.named_parameters()}
        del ref_model
    elif args.checkpoint and need_residual:
        ref_model, _ = load_model(
            args.model,
            pretrained=True,
            device='cpu',
            dataset_name=args.dataset,
        )
        reference_weights = {name: p.detach().cpu() for name, p in ref_model.named_parameters()}
        del ref_model
    elif need_residual:
        print("  [WARN] 无 --checkpoint 和 --reference_checkpoint，residual-magnitude 残差为 0，已跳过")
        methods_to_run = [m for m in methods_to_run if m['importance'] != 'residual-magnitude']
    
    # 5. 评估 baseline
    print("[4/6] 评估 baseline（未剪枝）...")
    model_eval = copy.deepcopy(model).to(args.device)
    baseline_metrics = evaluate(model_eval, cached_eval, task_type, args.device)
    print(f"  Baseline: {baseline_metrics}")
    del model_eval
    
    # 6. 预计算所有方法的重要性得分
    print("[5/6] 计算重要性得分并运行实验...")

    needed_importances = sorted({
        m['importance'] for m in methods_to_run if m['importance'] != 'per-type-auto'
    })
    # 2D / protected 模式需要 magnitude 得分
    if args.scoring_mode in ('2d', 'protected') and 'magnitude' not in needed_importances:
        needed_importances.append('magnitude')
        needed_importances.sort()
    # per-type-auto 使用搜索结果中的最优 metric
    if per_type_allocator is not None:
        per_type_imp = per_type_allocator.get_importance_method()
        if per_type_imp not in needed_importances:
            needed_importances.append(per_type_imp)
            needed_importances.sort()

    # 非 HVP 二阶方法的 importance 列表（magnitude / first-order / residual-magnitude）
    basic_imps = [m for m in needed_importances if m != 'second-order-hvp']
    need_hvp_second_order = 'second-order-hvp' in needed_importances

    # 先计算基础方法得分（magnitude / first-order / residual-magnitude）
    all_scores = {}
    if basic_imps:
        model_for_scoring = copy.deepcopy(model).to(args.device)
        basic_scores = compute_scores_by_method(
            model_for_scoring,
            cached_train,
            task_type,
            methods=basic_imps,
            alpha=alpha_values[0],  # 基础方法不使用 alpha
            hvp_batches=args.hvp_batches,
            hvp_mode=args.hvp_mode,
            chunk_size=args.chunk_size,
            reference_weights=reference_weights if 'residual-magnitude' in basic_imps else None,
            grad_batches_first_order=args.grad_batches_first_order,
            model_family=model_family,
            normalize=args.normalize,
        )
        all_scores.update(basic_scores)
        del model_for_scoring
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 对每个 alpha 值计算 HVP 二阶得分
    alpha_scores = {}  # {alpha_val: {param_name: score_tensor}}
    if need_hvp_second_order:
        for alpha_val in alpha_values:
            print(f"\n  [二阶 HVP] 计算 alpha={alpha_val} 的得分...")
            model_for_scoring = copy.deepcopy(model).to(args.device)
            so_scores = compute_scores_by_method(
                model_for_scoring,
                cached_train,
                task_type,
                methods=['second-order-hvp'],
                alpha=alpha_val,
                hvp_batches=args.hvp_batches,
                hvp_mode=args.hvp_mode,
                chunk_size=args.chunk_size,
                model_family=model_family,
                normalize=args.normalize,
            )
            alpha_scores[alpha_val] = so_scores['second-order-hvp']
            del model_for_scoring
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 8. 运行实验
    all_results = []
    
    def model_factory():
        return copy.deepcopy(model)
    
    for method in methods_to_run:
        imp = method['importance']
        alloc = method['allocation']

        if imp == 'per-type-auto' and per_type_allocator is not None:
            actual_imp = per_type_allocator.get_importance_method()
            print(f"\n--- per-type (metric={per_type_allocator.best_metric}, "
                  f"importance={actual_imp}) ---")
            scores_for_method = all_scores[actual_imp]
            method_display = {'importance': actual_imp, 'allocation': 'per-type'}
            results = _run_per_type_method(
                model_factory, scores_for_method, per_type_allocator,
                method_display, prune_ratios, cached_eval, task_type,
                args.device, baseline_metrics)
            all_results.extend(results)
        elif imp == 'second-order-hvp':
            # 对每个 alpha 值分别运行（HVP 版二阶）
            for alpha_val in alpha_values:
                alpha_suffix = f"(α={alpha_val})" if len(alpha_values) > 1 else ""
                print(f"\n--- {imp} + {alloc} {alpha_suffix} ---")
                scores_for_alpha = alpha_scores[alpha_val]
                method_with_alpha = dict(method)
                if len(alpha_values) > 1:
                    method_with_alpha['_method_name_override'] = f"{imp}+{alloc}(α={alpha_val})"

                # 2D / protection 模式：对 second-order-hvp 应用组合得分
                pruning_scores = scores_for_alpha  # 默认：1d 模式
                allocation_scores_override = None    # 默认：allocation 和 pruning 用同一得分
                mode_tag = ""

                if args.scoring_mode == '2d' and 'magnitude' in all_scores:
                    mag_scores = all_scores['magnitude']
                    pruning_scores = combine_scores_2d_with_protection(
                        mag_scores, scores_for_alpha,
                        protection_ratio=args.protection_ratio)
                    allocation_scores_override = scores_for_alpha  # 层间分配用 damage score
                    mode_tag = f"[2D prot={args.protection_ratio}] "
                    print(f"  {mode_tag}组合 magnitude + damage 得分 (protection={args.protection_ratio:.2%})")
                elif args.scoring_mode == 'protected' and 'magnitude' in all_scores:
                    weights_cpu = {n: p.detach().cpu() for n, p in model.named_parameters()}
                    from dacp.pruning import filter_prunable_params
                    prunable_w = filter_prunable_params(weights_cpu)
                    pruning_scores = apply_magnitude_protection(
                        scores_for_alpha, prunable_w,
                        protection_ratio=args.protection_ratio)
                    mode_tag = f"[protected prot={args.protection_ratio}] "
                    print(f"  {mode_tag}magnitude protection (protection={args.protection_ratio:.2%})")

                results = run_single_method(
                    model_factory, pruning_scores, method_with_alpha, prune_ratios,
                    cached_eval, task_type, args.device, baseline_metrics,
                    allocation_scores=allocation_scores_override)
                for r in results:
                    r['alpha'] = alpha_val
                    r['scoring_mode'] = args.scoring_mode
                    r['protection_ratio'] = args.protection_ratio
                    if len(alpha_values) > 1:
                        r['method'] = f"{imp}+{alloc}(α={alpha_val})"
                    if args.scoring_mode != '1d':
                        r['method'] = mode_tag + r['method']
                all_results.extend(results)
        else:
            print(f"\n--- {imp} + {alloc} ---")
            results = run_single_method(
                model_factory, all_scores[imp], method, prune_ratios,
                cached_eval, task_type, args.device, baseline_metrics)
            all_results.extend(results)
    
    # 9. 保存结果
    print("\n" + "=" * 70)
    config = vars(args)
    config['baseline_metrics'] = {k: float(v) for k, v in baseline_metrics.items()}
    save_results(all_results, args.output_dir, f"table1_{args.model}_{args.dataset}", config)
    
    print("\n结果汇总:")
    metric_key = 'perplexity' if task_type == 'lm' else ('pearson' if task_type == 'reg' else 'accuracy')
    print_results_table(all_results, ['method', 'target_ratio', 'actual_ratio', 'loss', 
                                       metric_key])
    print("\n实验完成！")


if __name__ == '__main__':
    main()
