"""验证 Block-wise HVP 近似精度的实验脚本。

核心实验：
1. 对同一 checkpoint，分别计算 global HVP 和 block-wise HVP 的 damage score
2. 度量两者的相关性（Spearman/Pearson）、relative L2 error、pruning mask IoU
3. 可变 seq_length 来改变 γ = T/(12d)，画出 γ → error 趋势

理论预期（Lemma 1 秩约束 rank(H_{ℓℓ'}) ≤ dT）：
  - γ 越小，block-wise 近似越好
  - 模型越大（d 越大），γ 越小

用法:
    # 单一 seq_length
    python -m experiments.scripts.run_blockwise_hvp_analysis \
        --model gpt2-medium --dataset wikitext2 --seq_length 512

    # 扫描多个 seq_length（变 γ）
    python -m experiments.scripts.run_blockwise_hvp_analysis \
        --model gpt2-medium --dataset wikitext2 \
        --seq_lengths 128,256,512,1024

    # 多模型对比
    python -m experiments.scripts.run_blockwise_hvp_analysis \
        --model gpt2-small --dataset wikitext2 --seq_length 512
"""

import copy
import gc
import json
import os
import sys
import time
from datetime import datetime

import torch

# 添加项目根路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.lib.models import load_model
from experiments.lib.data import get_data_loaders, cache_batches
from experiments.lib.args import create_base_parser, add_scoring_args
from experiments.lib.losses import make_task_loss
from experiments.lib.importance_compare.score_comparison import (
    compute_relative_l2_error,
    compute_rank_correlation,
    compute_mask_iou,
    compute_gamma,
    summarize_comparison,
)
from dacp.pruning.pruner import filter_prunable_params


def parse_args():
    parser = create_base_parser('Block-wise HVP approximation analysis')
    add_scoring_args(parser)
    # 本脚本特有参数
    parser.add_argument('--seq_lengths', type=str, default=None,
                        help='逗号分隔 seq_lengths，用于扫描 γ（覆盖 --seq_length）')
    parser.add_argument('--prune_ratios', type=str, default='0.2,0.4',
                        help='mask IoU 的剪枝率')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='对一阶/二阶项做 per-tensor 尺度归一化后再合并')
    parser.add_argument('--abs_combine', action='store_true', default=False,
                        help='s=|first|+α|second| 独立取绝对值（避免符号对消）')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='先做 N 步训练让梯度变大（模拟训练中途 checkpoint）')
    parser.add_argument('--warmup_lr', type=float, default=1e-2,
                        help='warmup 训练的学习率（默认 1e-2）')
    parser.add_argument('--alpha_sweep', type=str, default=None,
                        help='逗号分隔的 alpha 值，用于 α sweep（覆盖 --alpha）')
    # 覆盖默认值
    parser.set_defaults(output_dir='experiments/results/blockwise_hvp_analysis',
                        hvp_batches=2)
    return parser.parse_args()


def _make_loss_fn(task_type):
    """构造 blockwise 分析使用的损失函数，保留 cls 的历史 LM 适配。"""
    if task_type == 'cls':
        task_type = 'lm'
    if task_type not in ('lm', 'cv'):
        raise ValueError(f"Unknown task_type for blockwise analysis: {task_type}")
    return make_task_loss(task_type)


def _compute_shared_gradient(model, loss_fn, gpu_batches, num_batches):
    """计算跨 batch 平均梯度，与 HVP 模式无关。"""
    from collections import defaultdict
    accumulated = defaultdict(lambda: 0)
    actual = min(num_batches, len(gpu_batches))

    for i in range(actual):
        model.zero_grad()
        loss = loss_fn(model, gpu_batches[i])
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                accumulated[name] = accumulated[name] + p.grad.detach().clone()

    return {name: g / actual for name, g in accumulated.items()}


def _compute_full_hvp(model, loss_fn, gpu_batches, prunable_names, num_batches):
    """全局 HVP: u = Hθ（模型级 Hessian）。"""
    from dacp.tools.importance import compute_hvp_batched

    vector = {}
    for name, p in model.named_parameters():
        if name in prunable_names:
            vector[name] = p.data.clone()
        else:
            vector[name] = torch.zeros_like(p.data)

    return compute_hvp_batched(model, loss_fn, gpu_batches, vector, num_batches)


def _compute_block_hvp(model, loss_fn, gpu_batches, model_family, num_batches):
    """Block-wise HVP: u_b = H_b θ_b（block-diagonal Hessian）。"""
    from dacp.tools.importance import (
        build_transformer_blocks, compute_hvp_blockwise_batched,
    )

    blocks = build_transformer_blocks(model, model_family)
    return compute_hvp_blockwise_batched(model, loss_fn, gpu_batches, blocks, num_batches)


def _scores_from_grad_hvp(gradients, hvp_result, prunable_w, alpha,
                          normalize=False, abs_combine=False):
    """从共享梯度 + HVP 计算 damage score 和纯二阶项。

    三种模式:
      默认:       s = |first + α·second|            (论文公式，有符号对消)
      normalize:  s = |first/μ1 + α·second/μ2|      (等权归一化)
      abs_combine: s = |first| + α·|second|          (独立取绝对值，无对消)

    abs_combine 数学动机: 使用各项的绝对贡献度 (|ΔL_1st| + |ΔL_2nd|)
    代替有符号组合，避免正负项部分抵消造成的排序不稳定。
    """
    scores = {}
    second_order_only = {}
    _EPS = 1e-12
    for name in prunable_w:
        theta = prunable_w[name]
        grad = gradients.get(name, torch.zeros_like(theta))
        hvp = hvp_result.get(name, torch.zeros_like(theta))
        if hvp.is_cuda:
            hvp = hvp.cpu()
        if grad.is_cuda:
            grad = grad.cpu()

        first_order = -grad * theta
        second_term = alpha * theta * hvp

        if abs_combine:
            scores[name] = first_order.abs() + alpha * (theta * hvp).abs()
        elif normalize:
            mu1 = first_order.abs().mean() + _EPS
            mu2 = second_term.abs().mean() + _EPS
            scores[name] = torch.abs(first_order / mu1 + second_term / mu2)
        else:
            scores[name] = torch.abs(first_order + second_term)
        second_order_only[name] = torch.abs(second_term)

    return scores, second_order_only


def run_single_comparison(model, model_name, model_family, dataset_name,
                          seq_length, batch_size, alpha, hvp_batches,
                          prune_ratios, device, normalize=False,
                          abs_combine=False):
    """对一个 (model, seq_length) 组合运行 global vs block-wise 对比。

    关键设计：梯度仅计算一次（共享），两种模式仅 HVP 部分不同，
    从而精确隔离 block-diagonal 近似造成的差异。
    """

    gamma_info = compute_gamma(model_name, seq_length)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}, seq_length={seq_length}, γ={gamma_info['gamma_pct']:.2f}%")
    print(f"{'='*60}")

    # 加载数据
    print(f"[Data] 加载数据 seq_length={seq_length}...")
    train_loader, _, task_type = get_data_loaders(
        model_name, dataset_name, batch_size, seq_length)
    cached_train = cache_batches(train_loader, hvp_batches + 2, task_type)

    loss_fn = _make_loss_fn(task_type)
    gpu_batches = [{k: v.to(device) for k, v in b.items()}
                   for b in cached_train[:hvp_batches]]

    # 收集 prunable 参数名集合（直接引用 model 参数，不做冗余拷贝）
    # model 始终在 CPU 上，deepcopy 才上 GPU，原始参数数据稳定
    prunable_w = filter_prunable_params(
        {n: p.data for n, p in model.named_parameters()})

    # ---- 共享：梯度计算 ----
    print("[Shared] 计算共享梯度...")
    model_work = copy.deepcopy(model).to(device)
    shared_grad = _compute_shared_gradient(
        model_work, loss_fn, gpu_batches, hvp_batches)
    # 转 CPU
    shared_grad = {k: v.cpu() for k, v in shared_grad.items()}
    del model_work
    torch.cuda.empty_cache(); gc.collect()

    # ---- Global HVP ----
    print("\n[1/2] 计算 Global HVP...")
    model_g = copy.deepcopy(model).to(device)
    t0 = time.time()
    hvp_global = _compute_full_hvp(
        model_g, loss_fn, gpu_batches, set(prunable_w), hvp_batches)
    time_global = time.time() - t0
    scores_global, so_global = _scores_from_grad_hvp(
        shared_grad, hvp_global, prunable_w, alpha, normalize, abs_combine)
    print(f"  Global HVP time: {time_global:.1f}s")
    del model_g, hvp_global
    torch.cuda.empty_cache(); gc.collect()

    # ---- Block-wise HVP ----
    print("\n[2/2] 计算 Block-wise HVP...")
    model_b = copy.deepcopy(model).to(device)
    t0 = time.time()
    hvp_block = _compute_block_hvp(
        model_b, loss_fn, gpu_batches, model_family, hvp_batches)
    time_block = time.time() - t0
    scores_block, so_block = _scores_from_grad_hvp(
        shared_grad, hvp_block, prunable_w, alpha, normalize, abs_combine)
    print(f"  Block-wise HVP time: {time_block:.1f}s")
    del model_b, hvp_block
    torch.cuda.empty_cache(); gc.collect()

    # ---- 对齐 key 集合 ----
    common_keys = sorted(set(scores_global) & set(scores_block))
    sg = {k: scores_global[k] for k in common_keys}
    sb = {k: scores_block[k] for k in common_keys}
    sog = {k: so_global[k] for k in common_keys}
    sob = {k: so_block[k] for k in common_keys}

    total_params = sum(s.numel() for s in sg.values())
    print(f"\n[Compare] {len(common_keys)} layers, {total_params:,} params")

    # ---- 完整 score 度量 ----
    print("[Compare] L2 error (full)...", flush=True)
    l2_errors = compute_relative_l2_error(sg, sb)
    print("[Compare] Rank correlation (full)...", flush=True)
    correlations = compute_rank_correlation(sg, sb)
    print("[Compare] Mask IoU...", flush=True)
    mask_ious = {}
    for ratio in prune_ratios:
        mask_ious[f"{ratio:.0%}"] = compute_mask_iou(sg, sb, ratio)

    # ---- 纯二阶项度量 ----
    print("[Compare] L2 error (second-order)...", flush=True)
    l2_errors_so = compute_relative_l2_error(sog, sob)
    print("[Compare] Rank correlation (second-order)...", flush=True)
    correlations_so = compute_rank_correlation(sog, sob)

    # Summary
    summary = summarize_comparison(l2_errors, correlations, mask_ious, gamma_info)
    print(f"\n{summary}")
    print(f"\n--- Second-Order Only ---")
    print(f"Relative L2 Error:  {l2_errors_so.get('__global__', 0):.6f}")
    print(f"Spearman ρ:         {correlations_so.get('__global__', {}).get('spearman', 0):.6f}")
    print(f"Pearson r:          {correlations_so.get('__global__', {}).get('pearson', 0):.6f}")

    result = {
        'model': model_name,
        'seq_length': seq_length,
        'gamma': gamma_info,
        'time_global_s': time_global,
        'time_block_s': time_block,
        # 完整 score 度量
        'global_l2_error': l2_errors.get('__global__', 0),
        'global_spearman': correlations.get('__global__', {}).get('spearman', 0),
        'global_pearson': correlations.get('__global__', {}).get('pearson', 0),
        'mask_ious': mask_ious,
        # 纯二阶项度量
        'second_order_l2_error': l2_errors_so.get('__global__', 0),
        'second_order_spearman': correlations_so.get('__global__', {}).get('spearman', 0),
        'second_order_pearson': correlations_so.get('__global__', {}).get('pearson', 0),
        # 逐层
        'per_layer_l2_error': {k: v for k, v in l2_errors.items() if k != '__global__'},
        'per_layer_spearman': {
            k: v.get('spearman', 0) for k, v in correlations.items() if k != '__global__'
        },
    }

    return result


def main():
    args = parse_args()

    # 解析 seq_lengths
    if args.seq_lengths:
        seq_lengths = [int(s) for s in args.seq_lengths.split(',')]
    else:
        seq_lengths = [args.seq_length]

    prune_ratios = [float(r) for r in args.prune_ratios.split(',')]

    # 解析 alpha_sweep
    if args.alpha_sweep:
        alphas = [float(a) for a in args.alpha_sweep.split(',')]
    else:
        alphas = [args.alpha]

    # 加载模型（仅一次，放 CPU）
    print(f"[Init] 加载模型 {args.model} (CPU)...")
    model, model_family = load_model(
        args.model, pretrained=True,
        checkpoint_path=args.checkpoint, device='cpu')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable params: {total_params:,}")

    # Warmup: 做几步训练让梯度变大，模拟训练中途 checkpoint
    if args.warmup_steps > 0:
        print(f"  [Warmup] 做 {args.warmup_steps} 步训练...")
        _warmup_model = model.to(args.device)
        _wt_loader, _, _wt_type = get_data_loaders(
            args.model, args.dataset, args.batch_size, args.seq_length)
        _wt_batches = cache_batches(_wt_loader, args.warmup_steps + 1, _wt_type)
        _wt_loss_fn = _make_loss_fn(_wt_type)
        _opt = torch.optim.SGD(_warmup_model.parameters(), lr=args.warmup_lr)
        for step_i in range(min(args.warmup_steps, len(_wt_batches))):
            _opt.zero_grad()
            _b = {k: v.to(args.device) for k, v in _wt_batches[step_i].items()}
            _loss = _wt_loss_fn(_warmup_model, _b)
            _loss.backward()
            _opt.step()
            if (step_i + 1) % 10 == 0:
                print(f"    step {step_i+1}/{args.warmup_steps}, loss={_loss.item():.4f}")
        model = _warmup_model.cpu()
        del _wt_batches, _opt
        torch.cuda.empty_cache(); gc.collect()
        print(f"  [Warmup] 完成")

    if args.normalize:
        print(f"  [Normalize] per-tensor 尺度归一化 已启用")
    if args.abs_combine:
        print(f"  [AbsCombine] s=|first|+α|second| 模式已启用")
    if len(alphas) > 1:
        print(f"  [α sweep] alphas = {alphas}")

    # 增量保存：每组 (seq_len, alpha) 完成后立即写盘，防止 OOM 丢数据
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(
        args.output_dir,
        f'{timestamp}_{args.model}_blockwise_analysis.json')

    all_results = []
    for seq_len in seq_lengths:
        for alpha in alphas:
            result = run_single_comparison(
                model=model,
                model_name=args.model,
                model_family=model_family,
                dataset_name=args.dataset,
                seq_length=seq_len,
                batch_size=args.batch_size,
                alpha=alpha,
                hvp_batches=args.hvp_batches,
                prune_ratios=prune_ratios,
                device=args.device,
                normalize=args.normalize,
                abs_combine=args.abs_combine,
            )
            result['alpha'] = alpha
            result['normalize'] = args.normalize
            result['abs_combine'] = args.abs_combine
            all_results.append(result)

            # 每组完成后立即保存
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  [已保存] {len(all_results)} 组结果 → {result_file}")

    print(f"\n[Done] 全部 {len(all_results)} 组结果已保存至 {result_file}")

    # 如果有多个 seq_length 或 alpha，输出趋势表
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("Trend Table (Full Score / Second-Order Only)")
        print(f"{'='*70}")
        print(f"{'seq_len':>8} {'alpha':>6} {'γ%':>8} {'L2_err':>10} {'Spearman':>10} "
              f"{'SO_L2':>10} {'SO_Spear':>10}")
        print(f"{'-'*70}")
        for r in all_results:
            print(f"{r['seq_length']:>8} "
                  f"{r.get('alpha', args.alpha):>6.2f} "
                  f"{r['gamma']['gamma_pct']:>7.2f}% "
                  f"{r['global_l2_error']:>10.6f} "
                  f"{r['global_spearman']:>10.6f} "
                  f"{r['second_order_l2_error']:>10.6f} "
                  f"{r['second_order_spearman']:>10.6f}")


if __name__ == '__main__':
    main()
