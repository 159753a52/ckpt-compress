"""验证经验性 block-diagonal HVP 近似的实验脚本。

核心实验：
1. 对同一 checkpoint，分别计算 global HVP 和 block-wise HVP 的 damage score
2. 度量两者的相关性（Spearman/Pearson）、relative L2 error、pruning mask IoU
3. 可变 seq_length 来改变 γ = T/(12d)，画出 γ → error 趋势

本实验只做经验性 block-diagonal 近似验证，不把秩关系当作已经成立的定理：
  - 记录不同 seq_length 下的实际误差、相关性和运行开销
  - 运行时间与显存峰值以当前设备上的实测结果为准

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

import argparse
import copy
import gc
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

# 添加项目根路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dacp.pruning.pruner import filter_prunable_params
from experiments.lib.args import add_scoring_args, create_base_parser, nonnegative_int
from experiments.lib.data import cache_batches, get_data_loaders, get_task_type
from experiments.lib.importance_compare.score_comparison import (
    compute_gamma,
    compute_mask_iou,
    compute_rank_correlation,
    compute_relative_l2_error,
    summarize_comparison,
)
from experiments.lib.losses import make_task_loss
from experiments.lib.models import get_model_type, load_model
from experiments.lib.residual_runtime import (
    batch_hash,
    peak_memory_bytes,
    reset_peak_memory,
    set_seed,
    synchronize_device,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
MEASUREMENT_ORDER = ("global", "block")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_csv(raw: str, label: str, converter: Callable[[str], Any]) -> list[Any]:
    if not isinstance(raw, str):
        raise ValueError(f"{label} must be a comma-separated string")
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"{label} must be a non-empty comma-separated list")
    try:
        values = [converter(part) for part in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains an invalid value: {raw!r}") from exc
    return values


def _validate_positive_ints(values: Sequence[Any], label: str) -> list[int]:
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must contain positive integers, got {values!r}")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must not contain duplicates: {normalized!r}")
    return normalized


def _validate_unit_interval(values: Sequence[Any], label: str) -> list[float]:
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must contain finite values in [0, 1], got {values!r}")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must contain finite values in [0, 1], got {values!r}")
        normalized.append(value)
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return normalized


def parse_seq_lengths(args) -> list[int]:
    raw = getattr(args, "seq_lengths", None)
    if raw is None:
        values = [getattr(args, "seq_length")]
    else:
        values = _parse_csv(raw, "seq_lengths", int)
    return _validate_positive_ints(values, "seq_lengths")


def parse_alphas(args) -> list[float]:
    raw = getattr(args, "alpha_sweep", None)
    values = [getattr(args, "alpha")] if raw is None else _parse_csv(raw, "alpha_sweep", float)
    return _validate_unit_interval(values, "alpha values")


def parse_prune_ratios(args) -> list[float]:
    raw = getattr(args, "prune_ratios", None)
    if raw is None:
        raise ValueError("prune_ratios is required")
    if isinstance(raw, str):
        values = _parse_csv(raw, "prune_ratios", float)
    else:
        values = list(raw)
    return _validate_unit_interval(values, "prune_ratios")


def _result_path(args) -> Path:
    output_file = getattr(args, "output_file", None)
    if output_file:
        path = Path(output_file)
    else:
        path = Path(args.output_dir) / f"{args.model}_seed{args.seed}_blockwise_analysis.json"
    if path.suffix.lower() != ".json":
        raise ValueError(f"Result output must be a .json file: {path}")
    required_tokens = (str(args.model), f"seed{args.seed}")
    if any(token not in path.name for token in required_tokens):
        raise ValueError(
            "Result filename must contain both the model and seed; "
            "use a different --output-dir or an explicit --output-file with those tokens"
        )
    return path


def validate_args(args, *, check_output: bool = True) -> dict[str, object]:
    """Validate the dry-run-safe experiment configuration and return its output plan."""
    if not isinstance(args.model, str) or not args.model:
        raise ValueError("model must be a non-empty string")
    if not isinstance(args.dataset, str) or not args.dataset:
        raise ValueError("dataset must be a non-empty string")
    get_model_type(args.model)
    get_task_type(args.dataset)

    seq_lengths = parse_seq_lengths(args)
    alphas = parse_alphas(args)
    prune_ratios = parse_prune_ratios(args)

    seed = getattr(args, "seed", None)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    for name in ("batch_size", "hvp_batches"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value}")
    warmup_steps = getattr(args, "warmup_steps", 0)
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError(f"warmup_steps must be a non-negative integer, got {warmup_steps}")
    warmup_lr = float(getattr(args, "warmup_lr", 1e-2))
    if not math.isfinite(warmup_lr) or warmup_lr <= 0:
        raise ValueError(f"warmup_lr must be a positive finite number, got {warmup_lr}")

    output_file = _result_path(args)
    exists = output_file.exists()
    if check_output and exists:
        raise FileExistsError(
            f"Refusing to overwrite existing result file {output_file}; "
            "choose a different --output-dir or an explicit --output-file"
        )
    return {
        "model": args.model,
        "dataset": args.dataset,
        "seq_lengths": seq_lengths,
        "alphas": alphas,
        "prune_ratios": prune_ratios,
        "seed": seed,
        "output_file": str(output_file),
        "output_exists": exists,
    }


def parse_args(argv=None):
    parser = create_base_parser("Block-wise HVP approximation analysis")
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=argparse.SUPPRESS,
        help="--output_dir 的连字符别名",
    )
    add_scoring_args(parser)
    parser.add_argument(
        "--seq_lengths",
        type=str,
        default=None,
        help="逗号分隔 seq_lengths，用于扫描经验误差（覆盖 --seq_length）",
    )
    parser.add_argument("--prune_ratios", type=str, default="0.2,0.4", help="mask IoU 的剪枝率")
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="对一阶/二阶项做 per-tensor 尺度归一化后再合并",
    )
    parser.add_argument(
        "--abs_combine",
        action="store_true",
        default=False,
        help="s=|first|+α|second| 独立取绝对值（避免符号对消）",
    )
    parser.add_argument(
        "--warmup_steps",
        type=nonnegative_int,
        default=0,
        help="先做 N 步训练让梯度变大（模拟训练中途 checkpoint）",
    )
    parser.add_argument(
        "--warmup_lr", type=float, default=1e-2, help="warmup 训练的学习率（默认 1e-2）"
    )
    parser.add_argument(
        "--alpha_sweep",
        type=str,
        default=None,
        help="逗号分隔的 alpha 值，用于 α sweep（覆盖 --alpha）",
    )
    parser.add_argument("--seed", type=nonnegative_int, default=42, help="实验随机种子")
    parser.add_argument(
        "--output_file",
        "--output-file",
        dest="output_file",
        default=None,
        help="显式结果文件路径；文件名仍必须包含 model 和 seed",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="只校验配置和输出计划，不加载模型、数据或 CUDA",
    )
    parser.set_defaults(output_dir="experiments/results/blockwise_hvp_analysis", hvp_batches=2)
    return parser.parse_args(argv)


def _source_git_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return {
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
    }


def _resolve_device_name(device: str) -> str:
    if not device.startswith("cuda"):
        return device
    return str(torch.cuda.get_device_name(torch.device(device)))


def _model_dtype(model) -> str | list[str]:
    dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if not dtypes:
        return "unknown"
    return dtypes[0] if len(dtypes) == 1 else dtypes


def _cleanup_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()


def _measure_hvp(
    device: str,
    hvp_fn: Callable[[], Mapping[str, torch.Tensor]],
) -> tuple[Mapping[str, torch.Tensor], dict[str, float | int]]:
    """Measure one HVP path after synchronizing and resetting device statistics."""
    synchronize_device(device)
    reset_peak_memory(device)
    started = time.perf_counter()
    hvp_result = hvp_fn()
    synchronize_device(device)
    measured_peak = peak_memory_bytes(device)
    measurement = {
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": measured_peak if device.startswith("cuda") else 0,
    }
    return hvp_result, measurement


def _make_loss_fn(task_type):
    """Construct the task-appropriate loss used by the comparison."""
    return make_task_loss(task_type)


def _compute_shared_gradient(model, loss_fn, gpu_batches, num_batches):
    """计算跨 batch 平均梯度，与 HVP 模式无关。"""
    if isinstance(num_batches, bool) or not isinstance(num_batches, int) or num_batches < 1:
        raise ValueError(f"num_batches must be a positive integer, got {num_batches}")
    if not gpu_batches:
        raise ValueError("gpu_batches must contain at least one batch")
    accumulated: dict[str, torch.Tensor] = {}
    actual = min(num_batches, len(gpu_batches))

    for i in range(actual):
        model.zero_grad()
        loss = loss_fn(model, gpu_batches[i])
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                gradient = p.grad.detach().clone()
                if name in accumulated:
                    accumulated[name].add_(gradient)
                else:
                    accumulated[name] = gradient

    return {name: g / actual for name, g in accumulated.items()}


def _compute_full_hvp(model, loss_fn, gpu_batches, prunable_names, num_batches):
    """全局 HVP: u = Hθ（模型级 Hessian）。"""
    from dacp.tools.importance import compute_hvp_batched

    vector = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in prunable_names:
            vector[name] = p.data.clone()
        else:
            vector[name] = torch.zeros_like(p.data)

    return compute_hvp_batched(model, loss_fn, gpu_batches, vector, num_batches)


def _compute_block_hvp(model, loss_fn, gpu_batches, model_family, prunable_names, num_batches):
    """Block-wise HVP: u_b = H_b θ_b（block-diagonal Hessian）。"""
    from dacp.tools.importance import (
        build_transformer_blocks,
        complete_blockwise_vector,
        compute_hvp_blockwise_batched,
        include_parameter_blocks,
    )

    blocks = include_parameter_blocks(
        build_transformer_blocks(model, model_family),
        prunable_names,
    )
    named_params = dict(model.named_parameters())
    vector = complete_blockwise_vector(
        model,
        blocks,
        {name: named_params[name].data.clone() for name in prunable_names},
    )
    return compute_hvp_blockwise_batched(
        model, loss_fn, gpu_batches, blocks, num_batches, vector=vector
    )


def _scores_from_grad_hvp(
    gradients, hvp_result, prunable_w, alpha, normalize=False, abs_combine=False
):
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
    for label, values in (("Gradient", gradients), ("HVP", hvp_result)):
        missing = sorted(set(prunable_w).difference(values))
        if missing:
            raise RuntimeError(f"{label} coverage is missing prunable parameters: {missing}")
    for name in prunable_w:
        theta = prunable_w[name]
        grad = gradients[name]
        hvp = hvp_result[name]
        if hvp.is_cuda:
            hvp = hvp.cpu()
        if grad.is_cuda:
            grad = grad.cpu()
        for label, value in (("Gradient", grad), ("HVP", hvp)):
            if value.shape != theta.shape:
                raise ValueError(
                    f"{label} shape for {name!r} must match the parameter: "
                    f"{tuple(value.shape)} != {tuple(theta.shape)}"
                )
            if not value.is_floating_point() or not torch.isfinite(value).all().item():
                raise ValueError(f"{label} for {name!r} must be finite floating point")

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


def run_single_comparison(
    model,
    model_name,
    model_family,
    dataset_name,
    seq_length,
    batch_size,
    alpha,
    hvp_batches,
    prune_ratios,
    device,
    normalize=False,
    abs_combine=False,
    seed=42,
    checkpoint=None,
    created_at=None,
    device_name=None,
    dtype=None,
    source_git=None,
):
    """对一个 (model, seq_length) 组合运行经验性 global vs block-wise 对比。

    关键设计：梯度仅计算一次（共享），两种模式仅 HVP 部分不同，
    从而精确隔离 block-diagonal 近似造成的差异。
    """

    set_seed(seed)
    gamma_info = compute_gamma(model_name, seq_length)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}, seq_length={seq_length}, γ={gamma_info['gamma_pct']:.2f}%")
    print(f"{'='*60}")

    # 加载数据
    print(f"[Data] 加载数据 seq_length={seq_length}...")
    train_loader, _, task_type = get_data_loaders(model_name, dataset_name, batch_size, seq_length)
    cached_train = cache_batches(train_loader, hvp_batches + 2, task_type)
    if not cached_train:
        raise RuntimeError("The data loader returned no batches for HVP analysis")
    cached_hvp = cached_train[:hvp_batches]
    cached_batch_hash = batch_hash(cached_hvp)

    loss_fn = _make_loss_fn(task_type)
    gpu_batches = [{k: v.to(device) for k, v in b.items()} for b in cached_hvp]

    # 收集 prunable 参数名集合（直接引用 model 参数，不做冗余拷贝）
    # model 始终在 CPU 上，deepcopy 才上 GPU，原始参数数据稳定
    prunable_w = filter_prunable_params({n: p.data for n, p in model.named_parameters()})

    # ---- 共享：梯度计算 ----
    print("[Shared] 计算共享梯度...")
    model_work = copy.deepcopy(model).to(device)
    shared_grad = _compute_shared_gradient(model_work, loss_fn, gpu_batches, hvp_batches)
    # 转 CPU
    shared_grad = {k: v.cpu() for k, v in shared_grad.items()}
    del model_work
    _cleanup_device(device)

    # ---- Global HVP ----
    print("\n[1/2] 计算 Global HVP...")
    set_seed(seed)
    model_g = copy.deepcopy(model).to(device)
    hvp_global, global_measurement = _measure_hvp(
        device,
        lambda: _compute_full_hvp(model_g, loss_fn, gpu_batches, set(prunable_w), hvp_batches),
    )
    scores_global, so_global = _scores_from_grad_hvp(
        shared_grad, hvp_global, prunable_w, alpha, normalize, abs_combine
    )
    print(f"  Global HVP time: {global_measurement['wall_seconds']:.1f}s")
    del model_g, hvp_global
    _cleanup_device(device)

    # ---- Block-wise HVP ----
    print("\n[2/2] 计算 Block-wise HVP...")
    set_seed(seed)
    model_b = copy.deepcopy(model).to(device)
    hvp_block, block_measurement = _measure_hvp(
        device,
        lambda: _compute_block_hvp(
            model_b, loss_fn, gpu_batches, model_family, set(prunable_w), hvp_batches
        ),
    )
    scores_block, so_block = _scores_from_grad_hvp(
        shared_grad, hvp_block, prunable_w, alpha, normalize, abs_combine
    )
    print(f"  Block-wise HVP time: {block_measurement['wall_seconds']:.1f}s")
    del model_b, hvp_block
    _cleanup_device(device)

    total_params = sum(score.numel() for score in scores_global.values())
    print(f"\n[Compare] {len(scores_global)} layers, {total_params:,} params")

    # ---- 完整 score 度量 ----
    print("[Compare] L2 error (full)...", flush=True)
    l2_errors = compute_relative_l2_error(scores_global, scores_block)
    print("[Compare] Rank correlation (full)...", flush=True)
    correlations = compute_rank_correlation(scores_global, scores_block)
    print("[Compare] Mask IoU...", flush=True)
    mask_ious = {}
    for ratio in prune_ratios:
        mask_ious[f"{ratio:.0%}"] = compute_mask_iou(scores_global, scores_block, ratio)

    # ---- 纯二阶项度量 ----
    print("[Compare] L2 error (second-order)...", flush=True)
    l2_errors_so = compute_relative_l2_error(so_global, so_block)
    print("[Compare] Rank correlation (second-order)...", flush=True)
    correlations_so = compute_rank_correlation(so_global, so_block)

    # Summary
    summary = summarize_comparison(l2_errors, correlations, mask_ious, gamma_info)
    print(f"\n{summary}")
    print(f"\n--- Second-Order Only ---")
    print(f"Relative L2 Error:  {l2_errors_so['__global__']:.6f}")
    print(f"Spearman ρ:         {correlations_so['__global__']['spearman']:.6f}")
    print(f"Pearson r:          {correlations_so['__global__']['pearson']:.6f}")

    resolved_device_name = device_name if device_name is not None else _resolve_device_name(device)
    resolved_dtype = dtype if dtype is not None else _model_dtype(model)
    resolved_source_git = source_git if source_git is not None else _source_git_state()
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at or _utc_timestamp(),
        "seed": seed,
        "model": model_name,
        "dataset": dataset_name,
        "checkpoint": checkpoint,
        "device": device,
        "device_name": resolved_device_name,
        "torch_version": torch.__version__,
        "dtype": resolved_dtype,
        "hvp_batches": hvp_batches,
        "batch_hash": cached_batch_hash,
        "source": resolved_source_git,
        "measurement_order": list(MEASUREMENT_ORDER),
        "seq_length": seq_length,
        "gamma": gamma_info,
        "paths": {
            "global": global_measurement,
            "block": block_measurement,
        },
        "time_global_s": global_measurement["wall_seconds"],
        "time_block_s": block_measurement["wall_seconds"],
        "peak_gpu_memory_bytes_global": global_measurement["peak_gpu_memory_bytes"],
        "peak_gpu_memory_bytes_block": block_measurement["peak_gpu_memory_bytes"],
        # 完整 score 度量
        "global_l2_error": l2_errors["__global__"],
        "global_spearman": correlations["__global__"]["spearman"],
        "global_pearson": correlations["__global__"]["pearson"],
        "mask_ious": mask_ious,
        # 纯二阶项度量
        "second_order_l2_error": l2_errors_so["__global__"],
        "second_order_spearman": correlations_so["__global__"]["spearman"],
        "second_order_pearson": correlations_so["__global__"]["pearson"],
        # 逐层
        "per_layer_l2_error": {k: v for k, v in l2_errors.items() if k != "__global__"},
        "per_layer_spearman": {
            key: value["spearman"] for key, value in correlations.items() if key != "__global__"
        },
    }

    return result


def main():
    args = parse_args()
    plan = validate_args(args)
    if args.dry_run:
        print("[Dry-run] 配置有效，未加载模型、数据或 CUDA")
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return

    seq_lengths = plan["seq_lengths"]
    alphas = plan["alphas"]
    prune_ratios = plan["prune_ratios"]
    result_file = Path(plan["output_file"])
    set_seed(args.seed)
    created_at = _utc_timestamp()
    source_git = _source_git_state()

    # 加载模型（仅一次，放 CPU）
    print(f"[Init] 加载模型 {args.model} (CPU)...")
    model, model_family = load_model(
        args.model, pretrained=True, checkpoint_path=args.checkpoint, device="cpu"
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable params: {total_params:,}")

    # Warmup: 做几步训练让梯度变大，模拟训练中途 checkpoint
    if args.warmup_steps > 0:
        set_seed(args.seed)
        print(f"  [Warmup] 做 {args.warmup_steps} 步训练...")
        _warmup_model = model.to(args.device)
        _wt_loader, _, _wt_type = get_data_loaders(
            args.model, args.dataset, args.batch_size, args.seq_length
        )
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
        _cleanup_device(args.device)
        print(f"  [Warmup] 完成")

    if args.normalize:
        print(f"  [Normalize] per-tensor 尺度归一化 已启用")
    if args.abs_combine:
        print(f"  [AbsCombine] s=|first|+α|second| 模式已启用")
    if len(alphas) > 1:
        print(f"  [α sweep] alphas = {alphas}")

    result_file.parent.mkdir(parents=True, exist_ok=True)
    resolved_device_name = _resolve_device_name(args.device)
    resolved_dtype = _model_dtype(model)

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
                seed=args.seed,
                checkpoint=args.checkpoint,
                created_at=created_at,
                device_name=resolved_device_name,
                dtype=resolved_dtype,
                source_git=source_git,
            )
            result["alpha"] = alpha
            result["normalize"] = args.normalize
            result["abs_combine"] = args.abs_combine
            all_results.append(result)

    if result_file.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result file {result_file}; "
            "choose a different --output-dir or an explicit --output-file"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "seed": args.seed,
        "model": args.model,
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "device_name": resolved_device_name,
        "torch_version": torch.__version__,
        "dtype": resolved_dtype,
        "hvp_batches": args.hvp_batches,
        "batch_hashes": [result["batch_hash"] for result in all_results],
        "source": source_git,
        "measurement_order": list(MEASUREMENT_ORDER),
        "results": all_results,
    }
    write_json(result_file, payload)
    print(f"  [已保存] {len(all_results)} 组结果 → {result_file}")

    print(f"\n[Done] 全部 {len(all_results)} 组结果已保存至 {result_file}")

    # 如果有多个 seq_length 或 alpha，输出趋势表
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("Trend Table (Full Score / Second-Order Only)")
        print(f"{'='*70}")
        print(
            f"{'seq_len':>8} {'alpha':>6} {'γ%':>8} {'L2_err':>10} {'Spearman':>10} "
            f"{'SO_L2':>10} {'SO_Spear':>10}"
        )
        print(f"{'-'*70}")
        for r in all_results:
            print(
                f"{r['seq_length']:>8} "
                f"{r.get('alpha', args.alpha):>6.2f} "
                f"{r['gamma']['gamma_pct']:>7.2f}% "
                f"{r['global_l2_error']:>10.6f} "
                f"{r['global_spearman']:>10.6f} "
                f"{r['second_order_l2_error']:>10.6f} "
                f"{r['second_order_spearman']:>10.6f}"
            )


if __name__ == "__main__":
    main()
