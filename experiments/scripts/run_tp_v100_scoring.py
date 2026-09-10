"""Executable benchmark and diagnostic for 2x V100 Tensor Parallelism scoring.

Run with:
    torchrun --nproc_per_node=2 experiments/scripts/run_tp_v100_scoring.py --device cuda
Or single-process simulation:
    python experiments/scripts/run_tp_v100_scoring.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.residual_tp import (
    TPConfig,
    evaluate_tp_candidates_streamed,
    shard_residual_scope_tp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2x V100 TP Candidate Scoring Benchmark")
    parser.add_argument("--model-dim", type=int, default=768, help="Hidden dimension (default: 768 for GPT-2)")
    parser.add_argument("--layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size")
    parser.add_argument("--prune-ratio", type=float, default=0.5, help="Pruning ratio (0, 1)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", type=str, default=None, help="Distributed backend (nccl/gloo)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "diagnostics" / "tp_v100_scoring",
    )
    return parser.parse_args()


class AllReduceAutograd(torch.autograd.Function):
    """Autograd-aware AllReduce for Megatron-style Tensor Parallelism."""

    @staticmethod
    def forward(ctx, x):
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out = grad_output.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out


class TPBenchmarkTransformerBlock(nn.Module):
    """Synthetic standard Transformer block matching GPT-2 dimensions for TP."""

    def __init__(self, hidden_dim: int, rank: int, world_size: int):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        # ColParallel (qkv): split out_features
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim // world_size, bias=False)
        # RowParallel (out_proj): split in_features
        self.out_proj = nn.Linear(hidden_dim // world_size, hidden_dim, bias=False)
        # ColParallel (fc1): split out_features
        self.fc1 = nn.Linear(hidden_dim, 4 * hidden_dim // world_size, bias=False)
        # RowParallel (fc2): split in_features
        self.fc2 = nn.Linear(4 * hidden_dim // world_size, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_out = self.out_proj(q)
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            attn_out = AllReduceAutograd.apply(attn_out)
        h = x + attn_out
        mlp_act = torch.relu(self.fc1(h))
        mlp_out = self.fc2(mlp_act)
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            mlp_out = AllReduceAutograd.apply(mlp_out)
        return h + mlp_out


def build_candidate_masks(
    delta: dict[str, torch.Tensor],
    prune_ratio: float,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build three candidate masks: global (lambda=0), local (lambda=1), hybrid (lambda=0.5)."""
    # Candidate 1: lambda = 0 (Global magnitude)
    all_abs = torch.cat([t.flatten().abs() for t in delta.values()])
    k_global = int(len(all_abs) * (1 - prune_ratio))
    thresh_global = torch.kthvalue(all_abs, len(all_abs) - k_global + 1).values.item() if k_global > 0 else float("inf")
    mask_global = {k: v.abs() >= thresh_global for k, v in delta.items()}

    # Candidate 2: lambda = 1 (Uniform per-tensor)
    mask_uniform = {}
    for k, v in delta.items():
        v_abs = v.flatten().abs()
        k_local = int(len(v_abs) * (1 - prune_ratio))
        thresh_local = torch.kthvalue(v_abs, len(v_abs) - k_local + 1).values.item() if k_local > 0 else float("inf")
        mask_uniform[k] = v.abs() >= thresh_local

    # Candidate 3: lambda = 0.5 (Hybrid intersection/union)
    mask_hybrid = {k: (mask_global[k] | mask_uniform[k]) for k in delta}

    return {
        "candidate_global_lambda0.0": mask_global,
        "candidate_hybrid_lambda0.5": mask_hybrid,
        "candidate_uniform_lambda1.0": mask_uniform,
    }


def main() -> None:
    args = parse_args()
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if is_distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        backend = args.backend or ("nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
    else:
        rank = 0
        world_size = 1
        device = args.device

    tp_config = TPConfig(rank=rank, world_size=world_size, device=device)

    # 1. Build synthetic TP model & inputs
    torch.manual_seed(42 + rank)
    model = TPBenchmarkTransformerBlock(args.model_dim, rank, world_size).to(device)
    x = torch.randn(args.batch_size, args.seq_len, args.model_dim, device=device)

    # 2. Build full reference delta (unpartitioned) across all layers
    torch.manual_seed(100)
    full_delta: dict[str, torch.Tensor] = {
        "qkv.weight": torch.randn(3 * args.model_dim, args.model_dim) * 0.02,
        "out_proj.weight": torch.randn(args.model_dim, args.model_dim) * 0.02,
        "fc1.weight": torch.randn(4 * args.model_dim, args.model_dim) * 0.02,
        "fc2.weight": torch.randn(args.model_dim, 4 * args.model_dim) * 0.02,
    }

    candidate_masks = build_candidate_masks(full_delta, args.prune_ratio)

    # 3. Shard for TP
    local_delta = shard_residual_scope_tp(full_delta, rank, world_size)
    local_candidate_masks = {
        cand_name: shard_residual_scope_tp(cand_masks, rank, world_size)
        for cand_name, cand_masks in candidate_masks.items()
    }

    # 4. Probe loss function
    def probe_loss():
        out = model(x)
        # Perturbation probe
        probe = torch.randn_like(out)
        return (out * probe).sum()

    # Track VRAM before
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
    else:
        mem_before = 0

    # 5. Execute Streamed TP Scoring
    winning_idx, global_scores, metrics = evaluate_tp_candidates_streamed(
        model=model,
        probe_loss_fn=probe_loss,
        local_delta=local_delta,
        candidate_masks=local_candidate_masks,
        tp_config=tp_config,
    )

    if torch.cuda.is_available() and "cuda" in device:
        peak_mem = torch.cuda.max_memory_allocated()
        mem_overhead = (peak_mem - mem_before) / (1024 * 1024)
    else:
        peak_mem = 0
        mem_overhead = 0.0

    metrics["peak_vram_mb"] = peak_mem / (1024 * 1024)
    metrics["overhead_vram_mb"] = mem_overhead

    # 6. Rank 0 Reporting
    if rank == 0:
        print("\n" + "=" * 60)
        print("  2x V100 TP Candidate Scoring Benchmark Results")
        print("=" * 60)
        print(f"  Device: {device} | Backend: {dist.get_backend() if is_distributed else 'SingleProcess'}")
        print(f"  TP World Size: {world_size} | Rank: {rank}")
        print(f"  Hidden Dim: {args.model_dim} | Prune Ratio: {args.prune_ratio}")
        print(f"  Winning Candidate: {metrics['winning_candidate']} (idx={winning_idx})")
        print(f"  Global Candidate Scores: {metrics['global_scores']}")
        print(f"  Total Duration: {metrics['total_duration_seconds'] * 1000:.2f} ms")
        print(f"  All-Reduce Comm Duration: {metrics['comm_duration_seconds'] * 1000:.4f} ms")
        print(f"  Communicated Bytes: {metrics['communicated_bytes']} bytes (Scalar All-Reduce)")
        if torch.cuda.is_available() and "cuda" in device:
            print(f"  Peak VRAM: {metrics['peak_vram_mb']:.2f} MB | Overhead: {metrics['overhead_vram_mb']:.2f} MB")
        print("=" * 60 + "\n")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = args.output_dir / f"tp_scoring_rank0_{int(time.time())}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved results to: {output_file}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
