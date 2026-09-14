"""Real V100 Incremental Probing and Adaptive Early-Stopping Showdown.

Evaluates how many probing batches are actually needed for DACP curvature scoring,
and demonstrates that our online incremental early-stopping achieves lossless quality
(Delta PPL <= 0.15) while slashing probing compute overhead.
"""

from __future__ import annotations

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dacp.utils.data_loader import get_wikitext2_dataloader


def compute_ppl(loss_val: float) -> float:
    try:
        return math.exp(min(loss_val, 20.0))
    except OverflowError:
        return float("inf")


def evaluate_dataloader(model: nn.Module, data_loader, device: str, max_batches: int = 30) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device) if "labels" in batch else input_ids
            loss = model(input_ids=input_ids, labels=labels).loss
            total_loss += loss.item()
            count += 1
    return total_loss / max(1, count)


def run_incremental_probing_score(
    model: nn.Module,
    delta: Dict[str, torch.Tensor],
    eligible_names: List[str],
    named_params: Dict[str, nn.Parameter],
    scoring_batches: List[Dict[str, torch.Tensor]],
    opt: torch.optim.Optimizer,
    device: str,
    early_stopping_tol: float | None = None,
    min_batches: int = 1,
) -> Tuple[Dict[str, torch.Tensor], int, float, bool]:
    """Compute DACP curvature score using online incremental accumulation and early stopping."""
    t0 = time.perf_counter()
    model.eval()
    el_params = [named_params[n] for n in eligible_names]

    running_grad = {n: torch.zeros_like(delta[n]) for n in eligible_names}
    running_hvp = {n: torch.zeros_like(delta[n]) for n in eligible_names}
    prev_scores = None
    batches_used = 0
    early_stopped = False

    for b_idx, batch in enumerate(scoring_batches):
        batches_used += 1
        inp = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device) if "labels" in batch else inp

        with sdpa_kernel(SDPBackend.MATH):
            loss = model(input_ids=inp, labels=lbl).loss
            grads = torch.autograd.grad(loss, el_params, create_graph=True, retain_graph=True)
            probe = sum((g * delta[n]).sum() for n, g in zip(eligible_names, grads))
            hvps = torch.autograd.grad(probe, el_params, retain_graph=False)

        step_alpha = 1.0 / batches_used
        for n, g, h in zip(eligible_names, grads, hvps):
            running_grad[n].add_(g.detach() - running_grad[n], alpha=step_alpha)
            running_hvp[n].add_(h.detach() - running_hvp[n], alpha=step_alpha)

        del loss, grads, probe, hvps

        # Compute current HVP scores
        cur_hvp_scores = {
            n: (-running_grad[n] * delta[n] + 0.5 * delta[n] * running_hvp[n]).abs()
            for n in eligible_names
        }

        if (
            early_stopping_tol is not None
            and batches_used >= min_batches
            and batches_used < len(scoring_batches)
        ):
            if prev_scores is not None:
                tot_diff = sum((cur_hvp_scores[n] - prev_scores[n]).abs().sum().item() for n in eligible_names)
                tot_norm = sum(prev_scores[n].abs().sum().item() for n in eligible_names)
                rel_diff = tot_diff / (tot_norm + 1e-8)
                if rel_diff <= early_stopping_tol:
                    early_stopped = True
                    break
            prev_scores = {n: cur_hvp_scores[n].clone() for n in eligible_names}

    # Final Rank Fusion with Adam Momentum
    scores_mom = {}
    for n in eligible_names:
        st = opt.state.get(named_params[n], {})
        mt = st.get("exp_avg_sq", torch.ones_like(named_params[n])).to(device)
        scores_mom[n] = delta[n].abs() * torch.sqrt(mt + 1e-8)

    all_m = torch.cat([scores_mom[n].flatten() for n in eligible_names])
    all_h = torch.cat([cur_hvp_scores[n].flatten() for n in eligible_names])
    N_p = all_m.numel()
    rm = torch.argsort(torch.argsort(all_m, stable=True), stable=True).float() / N_p
    rh = torch.argsort(torch.argsort(all_h, stable=True), stable=True).float() / N_p

    comb = {}
    off = 0
    for n in eligible_names:
        sz = delta[n].numel()
        c = 0.60 * rh[off : off + sz].view_as(delta[n]) + 0.40 * rm[off : off + sz].view_as(delta[n])
        c[(rm[off : off + sz].view_as(delta[n]) >= 0.99) | (rh[off : off + sz].view_as(delta[n]) >= 0.99)] = 1.0
        comb[n] = c
        off += sz

    duration = time.perf_counter() - t0
    return comb, batches_used, duration, early_stopped


def apply_compression(
    curr_w: Dict[str, torch.Tensor],
    prev_w: Dict[str, torch.Tensor],
    delta: Dict[str, torch.Tensor],
    scores: Dict[str, torch.Tensor],
    eligible_names: List[str],
    sparsity: float,
) -> Dict[str, torch.Tensor]:
    all_s = torch.cat([s.flatten() for s in scores.values()])
    k = max(1, int(all_s.numel() * (1 - sparsity)))
    th = torch.kthvalue(all_s, all_s.numel() - k + 1).values.item()

    restored = {}
    for n in curr_w:
        if n in eligible_names:
            mask = (scores[n] >= th).to(delta[n].dtype)
            restored[n] = prev_w[n] + delta[n] * mask
        else:
            restored[n] = curr_w[n]
    if "lm_head.weight" in restored and "transformer.wte.weight" in restored:
        restored["lm_head.weight"] = restored["transformer.wte.weight"]
    return restored


def main():
    parser = argparse.ArgumentParser(description="Incremental Probing and Early Stopping Benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ft-steps", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--output-json", type=str, default="experiments/results/v100_incremental_probing_real.json")
    args = parser.parse_args()
    device = args.device

    print("=" * 115)
    print("  Real V100 Incremental Probing Showdown: Evaluating Probing Compute Reduction & Early Stopping")
    print(f"  Device: {device} | FT Steps: {args.ft_steps} | Max Probing Batches: {args.max_batches}")
    print("=" * 115)

    train_loader = get_wikitext2_dataloader(split="train", batch_size=4, seq_length=256, shuffle=True)
    val_loader = get_wikitext2_dataloader(split="validation", batch_size=4, seq_length=256, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    state_w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    loss_w0 = evaluate_dataloader(model, val_loader, device)
    ppl_w0 = compute_ppl(loss_w0)
    print(f"Base Model W0: Loss = {loss_w0:.4f} | PPL = {ppl_w0:.2f}")

    print(f"\nTraining model for {args.ft_steps} steps with AdamW (lr=1e-4)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    train_iter = iter(train_loader)
    model.train()
    for step in range(args.ft_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        inp = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device) if "labels" in batch else inp
        optimizer.zero_grad()
        loss = model(input_ids=inp, labels=lbl).loss
        loss.backward()
        optimizer.step()

    state_w1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    loss_w1 = evaluate_dataloader(model, val_loader, device)
    ppl_w1 = compute_ppl(loss_w1)
    print(f"Target Uncompressed Model W1: Loss = {loss_w1:.4f} | PPL = {ppl_w1:.2f} (Strict Target Baseline)")

    named_params = dict(model.named_parameters())
    eligible_names = [
        n for n in state_w1
        if n in named_params
        and state_w1[n].dim() >= 2
        and not any(p in n for p in ["wte", "wpe", "lm_head"])
    ]
    delta = {k: state_w1[k] - state_w0[k] for k in state_w1}

    # Prepare scoring batches
    scoring_batches = []
    for b in train_loader:
        scoring_batches.append(b)
        if len(scoring_batches) >= args.max_batches:
            break

    # Pre-measure baseline 8-batch scoring time
    _, _, base_8_time, _ = run_incremental_probing_score(
        model=model,
        delta=delta,
        eligible_names=eligible_names,
        named_params=named_params,
        scoring_batches=scoring_batches,
        opt=optimizer,
        device=device,
        early_stopping_tol=None,
        min_batches=1,
    )

    configurations = [
        {"name": "Fixed 1-Batch", "max_b": 1, "tol": None, "min_b": 1},
        {"name": "Fixed 2-Batches", "max_b": 2, "tol": None, "min_b": 1},
        {"name": "Fixed 4-Batches", "max_b": 4, "tol": None, "min_b": 1},
        {"name": "Fixed 8-Batches (Full)", "max_b": 8, "tol": None, "min_b": 1},
        {"name": "Adaptive Incremental (tol=0.35)", "max_b": 8, "tol": 0.35, "min_b": 2},
        {"name": "Adaptive Incremental (tol=0.25)", "max_b": 8, "tol": 0.25, "min_b": 2},
    ]

    target_sparsities = [0.75, 0.85]  # 4.0x and 6.7x physical compression
    full_results = []

    for sparsity in target_sparsities:
        comp_ratio = 1.0 / (1.0 - sparsity)
        print(f"\n{'=' * 115}")
        print(f"  Evaluating Sparsity: {sparsity:.1%} ({comp_ratio:.1f}x Compression) | Uncompressed W1 PPL: {ppl_w1:.2f}")
        print(f"{'=' * 115}")
        print(f"{'Method / Configuration':<35} | {'Batches':<8} | {'Scoring Time':<13} | {'PPL':<7} | {'ΔPPL':<8} | {'Speedup':<8} | {'Lossless?'}")
        print("-" * 115)

        for cfg in configurations:
            sub_batches = scoring_batches[: cfg["max_b"]]
            scores, b_used, s_time, stopped = run_incremental_probing_score(
                model=model,
                delta=delta,
                eligible_names=eligible_names,
                named_params=named_params,
                scoring_batches=sub_batches,
                opt=optimizer,
                device=device,
                early_stopping_tol=cfg["tol"],
                min_batches=cfg["min_b"],
            )

            restored = apply_compression(state_w1, state_w0, delta, scores, eligible_names, sparsity)
            model.load_state_dict(restored)
            rec_loss = evaluate_dataloader(model, val_loader, device)
            rec_ppl = compute_ppl(rec_loss)
            delta_ppl = rec_ppl - ppl_w1

            speedup = base_8_time / s_time
            is_lossless = delta_ppl <= 0.15
            lossless_str = "★ YES" if is_lossless else "NO"

            method_display = cfg["name"] + (" (Early Stopped)" if stopped else "")
            print(
                f"{method_display:<35} | {b_used:<8} | {s_time * 1000:>7.1f} ms    | {rec_ppl:<7.2f} | {delta_ppl:>+7.2f} | {speedup:>6.2f}x | {lossless_str}"
            )

            full_results.append({
                "sparsity": sparsity,
                "compression_ratio": comp_ratio,
                "configuration": cfg["name"],
                "batches_evaluated": b_used,
                "scoring_time_ms": s_time * 1000,
                "restored_ppl": rec_ppl,
                "delta_ppl": delta_ppl,
                "speedup": round(speedup, 3),
                "early_stopped": stopped,
                "is_lossless": is_lossless,
            })

    out_path = ROOT / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "uncompressed_w1_ppl": ppl_w1,
                "pretrained_w0_ppl": ppl_w0,
                "results": full_results,
            },
            f,
            indent=2,
        )
    print("\n" + "=" * 115)
    print(f"  Benchmark complete. Structured raw JSON saved to: {out_path}")
    print("=" * 115)


if __name__ == "__main__":
    main()
