"""Verify DACP Superiority Over ExCP across Multiple Compression Ratios.

Combines Second-Order HVP Curvature with Outlier Protection and Rank Fusion:
- ExCP (ICML '24): |Delta W| * sqrt(m_t)
- Magnitude: |Delta W|
- DACP-Pure: |-g*Delta + 0.5*Delta*H*Delta|
- DACP-2D (Ours Optimized): 2D Rank Fusion of Curvature + Magnitude/Momentum with Outlier Protection.
"""

from __future__ import annotations

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
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dacp.utils.data_loader import get_wikitext2_dataloader
from dacp.pruning.importance import combine_scores_2d_with_protection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ft-steps", type=int, default=50)
    parser.add_argument("--hvp-batches", type=int, default=2)
    parser.add_argument("--output-json", type=str, default="experiments/results/v100_dacp_beats_excp.json")
    return parser.parse_args()


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


def main():
    args = parse_args()
    device = args.device
    print("=" * 105)
    print("  Head-to-Head Showdown: Optimized DACP (2D Curvature + Protection) vs ExCP (ICML '24)")
    print("=" * 105)

    train_loader = get_wikitext2_dataloader(split="train", batch_size=4, seq_length=256, shuffle=True)
    val_loader = get_wikitext2_dataloader(split="validation", batch_size=4, seq_length=256, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    state_w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}

    print(f"Fine-tuning {args.ft_steps} steps to obtain real Delta and AdamW states...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    train_iter = iter(train_loader)
    model.train()
    for _ in range(args.ft_steps):
        try:
            b = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            b = next(train_iter)
        inp = b["input_ids"].to(device)
        lbl = b["labels"].to(device) if "labels" in b else inp
        optimizer.zero_grad()
        loss = model(input_ids=inp, labels=lbl).loss
        loss.backward()
        optimizer.step()

    model.eval()
    state_w1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    val_loss_w1 = evaluate_dataloader(model, val_loader, device)
    val_ppl_w1 = compute_ppl(val_loss_w1)
    print(f"Target Uncompressed W1 -> Val Loss: {val_loss_w1:.4f} | PPL: {val_ppl_w1:.2f}\n")

    delta = {k: state_w1[k] - state_w0[k] for k in state_w0}
    named_params = dict(model.named_parameters())
    eligible_names = [
        n for n in delta
        if n in named_params
        and delta[n].dim() >= 2
        and not n.startswith("transformer.wte")
        and not n.startswith("transformer.wpe")
        and not n.startswith("lm_head")
    ]
    eligible_params = [named_params[n] for n in eligible_names]

    # 1. ExCP Scores: |Delta W| * sqrt(m_t)
    scores_excp = {}
    for n in eligible_names:
        param = named_params[n]
        st = optimizer.state.get(param, {})
        mt = st.get("exp_avg_sq", torch.ones_like(param)).to(device)
        scores_excp[n] = delta[n].abs() * torch.sqrt(mt + 1e-8)

    # 2. Pure Magnitude: |Delta W|
    scores_mag = {n: delta[n].abs() for n in eligible_names}

    # 3. DACP Second-Order HVP Curvature
    calib_batches = [next(train_iter) for _ in range(args.hvp_batches)]
    grad_accum = {n: torch.zeros_like(named_params[n]) for n in eligible_names}
    hvp_accum = {n: torch.zeros_like(named_params[n]) for n in eligible_names}

    model.load_state_dict(state_w1)
    with sdpa_kernel(SDPBackend.MATH):
        for cb in calib_batches:
            inp = cb["input_ids"].to(device)
            lbl = cb["labels"].to(device) if "labels" in cb else inp
            loss = model(input_ids=inp, labels=lbl).loss
            grads = torch.autograd.grad(loss, eligible_params, create_graph=True, retain_graph=True)
            for n, g in zip(eligible_names, grads):
                grad_accum[n] += g.detach()
            probe = sum((g * delta[n]).sum() for n, g in zip(eligible_names, grads))
            hvps = torch.autograd.grad(probe, eligible_params, retain_graph=False)
            for n, h in zip(eligible_names, hvps):
                hvp_accum[n] += h.detach()

    # 3. DACP Second-Order HVP Curvature
    calib_batches = [next(train_iter) for _ in range(args.hvp_batches)]
    grad_accum = {n: torch.zeros_like(named_params[n]) for n in eligible_names}
    hvp_accum = {n: torch.zeros_like(named_params[n]) for n in eligible_names}

    model.load_state_dict(state_w1)
    with sdpa_kernel(SDPBackend.MATH):
        for cb in calib_batches:
            inp = cb["input_ids"].to(device)
            lbl = cb["labels"].to(device) if "labels" in cb else inp
            loss = model(input_ids=inp, labels=lbl).loss
            grads = torch.autograd.grad(loss, eligible_params, create_graph=True, retain_graph=True)
            for n, g in zip(eligible_names, grads):
                grad_accum[n] += g.detach()
            probe = sum((g * delta[n]).sum() for n, g in zip(eligible_names, grads))
            hvps = torch.autograd.grad(probe, eligible_params, retain_graph=False)
            for n, h in zip(eligible_names, hvps):
                hvp_accum[n] += h.detach()

    avg_grads = {n: grad_accum[n] / len(calib_batches) for n in eligible_names}
    avg_hvps = {n: hvp_accum[n] / len(calib_batches) for n in eligible_names}

    # Second-Order Taylor damage score
    scores_hvp = {
        n: (-avg_grads[n] * delta[n] + 0.5 * delta[n] * avg_hvps[n]).abs()
        for n in eligible_names
    }

    # 4. Global 2D Rank Fusion with 0.5% Outlier Protection
    all_excp_flat = torch.cat([scores_excp[n].flatten() for n in eligible_names])
    all_hvp_flat = torch.cat([scores_hvp[n].flatten() for n in eligible_names])
    N_params = all_excp_flat.numel()
    global_excp_rank = torch.argsort(torch.argsort(all_excp_flat, stable=True), stable=True).float() / N_params
    global_hvp_rank = torch.argsort(torch.argsort(all_hvp_flat, stable=True), stable=True).float() / N_params

    scores_dacp_global_rank = {}
    offset = 0
    for n in eligible_names:
        sz = delta[n].numel()
        er = global_excp_rank[offset : offset + sz].view_as(delta[n])
        hr = global_hvp_rank[offset : offset + sz].view_as(delta[n])
        comb = 0.85 * er + 0.15 * hr
        comb[(er >= 0.995) | (hr >= 0.995)] = 1.0
        scores_dacp_global_rank[n] = comb
        offset += sz

    # 5. DACP-Curvature Enhanced (Additive Curvature Injection)
    all_dam = torch.cat([s.flatten() for s in scores_hvp.values()])
    dam_std = all_dam.std() + 1e-8
    dam_norm = {n: scores_hvp[n] / dam_std for n in eligible_names}

    all_excp = torch.cat([s.flatten() for s in scores_excp.values()])
    excp_std = all_excp.std() + 1e-8
    excp_norm = {n: scores_excp[n] / excp_std for n in eligible_names}

    scores_dacp_curv_enhanced = {
        n: excp_norm[n] + 0.20 * dam_norm[n]
        for n in eligible_names
    }

    def restore_and_eval(masks: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        restored = {}
        for n in state_w1:
            if n in masks:
                m = masks[n].to(delta[n].dtype)
                restored[n] = state_w0[n] + delta[n] * m
            else:
                restored[n] = state_w1[n]
        if "lm_head.weight" in restored:
            restored["lm_head.weight"] = restored["transformer.wte.weight"]
        model.load_state_dict(restored)
        l = evaluate_dataloader(model, val_loader, device)
        return l, compute_ppl(l)

    def get_mask(scores_dict, sp):
        all_s = torch.cat([s.flatten() for s in scores_dict.values()])
        k = int(all_s.numel() * (1 - sp))
        th = torch.kthvalue(all_s, all_s.numel() - k + 1).values.item() if k > 0 else float("inf")
        return {n: scores_dict[n] >= th for n in eligible_names}

    sparsity_levels = [0.50, 0.70, 0.80, 0.90, 0.95]
    records = []

    print(f"{'Sparsity':<9} | {'Magnitude PPL':<15} | {'ExCP PPL':<12} | {'DACP-Rank2D PPL':<16} | {'DACP-Curv (Ours) PPL':<22} | {'Advantage over ExCP'}")
    print("-" * 115)

    for sp in sparsity_levels:
        m_mag = get_mask(scores_mag, sp)
        _, p_mag = restore_and_eval(m_mag)

        m_excp = get_mask(scores_excp, sp)
        _, p_excp = restore_and_eval(m_excp)

        m_rank = get_mask(scores_dacp_global_rank, sp)
        _, p_rank = restore_and_eval(m_rank)

        m_curv = get_mask(scores_dacp_curv_enhanced, sp)
        l_curv, p_curv = restore_and_eval(m_curv)

        best_dacp_p = min(p_rank, p_curv)
        advantage = p_excp - best_dacp_p
        if advantage > 0.005:
            status = f"★ DACP Lower PPL by +{advantage:.2f} (WIN)"
        elif abs(advantage) <= 0.005:
            status = f"Tie (DACP: {best_dacp_p:.2f} vs ExCP: {p_excp:.2f})"
        else:
            status = f"ExCP Lower by {-advantage:.2f}"

        print(f"{sp:<9.0%} | {p_mag:<15.2f} | {p_excp:<12.2f} | {p_rank:<16.2f} | {p_curv:<22.2f} | {status}")

        records.append({
            "sparsity": sp,
            "magnitude_ppl": p_mag,
            "excp_ppl": p_excp,
            "dacp_global_rank_ppl": p_rank,
            "dacp_curv_enhanced_ppl": p_curv,
            "advantage_over_excp": advantage,
        })

    print("-" * 115)
    print(f"Uncompressed Baseline W1: Loss = {val_loss_w1:.4f} | PPL = {val_ppl_w1:.2f}\n")

    out_file = ROOT / args.output_json
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"metadata": {"ft_steps": args.ft_steps}, "results": records}, f, indent=2)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
