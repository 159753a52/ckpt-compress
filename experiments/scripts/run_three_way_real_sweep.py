"""Real 3-Way Fine-Grained Sweep on V100: ExCP vs InShrinkerator vs DACP.
Runs authentic execution of all three methods across fine-grained sparsity levels.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ft-steps", type=int, default=100)
    parser.add_argument("--hvp-batches", type=int, default=16)
    parser.add_argument("--output-json", type=str, default="experiments/results/v100_three_way_sweep_real.json")
    args = parser.parse_args()
    device = args.device

    print("=" * 110)
    print("  Real 3-Way Benchmark Sweep on V100: ExCP (ICML '24) vs InShrinkerator (SoCC '24) vs DACP (Ours)")
    print(f"  Device: {device} | FT Steps: {args.ft_steps} | HVP Batches: {args.hvp_batches}")
    print("=" * 110)

    train_loader = get_wikitext2_dataloader(split="train", batch_size=4, seq_length=256, shuffle=True)
    val_loader = get_wikitext2_dataloader(split="validation", batch_size=4, seq_length=256, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    state_w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    val_loss_w0 = evaluate_dataloader(model, val_loader, device)
    ppl_w0 = compute_ppl(val_loss_w0)
    print(f"Base Pretrained W0: PPL = {ppl_w0:.2f}")

    print(f"Fine-tuning {args.ft_steps} steps with AdamW (lr=1e-4)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    train_iter = iter(train_loader)
    model.train()
    for step in range(args.ft_steps):
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
    print(f"Target Uncompressed W1: PPL = {val_ppl_w1:.2f} (FT Gain: {ppl_w0 - val_ppl_w1:.2f} PPL drop)\n")

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

    # 1. ExCP: Optimizer second moment mt
    print("[1/3] Computing ExCP (ICML '24) scores...")
    scores_excp = {}
    for n in eligible_names:
        param = named_params[n]
        st = optimizer.state.get(param, {})
        mt = st.get("exp_avg_sq", torch.ones_like(param)).to(device)
        scores_excp[n] = delta[n].abs() * torch.sqrt(mt + 1e-8)

    # 2. InShrinkerator: First-order sensitivity |g * delta_w|
    print("[2/3] Computing InShrinkerator (SoCC '24) first-order scores...")
    model.load_state_dict(state_w1)
    calib_batches = [next(train_iter) for _ in range(args.hvp_batches)]
    grad_accum = {n: torch.zeros_like(named_params[n]) for n in eligible_names}
    hvp_accum = {n: torch.zeros_like(named_params[n]) for n in eligible_names}

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

    # InShrinkerator scores: first-order gradient sensitivity
    scores_insh = {n: (avg_grads[n] * delta[n]).abs() for n in eligible_names}

    # 3. DACP: Second-order HVP + Signed Benefit + Curvature Protection
    print("[3/3] Computing DACP (Ours) second-order curvature scores...")
    # Signed Taylor damage: -g * delta + 0.5 * delta * H * delta
    # Parameters with higher damage if reverted should be preserved!
    scores_hvp_raw = {
        n: (-avg_grads[n] * delta[n] + 0.5 * delta[n] * avg_hvps[n]).abs()
        for n in eligible_names
    }

    # 2D Curvature + Variance Fusion with 1.0% Outlier Protection
    all_ex = torch.cat([scores_excp[n].flatten() for n in eligible_names])
    all_in = torch.cat([scores_insh[n].flatten() for n in eligible_names])
    all_hv = torch.cat([scores_hvp_raw[n].flatten() for n in eligible_names])
    N_params = all_ex.numel()

    r_ex = torch.argsort(torch.argsort(all_ex, stable=True), stable=True).float() / N_params
    r_in = torch.argsort(torch.argsort(all_in, stable=True), stable=True).float() / N_params
    r_hv = torch.argsort(torch.argsort(all_hv, stable=True), stable=True).float() / N_params

    scores_dacp = {}
    offset = 0
    for n in eligible_names:
        sz = delta[n].numel()
        re = r_ex[offset : offset + sz].view_as(delta[n])
        ri = r_in[offset : offset + sz].view_as(delta[n])
        rh = r_hv[offset : offset + sz].view_as(delta[n])
        # Curvature-dominant fusion: 60% HVP + 20% First-order + 20% Adam Variance
        comb = 0.60 * rh + 0.20 * ri + 0.20 * re
        # Outlier protection on top 1.0% critical parameters
        comb[(rh >= 0.990) | (ri >= 0.990)] = 1.0
        scores_dacp[n] = comb
        offset += sz

    # Selection functions
    def get_global_topk_mask(scores_dict, sp):
        all_s = torch.cat([s.flatten() for s in scores_dict.values()])
        k = int(all_s.numel() * (1 - sp))
        if k <= 0:
            return {n: torch.zeros_like(scores_dict[n], dtype=torch.bool) for n in eligible_names}
        th = torch.kthvalue(all_s, all_s.numel() - k + 1).values.item()
        return {n: scores_dict[n] >= th for n in eligible_names}

    def get_dacp_layer_floor_mask(sp):
        all_s = torch.cat([s.flatten() for s in scores_dacp.values()])
        total_target_keep = int(all_s.numel() * (1 - sp))
        layer_keep = {}
        kept = 0
        for n in eligible_names:
            s = scores_dacp[n]
            n_k = int(round(s.numel() * (1 - sp) * 0.30))  # 30% floor preservation per layer
            if n_k > 0:
                th_l = torch.kthvalue(s.flatten(), s.numel() - n_k + 1).values.item()
                m = s >= th_l
            else:
                m = torch.zeros_like(s, dtype=torch.bool)
            layer_keep[n] = m
            kept += m.sum().item()
        rem = max(0, total_target_keep - kept)
        if rem > 0:
            unsel = [scores_dacp[n][~layer_keep[n]].flatten() for n in eligible_names]
            pooled = torch.cat(unsel)
            if pooled.numel() >= rem:
                th_g = torch.kthvalue(pooled, pooled.numel() - rem + 1).values.item()
                for n in eligible_names:
                    layer_keep[n] = layer_keep[n] | ((~layer_keep[n]) & (scores_dacp[n] >= th_g))
        return layer_keep

    def eval_mask(mask_dict):
        compressed_state = {}
        for k, p in state_w1.items():
            if k in eligible_names:
                m = mask_dict[k]
                compressed_state[k] = state_w0[k] + delta[k] * m.float()
            else:
                compressed_state[k] = p
        model.load_state_dict(compressed_state)
        val_loss = evaluate_dataloader(model, val_loader, device)
        return compute_ppl(val_loss)

    sparsities = [0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]
    records = []

    print("-" * 110)
    print(f"{'Sparsity':<10} | {'CR':<6} | {'ExCP PPL':<11} {'(Δ)':<8} | {'InShrink PPL':<13} {'(Δ)':<8} | {'DACP PPL':<11} {'(Δ)':<8} | {'DACP vs InSh':<12}")
    print("-" * 110)

    for sp in sparsities:
        cr = 1.0 / (1.0 - sp)
        # 1. ExCP
        mask_excp = get_global_topk_mask(scores_excp, sp)
        ppl_excp = eval_mask(mask_excp)
        delta_excp = ppl_excp - val_ppl_w1

        # 2. InShrinkerator
        mask_insh = get_global_topk_mask(scores_insh, sp)
        ppl_insh = eval_mask(mask_insh)
        delta_insh = ppl_insh - val_ppl_w1

        # 3. DACP (Ours)
        mask_dacp = get_dacp_layer_floor_mask(sp)
        ppl_dacp = eval_mask(mask_dacp)
        delta_dacp = ppl_dacp - val_ppl_w1

        adv_over_insh = ppl_insh - ppl_dacp

        rec = {
            "sparsity": sp,
            "compression_ratio": cr,
            "excp_ppl": ppl_excp,
            "excp_delta": delta_excp,
            "inshrinkerator_ppl": ppl_insh,
            "inshrinkerator_delta": delta_insh,
            "dacp_ppl": ppl_dacp,
            "dacp_delta": delta_dacp,
            "dacp_advantage_over_inshrinkerator": adv_over_insh,
            "dacp_advantage_over_excp": ppl_excp - ppl_dacp,
        }
        records.append(rec)

        print(
            f"{sp*100:5.1f}%     | {cr:4.1f}× | "
            f"{ppl_excp:7.2f}    ({delta_excp:+5.2f}) | "
            f"{ppl_insh:7.2f}      ({delta_insh:+5.2f}) | "
            f"{ppl_dacp:7.2f}    ({delta_dacp:+5.2f}) | "
            f"{adv_over_insh:+5.2f} PPL lead"
        )

    print("-" * 110)
    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({"w0_ppl": ppl_w0, "w1_ppl": val_ppl_w1, "records": records}, f, indent=2)
    print(f"Results successfully saved to {out_p}")


if __name__ == "__main__":
    main()
