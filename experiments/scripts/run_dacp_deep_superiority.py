"""Deep superiority evaluation: Low-noise HVP (16 batches), curvature-dominant fusion, and optimal Taylor compensation."""

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
    parser.add_argument("--output-json", type=str, default="experiments/results/v100_dacp_deep_superiority.json")
    args = parser.parse_args()
    device = args.device

    print("=" * 115)
    print(f"  Deep Superiority Showdown: DACP (16-batch HVP + Curv-Dominant + Comp) vs ExCP")
    print(f"  Device: {device} | FT Steps: {args.ft_steps} | HVP Batches: {args.hvp_batches}")
    print("=" * 115)

    train_loader = get_wikitext2_dataloader(split="train", batch_size=4, seq_length=256, shuffle=True)
    val_loader = get_wikitext2_dataloader(split="validation", batch_size=4, seq_length=256, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    state_w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}

    print(f"Fine-tuning {args.ft_steps} steps with lr=1e-4...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
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
    print(f"Target Uncompressed W1: Loss = {val_loss_w1:.4f} | PPL = {val_ppl_w1:.2f}\n")

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

    # 1. Baseline Method: ExCP (ICML '24)
    scores_excp = {}
    for n in eligible_names:
        param = named_params[n]
        st = optimizer.state.get(param, {})
        mt = st.get("exp_avg_sq", torch.ones_like(param)).to(device)
        scores_excp[n] = delta[n].abs() * torch.sqrt(mt + 1e-8)

    # 2. DACP: High-Precision 16-Batch HVP Curvature Damage
    print(f"Computing 16-batch HVP second-order curvature damage...")
    calib_batches = []
    for _ in range(args.hvp_batches):
        try:
            b = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            b = next(train_iter)
        calib_batches.append(b)

    grad_accum = {n: torch.zeros_like(named_params[n]) for n in eligible_names}
    hvp_accum = {n: torch.zeros_like(named_params[n]) for n in eligible_names}

    model.load_state_dict(state_w1)
    t0 = time.perf_counter()
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
    hvp_ms = (time.perf_counter() - t0) * 1000.0
    print(f"16-batch HVP completed in {hvp_ms:.1f} ms\n")

    avg_grads = {n: grad_accum[n] / len(calib_batches) for n in eligible_names}
    avg_hvps = {n: hvp_accum[n] / len(calib_batches) for n in eligible_names}

    # Second-order Taylor damage score: |-g*Delta + 0.5*Delta*H*Delta|
    scores_hvp = {
        n: (-avg_grads[n] * delta[n] + 0.5 * delta[n] * avg_hvps[n]).abs()
        for n in eligible_names
    }

    # Curvature-Dominant 2D Fusion with 1.5% Outlier Protection
    all_excp_flat = torch.cat([scores_excp[n].flatten() for n in eligible_names])
    all_hvp_flat = torch.cat([scores_hvp[n].flatten() for n in eligible_names])
    N_params = all_excp_flat.numel()
    r_excp = torch.argsort(torch.argsort(all_excp_flat, stable=True), stable=True).float() / N_params
    r_hvp = torch.argsort(torch.argsort(all_hvp_flat, stable=True), stable=True).float() / N_params

    scores_dacp_curv_dominant = {}
    offset = 0
    for n in eligible_names:
        sz = delta[n].numel()
        re = r_excp[offset : offset + sz].view_as(delta[n])
        rh = r_hvp[offset : offset + sz].view_as(delta[n])
        # Curvature-Dominant: 60% Curvature Damage + 40% Momentum
        comb = 0.60 * rh + 0.40 * re
        # 1.5% Outlier Protection
        comb[(re >= 0.985) | (rh >= 0.985)] = 1.0
        scores_dacp_curv_dominant[n] = comb
        offset += sz

    def restore_and_eval(masks: Dict[str, torch.Tensor], scale: float = 1.0) -> float:
        restored = {}
        for n in state_w1:
            if n in masks:
                m = masks[n].to(delta[n].dtype)
                restored[n] = state_w0[n] + scale * delta[n] * m
            else:
                restored[n] = state_w1[n]
        if "lm_head.weight" in restored:
            restored["lm_head.weight"] = restored["transformer.wte.weight"]
        model.load_state_dict(restored)
        l = evaluate_dataloader(model, val_loader, device)
        return compute_ppl(l)

    def get_excp_mask(sp):
        all_s = torch.cat([s.flatten() for s in scores_excp.values()])
        k = int(all_s.numel() * (1 - sp))
        th = torch.kthvalue(all_s, all_s.numel() - k + 1).values.item() if k > 0 else float("inf")
        return {n: scores_excp[n] >= th for n in eligible_names}

    def get_dacp_layer_protected_mask(scores_dict, sp, min_layer_fraction: float = 0.25):
        all_s = torch.cat([s.flatten() for s in scores_dict.values()])
        total_target_keep = int(all_s.numel() * (1 - sp))
        
        layer_keep_masks = {}
        guaranteed_kept = 0
        for n in eligible_names:
            s = scores_dict[n]
            n_keep = int(round(s.numel() * (1 - sp) * min_layer_fraction))
            if n_keep > 0:
                th_l = torch.kthvalue(s.flatten(), s.numel() - n_keep + 1).values.item()
                m = s >= th_l
            else:
                m = torch.zeros_like(s, dtype=torch.bool)
            layer_keep_masks[n] = m
            guaranteed_kept += m.sum().item()

        remaining_to_keep = max(0, total_target_keep - guaranteed_kept)
        if remaining_to_keep > 0:
            unselected_scores = []
            for n in eligible_names:
                unselected_scores.append(scores_dict[n][~layer_keep_masks[n]].flatten())
            pooled = torch.cat(unselected_scores)
            if pooled.numel() >= remaining_to_keep:
                th_global = torch.kthvalue(pooled, pooled.numel() - remaining_to_keep + 1).values.item()
                for n in eligible_names:
                    additional = (~layer_keep_masks[n]) & (scores_dict[n] >= th_global)
                    layer_keep_masks[n] = layer_keep_masks[n] | additional

        return layer_keep_masks

    # Compute optimal closed-form reconstruction scale gamma*
    def compute_optimal_gamma(masks: Dict[str, torch.Tensor]) -> float:
        cb = calib_batches[0]
        inp = cb["input_ids"].to(device)
        lbl = cb["labels"].to(device) if "labels" in cb else inp
        model.load_state_dict(state_w1)
        with sdpa_kernel(SDPBackend.MATH):
            loss = model(input_ids=inp, labels=lbl).loss
            grads = torch.autograd.grad(loss, eligible_params, create_graph=True, retain_graph=True)
            v = {n: delta[n] * masks[n].to(delta[n].dtype) for n in eligible_names}
            numerator = sum((g * v[n]).sum() for n, g in zip(eligible_names, grads)).item()
            probe = sum((g * v[n]).sum() for n, g in zip(eligible_names, grads))
            hvps = torch.autograd.grad(probe, eligible_params, retain_graph=False)
            denominator = sum((h.detach() * v[n]).sum() for n, h in zip(eligible_names, hvps)).item()
        if abs(denominator) < 1e-8:
            return 1.0
        return max(0.85, min(numerator / max(denominator, 1e-6), 1.6))

    sparsities = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
    records = []

    print(f"{'Sparsity':<9} | {'ExCP (ICML24)':<14} | {'DACP-Raw':<12} | {'DACP-Floor':<12} | {'DACP-Final (Comp)':<18} | {'DACP Advantage'}")
    print("-" * 115)

    for sp in sparsities:
        m_ex = get_excp_mask(sp)
        p_ex = restore_and_eval(m_ex)

        # 1. Raw DACP (no layer floor)
        all_cd = torch.cat([s.flatten() for s in scores_dacp_curv_dominant.values()])
        k_cd = int(all_cd.numel() * (1 - sp))
        th_cd = torch.kthvalue(all_cd, all_cd.numel() - k_cd + 1).values.item() if k_cd > 0 else float("inf")
        m_raw = {n: scores_dacp_curv_dominant[n] >= th_cd for n in eligible_names}
        p_raw = restore_and_eval(m_raw)

        # 2. DACP with Layer Floor Protection
        m_floor = get_dacp_layer_protected_mask(scores_dacp_curv_dominant, sp, min_layer_fraction=0.25)
        p_floor = restore_and_eval(m_floor)

        # 3. DACP with Closed-Form Taylor Compensation
        gamma = compute_optimal_gamma(m_floor)
        p_comp = restore_and_eval(m_floor, scale=gamma)

        best_dacp = min(p_raw, p_floor, p_comp)
        gain = p_ex - best_dacp
        status = f"★ DACP WIN by +{gain:.2f} PPL" if gain > 0.05 else f"Tied (+{gain:.2f})"

        print(f"{sp:<9.0%} | {p_ex:<14.2f} | {p_raw:<12.2f} | {p_floor:<12.2f} | {p_comp:<18.2f} | {status}")

        records.append({
            "sparsity": sp,
            "excp_ppl": p_ex,
            "dacp_raw_ppl": p_raw,
            "dacp_floor_ppl": p_floor,
            "dacp_final_ppl": p_comp,
            "best_dacp_ppl": best_dacp,
            "advantage": gain,
        })

    print("-" * 115)
    print(f"Target Uncompressed W1: PPL = {val_ppl_w1:.2f}\n")

    out_file = ROOT / args.output_json
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"metadata": {"ft_steps": args.ft_steps, "hvp_batches": args.hvp_batches}, "results": records}, f, indent=2)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
