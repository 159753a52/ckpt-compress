"""Fine-grained sweep to pinpoint the exact Near-Lossless Compression Frontier.

Compares DACP vs ExCP across fine-grained sparsity levels:
[0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]

Identifies the maximum compression ratio (CR) under near-lossless criteria:
- Ultra-strict: Delta PPL <= 0.10
- Strict:       Delta PPL <= 0.20
- Tolerant:     Delta PPL <= 0.50
"""

from __future__ import annotations

import argparse
import gzip
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
from baselines.excp.quantization import kmeans_quantize_nonzero, pack_int4


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
    parser.add_argument("--output-json", type=str, default="experiments/results/v100_lossless_frontier_sweep.json")
    args = parser.parse_args()
    device = args.device

    print("=" * 115)
    print("  Fine-Grained Near-Lossless Frontier Sweep: DACP vs ExCP (ICML '24)")
    print(f"  Device: {device} | FT Steps: {args.ft_steps} | HVP Batches: {args.hvp_batches}")
    print("=" * 115)

    train_loader = get_wikitext2_dataloader(split="train", batch_size=4, seq_length=256, shuffle=True)
    val_loader = get_wikitext2_dataloader(split="validation", batch_size=4, seq_length=256, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    state_w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    val_loss_w0 = evaluate_dataloader(model, val_loader, device)
    ppl_w0 = compute_ppl(val_loss_w0)
    print(f"Base Pretrained W0: PPL = {ppl_w0:.2f}")

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
    print(f"Target Uncompressed W1: PPL = {val_ppl_w1:.2f} (Total FT Gain: {ppl_w0 - val_ppl_w1:.2f} PPL drop)\n")

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

    # 1. ExCP Scores
    scores_excp = {}
    for n in eligible_names:
        param = named_params[n]
        st = optimizer.state.get(param, {})
        mt = st.get("exp_avg_sq", torch.ones_like(param)).to(device)
        scores_excp[n] = delta[n].abs() * torch.sqrt(mt + 1e-8)

    # 2. DACP 16-Batch HVP Curvature
    print("Computing 16-batch HVP Curvature...")
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
    scores_hvp = {
        n: (-avg_grads[n] * delta[n] + 0.5 * delta[n] * avg_hvps[n]).abs()
        for n in eligible_names
    }

    # Curvature-Dominant 2D Fusion with 1.5% Outlier Protection
    all_ex = torch.cat([scores_excp[n].flatten() for n in eligible_names])
    all_hv = torch.cat([scores_hvp[n].flatten() for n in eligible_names])
    N_params = all_ex.numel()
    r_ex = torch.argsort(torch.argsort(all_ex, stable=True), stable=True).float() / N_params
    r_hv = torch.argsort(torch.argsort(all_hv, stable=True), stable=True).float() / N_params

    scores_dacp = {}
    offset = 0
    for n in eligible_names:
        sz = delta[n].numel()
        re = r_ex[offset : offset + sz].view_as(delta[n])
        rh = r_hv[offset : offset + sz].view_as(delta[n])
        comb = 0.60 * rh + 0.40 * re
        comb[(re >= 0.985) | (rh >= 0.985)] = 1.0
        scores_dacp[n] = comb
        offset += sz

    def get_excp_mask(sp):
        all_s = torch.cat([s.flatten() for s in scores_excp.values()])
        k = int(all_s.numel() * (1 - sp))
        th = torch.kthvalue(all_s, all_s.numel() - k + 1).values.item() if k > 0 else float("inf")
        return {n: scores_excp[n] >= th for n in eligible_names}

    def get_dacp_mask(sp):
        all_s = torch.cat([s.flatten() for s in scores_dacp.values()])
        total_target_keep = int(all_s.numel() * (1 - sp))
        layer_keep = {}
        kept = 0
        for n in eligible_names:
            s = scores_dacp[n]
            n_k = int(round(s.numel() * (1 - sp) * 0.25))
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

    def restore_and_eval(masks: Dict[str, torch.Tensor]) -> float:
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
        return compute_ppl(l)

    fine_grained_sparsities = [
        0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98
    ]

    print(f"{'Sparsity':<9} | {'Ratio':<7} | {'ExCP PPL':<10} | {'ExCP Δ':<9} | {'DACP PPL':<10} | {'DACP Δ':<9} | {'Near-Lossless Status'}")
    print("-" * 105)

    records = []
    for sp in fine_grained_sparsities:
        ratio = 1.0 / (1.0 - sp)
        m_ex = get_excp_mask(sp)
        p_ex = restore_and_eval(m_ex)
        d_ex = p_ex - val_ppl_w1

        m_da = get_dacp_mask(sp)
        p_da = restore_and_eval(m_da)
        d_da = p_da - val_ppl_w1

        # Near-lossless tag
        if d_da <= 0.10:
            status = "★ Ultra-Lossless (Δ ≤ 0.1)"
        elif d_da <= 0.20:
            status = "★ Strict-Lossless (Δ ≤ 0.2)"
        elif d_da <= 0.50:
            status = "★ Near-Lossless (Δ ≤ 0.5)"
        else:
            status = f"Graceful Drift (Δ = {d_da:+.2f})"

        print(f"{sp:<9.1%} | {ratio:<7.1f}x | {p_ex:<10.2f} | {d_ex:<+9.2f} | {p_da:<10.2f} | {d_da:<+9.2f} | {status}")

        records.append({
            "sparsity": sp,
            "compression_ratio": ratio,
            "excp_ppl": p_ex,
            "excp_delta": d_ex,
            "dacp_ppl": p_da,
            "dacp_delta": d_da,
            "advantage": p_ex - p_da,
        })

    print("-" * 105)
    print(f"Uncompressed Baseline W1: PPL = {val_ppl_w1:.2f}\n")

    out_file = ROOT / args.output_json
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {"w0_ppl": ppl_w0, "w1_ppl": val_ppl_w1, "ft_steps": args.ft_steps},
            "records": records,
        }, f, indent=2)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
