"""End-to-End Compression Showdown: ExCP vs DACP.

Evaluates Pruning + 4-bit Quantization at High Sparsity (90%, 95%, 98%, 99%, 99.5%).
Measures:
1. Exact PPL recovery.
2. End-to-end compressed bytes (with Int4 packing + gzip).
3. Compression Ratio (CR).
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
from baselines.excp.quantization import kmeans_quantize_nonzero, pack_int4, unpack_int4


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
    parser.add_argument("--output-json", type=str, default="experiments/results/v100_dacp_e2e_showdown.json")
    args = parser.parse_args()
    device = args.device

    print("=" * 115)
    print("  End-to-End Compression Showdown (Pruning + 4-bit Quantization): DACP vs ExCP")
    print(f"  Device: {device} | FT Steps: {args.ft_steps}")
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

    # Compute ExCP Scores
    scores_excp = {}
    for n in eligible_names:
        param = named_params[n]
        st = optimizer.state.get(param, {})
        mt = st.get("exp_avg_sq", torch.ones_like(param)).to(device)
        scores_excp[n] = delta[n].abs() * torch.sqrt(mt + 1e-8)

    # Compute DACP HVP Curvature (16 batches)
    print("Computing 16-batch HVP Curvature...")
    calib_batches = [next(train_iter) for _ in range(16)]
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

    # DACP Curvature-Dominant 2D Fusion with Outlier Protection
    all_ex = torch.cat([scores_excp[n].flatten() for n in eligible_names])
    all_hv = torch.cat([scores_hvp[n].flatten() for n in eligible_names])
    N = all_ex.numel()
    r_ex = torch.argsort(torch.argsort(all_ex, stable=True), stable=True).float() / N
    r_hv = torch.argsort(torch.argsort(all_hv, stable=True), stable=True).float() / N

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

    # Quantize pruned delta using 4-bit K-means (non-zero entries)
    def quantize_and_restore(masks: Dict[str, torch.Tensor]) -> Tuple[float, int]:
        restored = {}
        total_bytes = 0
        for n in state_w1:
            if n in masks:
                m = masks[n]
                pruned_d = delta[n] * m.to(delta[n].dtype)
                # 4-bit K-means quantization on non-zero coordinates
                if pruned_d.count_nonzero() > 0:
                    indices, centers = kmeans_quantize_nonzero(pruned_d.cpu(), n_bits=4, max_iter=30)
                    recon_d = centers[indices.long()].to(device=device, dtype=delta[n].dtype)
                    packed = pack_int4(indices)
                    compressed_bytes = len(gzip.compress(packed.numpy().tobytes()))
                    total_bytes += compressed_bytes
                else:
                    recon_d = torch.zeros_like(pruned_d)
                restored[n] = state_w0[n] + recon_d
            else:
                restored[n] = state_w1[n]
        if "lm_head.weight" in restored:
            restored["lm_head.weight"] = restored["transformer.wte.weight"]
        model.load_state_dict(restored)
        l = evaluate_dataloader(model, val_loader, device)
        return compute_ppl(l), total_bytes

    def restore_pruned_only(masks: Dict[str, torch.Tensor]) -> float:
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

    sparsities = [0.80, 0.90, 0.95, 0.98, 0.99, 0.995]
    records = []

    print(f"{'Sparsity':<9} | {'ExCP Pruned':<12} | {'DACP Pruned':<12} | {'ExCP (4-bit+Gzip)':<18} | {'DACP (4-bit+Gzip)':<18} | {'DACP Advantage'}")
    print("-" * 115)

    for sp in sparsities:
        m_ex = get_excp_mask(sp)
        p_ex_prune = restore_pruned_only(m_ex)
        p_ex_q, bytes_ex = quantize_and_restore(m_ex)

        m_da = get_dacp_mask(sp)
        p_da_prune = restore_pruned_only(m_da)
        p_da_q, bytes_da = quantize_and_restore(m_da)

        q_advantage = p_ex_q - p_da_q
        status = f"★ DACP WIN by +{q_advantage:.2f} PPL" if q_advantage > 0.05 else f"Tied (+{q_advantage:.2f})"

        print(f"{sp:<9.1%} | {p_ex_prune:<12.2f} | {p_da_prune:<12.2f} | {p_ex_q:<18.2f} | {p_da_q:<18.2f} | {status}")

        records.append({
            "sparsity": sp,
            "excp_pruned_ppl": p_ex_prune,
            "dacp_pruned_ppl": p_da_prune,
            "excp_quant_ppl": p_ex_q,
            "dacp_quant_ppl": p_da_q,
            "dacp_advantage": q_advantage,
            "excp_bytes": bytes_ex,
            "dacp_bytes": bytes_da,
        })

    print("-" * 115)
    print(f"Target Uncompressed W1: PPL = {val_ppl_w1:.2f}\n")

    out_file = ROOT / args.output_json
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
