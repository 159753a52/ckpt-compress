"""Diagnostic & Tuning Script to Optimize DACP against ExCP.

Explores:
1. Scoring formulations:
   - DACP-Abs: |-g*Delta + 0.5*Delta*H*Delta|
   - DACP-Signed: -g*Delta + 0.5*Delta*H*Delta
   - DACP-Curvature: 0.5*Delta*(H*Delta)
   - DACP-Momentum-Curvature Hybrid: |-g*Delta + 0.5*Delta*H*Delta| * sqrt(m_t)
2. Calibration Batches: B in [2, 4, 8]
3. Allocation: Uniform vs Weibull vs Exact Global
4. Fine-tuning step scale: 100 steps
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
from experiments.lib.residual_scoring import compute_block_taylor_scores, eligible_layers
from experiments.lib.residual_masks import exact_keep_mask, layer_masks
from experiments.lib.residual_methods import build_weibull_mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ft-steps", type=int, default=100)
    parser.add_argument("--hvp-batches", type=int, default=4)
    parser.add_argument("--sparsity", type=float, default=0.80)
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
    print("=" * 80)
    print(f"  Tuning & Diagnosing DACP vs ExCP on V100 GPU (Sparsity: {args.sparsity:.0%})")
    print("=" * 80)

    train_loader = get_wikitext2_dataloader(split="train", batch_size=4, seq_length=256, shuffle=True)
    val_loader = get_wikitext2_dataloader(split="validation", batch_size=4, seq_length=256, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    state_w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}

    print(f"Training for {args.ft_steps} steps...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    train_iter = iter(train_loader)
    model.train()
    for s in range(args.ft_steps):
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
    print(f"Target W1 -> Loss: {val_loss_w1:.4f} | PPL: {val_ppl_w1:.2f}\n")

    delta = {k: state_w1[k] - state_w0[k] for k in state_w0}
    layers = eligible_layers(model, "gpt2")
    eligible_names = [name for layer in layers for name in layer]
    named_params = dict(model.named_parameters())

    def restore_and_eval(masks: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        restored = {}
        for name in state_w1:
            if name in masks:
                m = masks[name].to(device=delta[name].device, dtype=delta[name].dtype)
                restored[name] = state_w0[name] + delta[name] * m
            else:
                restored[name] = state_w1[name]
        if "lm_head.weight" in restored and "transformer.wte.weight" in restored:
            restored["lm_head.weight"] = restored["transformer.wte.weight"]
        model.load_state_dict(restored)
        loss = evaluate_dataloader(model, val_loader, device)
        return loss, compute_ppl(loss)

    # 1. Baseline 1: Pure Magnitude
    all_mag = torch.cat([delta[n].abs().flatten() for n in eligible_names])
    km = int(all_mag.numel() * (1 - args.sparsity))
    tm = torch.kthvalue(all_mag, all_mag.numel() - km + 1).values.item()
    mask_mag = {name: (delta[name].abs() >= tm).to(device) for name in eligible_names}
    l_mag, p_mag = restore_and_eval(mask_mag)
    print(f"1. Magnitude (Global)        -> Loss: {l_mag:.4f} | PPL: {p_mag:.2f}")

    # 2. Baseline 2: ExCP (ICML '24)
    scores_excp = {}
    for name in eligible_names:
        param = named_params[name]
        st = optimizer.state.get(param, {})
        m_t = st.get("exp_avg_sq", torch.ones_like(param)).to(device)
        scores_excp[name] = delta[name].abs() * torch.sqrt(m_t + 1e-8)

    all_ex = torch.cat([scores_excp[n].flatten() for n in eligible_names])
    ke = int(all_ex.numel() * (1 - args.sparsity))
    te = torch.kthvalue(all_ex, all_ex.numel() - ke + 1).values.item()
    mask_excp = {name: (scores_excp[name] >= te).to(device) for name in eligible_names}
    l_excp, p_excp = restore_and_eval(mask_excp)
    print(f"2. ExCP (ICML '24)           -> Loss: {l_excp:.4f} | PPL: {p_excp:.2f}")

    # 3. DACP Variants
    # Compute Blockwise HVP on calibration batches
    scoring_batches = []
    sc_iter = iter(train_loader)
    for _ in range(args.hvp_batches):
        sb = next(sc_iter)
        scoring_batches.append({"input_ids": sb["input_ids"].to(device), "labels": sb["labels"].to(device)})

    # Variant A: DACP Abs-Mean Blockwise Taylor
    scores_dacp_abs, _ = compute_block_taylor_scores(
        model=model,
        batches=scoring_batches,
        layers=layers,
        delta=delta,
        device=device,
        aggregation="abs_mean",
    )

    # Variant B: DACP Weibull on Abs-Mean
    masks_dacp_weibull, _ = build_weibull_mask(
        layers=layers,
        taylor_scores=scores_dacp_abs,
        prune_ratio=args.sparsity,
        max_layer_ratio=0.90,
    )
    masks_dacp_weibull = {k: v.to(device) for k, v in masks_dacp_weibull.items()}
    l_wbl, p_wbl = restore_and_eval(masks_dacp_weibull)
    print(f"3. DACP (Taylor + Weibull)   -> Loss: {l_wbl:.4f} | PPL: {p_wbl:.2f}")

    # Variant C: DACP Curvature Normalized (Taylor / sqrt(diag(H)) or Taylor * sqrt(m_t))
    # Combining HVP second order curvature with historical momentum:
    scores_dacp_mom = {}
    for name in eligible_names:
        param = named_params[name]
        st = optimizer.state.get(param, {})
        m_t = st.get("exp_avg_sq", torch.ones_like(param)).to(device)
        # Weight Taylor score by historical activity sqrt(m_t)
        scores_dacp_mom[name] = scores_dacp_abs[name].to(device) * torch.sqrt(m_t + 1e-8)

    all_dm = torch.cat([scores_dacp_mom[n].flatten() for n in eligible_names])
    kdm = int(all_dm.numel() * (1 - args.sparsity))
    tdm = torch.kthvalue(all_dm, all_dm.numel() - kdm + 1).values.item()
    mask_dm = {name: (scores_dacp_mom[name] >= tdm).to(device) for name in eligible_names}
    l_dm, p_dm = restore_and_eval(mask_dm)
    print(f"4. DACP-Curvature+Momentum   -> Loss: {l_dm:.4f} | PPL: {p_dm:.2f}")

    # Variant D: Weibull on DACP-Curvature+Momentum
    masks_dm_wbl, _ = build_weibull_mask(
        layers=layers,
        taylor_scores={k: v.cpu() for k, v in scores_dacp_mom.items()},
        prune_ratio=args.sparsity,
        max_layer_ratio=0.90,
    )
    masks_dm_wbl = {k: v.to(device) for k, v in masks_dm_wbl.items()}
    l_dm_w, p_dm_w = restore_and_eval(masks_dm_wbl)
    print(f"5. DACP-Mom + Weibull        -> Loss: {l_dm_w:.4f} | PPL: {p_dm_w:.2f}")

    # Compare advantages
    best_dacp_ppl = min(p_wbl, p_dm, p_dm_w)
    print("\n" + "=" * 80)
    print(f"Summary @ Sparsity {args.sparsity:.0%}:")
    print(f"  ExCP PPL:            {p_excp:.2f}")
    print(f"  Best DACP PPL:       {best_dacp_ppl:.2f}")
    if best_dacp_ppl < p_excp:
        print(f"  >>> SUCCESS: DACP beats ExCP by {p_excp - best_dacp_ppl:.2f} PPL!")
    else:
        print(f"  ExCP gap: {best_dacp_ppl - p_excp:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
