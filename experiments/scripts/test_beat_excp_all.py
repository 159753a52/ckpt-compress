"""Test formulations to beat ExCP across all sparsity levels (50%, 70%, 80%, 90%, 95%)."""

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
    parser.add_argument("--ft-steps", type=int, default=50)
    args = parser.parse_args()
    device = args.device

    train_loader = get_wikitext2_dataloader(split="train", batch_size=4, seq_length=256, shuffle=True)
    val_loader = get_wikitext2_dataloader(split="validation", batch_size=4, seq_length=256, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    state_w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}

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
    print(f"Uncompressed W1 PPL: {compute_ppl(val_loss_w1):.2f}")

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

    # Baseline: ExCP
    scores_excp = {}
    for n in eligible_names:
        param = named_params[n]
        st = optimizer.state.get(param, {})
        mt = st.get("exp_avg_sq", torch.ones_like(param)).to(device)
        scores_excp[n] = delta[n].abs() * torch.sqrt(mt + 1e-8)

    # DACP HVP Curvature
    calib_batches = [next(train_iter) for _ in range(2)]
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

    # Second-order Taylor damage: |-g*Delta + 0.5*Delta*H*Delta|
    scores_hvp = {
        n: (-avg_grads[n] * delta[n] + 0.5 * delta[n] * avg_hvps[n]).abs()
        for n in eligible_names
    }

    # Normalize damage scores globally
    all_dam = torch.cat([s.flatten() for s in scores_hvp.values()])
    dam_std = all_dam.std() + 1e-8
    dam_norm = {n: scores_hvp[n] / dam_std for n in eligible_names}

    all_excp = torch.cat([s.flatten() for s in scores_excp.values()])
    excp_std = all_excp.std() + 1e-8
    excp_norm = {n: scores_excp[n] / excp_std for n in eligible_names}

    # Formulations to test
    # 1. Global 2D Rank Fusion with 0.5% protection
    all_excp_flat = torch.cat([scores_excp[n].flatten() for n in eligible_names])
    all_hvp_flat = torch.cat([scores_hvp[n].flatten() for n in eligible_names])
    N = all_excp_flat.numel()
    global_excp_rank = torch.argsort(torch.argsort(all_excp_flat, stable=True), stable=True).float() / N
    global_hvp_rank = torch.argsort(torch.argsort(all_hvp_flat, stable=True), stable=True).float() / N

    scores_global_rank = {}
    scores_global_rank_80 = {}
    scores_global_rank_90 = {}
    offset = 0
    for n in eligible_names:
        sz = delta[n].numel()
        er = global_excp_rank[offset : offset + sz].view_as(delta[n])
        hr = global_hvp_rank[offset : offset + sz].view_as(delta[n])
        
        comb85 = 0.85 * er + 0.15 * hr
        comb85[(er >= 0.995) | (hr >= 0.995)] = 1.0
        scores_global_rank[n] = comb85

        comb80 = 0.80 * er + 0.20 * hr
        comb80[(er >= 0.995) | (hr >= 0.995)] = 1.0
        scores_global_rank_80[n] = comb80

        comb90 = 0.90 * er + 0.10 * hr
        comb90[(er >= 0.995) | (hr >= 0.995)] = 1.0
        scores_global_rank_90[n] = comb90

        offset += sz

    # 2. Raw additive fusion: excp_norm + lambda * dam_norm
    scores_add_01 = {n: excp_norm[n] + 0.10 * dam_norm[n] for n in eligible_names}
    scores_add_02 = {n: excp_norm[n] + 0.20 * dam_norm[n] for n in eligible_names}
    scores_add_005 = {n: excp_norm[n] + 0.05 * dam_norm[n] for n in eligible_names}

    # 3. Multiplicative curvature modulation: |Delta| * (sqrt(mt) * (1 + 0.1 * dam_norm))
    scores_mult_01 = {
        n: scores_excp[n] * (1.0 + 0.10 * torch.clamp(dam_norm[n], 0.0, 5.0))
        for n in eligible_names
    }

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

    def get_mask(scores_dict, sp):
        all_s = torch.cat([s.flatten() for s in scores_dict.values()])
        k = int(all_s.numel() * (1 - sp))
        th = torch.kthvalue(all_s, all_s.numel() - k + 1).values.item() if k > 0 else float("inf")
        return {n: scores_dict[n] >= th for n in eligible_names}

    candidates = {
        "ExCP (ICML '24)": scores_excp,
        "DACP-GlobalRank-85": scores_global_rank,
        "DACP-GlobalRank-80": scores_global_rank_80,
        "DACP-GlobalRank-90": scores_global_rank_90,
        "DACP-Add-0.05": scores_add_005,
        "DACP-Add-0.10": scores_add_01,
        "DACP-Add-0.20": scores_add_02,
        "DACP-Mult-0.10": scores_mult_01,
    }

    sparsities = [0.50, 0.70, 0.80, 0.90, 0.95]
    print(f"\n{'Method':<22} | " + " | ".join(f"{sp:.0%} PPL" for sp in sparsities))
    print("-" * 75)

    for name, s_dict in candidates.items():
        ppls = []
        for sp in sparsities:
            m = get_mask(s_dict, sp)
            p = restore_and_eval(m)
            ppls.append(p)
        ppl_str = " | ".join(f"{p:8.2f}" for p in ppls)
        print(f"{name:<22} | {ppl_str}")

    print("-" * 75)


if __name__ == "__main__":
    main()
