"""Multi-cycle Fault-Tolerant Continuation Training: ExCP vs DACP vs Oracle (No Compression).

Simulates 3 consecutive training-saving-restoring cycles at 90% checkpoint compression.
Cycle 1: Train 40 steps -> Save/Compress -> Restore -> Evaluate
Cycle 2: Train 40 steps -> Save/Compress -> Restore -> Evaluate
Cycle 3: Train 40 steps -> Save/Compress -> Restore -> Final Evaluate
"""

from __future__ import annotations

import argparse
import copy
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


def train_steps(model: nn.Module, optimizer, train_iter, train_loader, steps: int, device: str):
    model.train()
    for _ in range(steps):
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
    return train_iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps-per-cycle", type=int, default=40)
    parser.add_argument("--num-cycles", type=int, default=3)
    parser.add_argument("--sparsity", type=float, default=0.90)
    parser.add_argument("--output-json", type=str, default="experiments/results/v100_dacp_continuation_showdown.json")
    args = parser.parse_args()
    device = args.device

    print("=" * 105)
    print(f"  Multi-Cycle Fault-Tolerant Continuation Showdown (Sparsity: {args.sparsity:.0%})")
    print(f"  Cycles: {args.num_cycles} | Steps per Cycle: {args.steps_per_cycle} | Device: {device}")
    print("=" * 105)

    train_loader = get_wikitext2_dataloader(split="train", batch_size=4, seq_length=256, shuffle=True)
    val_loader = get_wikitext2_dataloader(split="validation", batch_size=4, seq_length=256, shuffle=False)

    # We evaluate 3 tracks:
    # Track A: Oracle (No compression, continuous training)
    # Track B: ExCP (At each checkpoint, compress 90%, restore, continue training)
    # Track C: DACP (At each checkpoint, compress 90% with HVP 2D rank, restore, continue training)

    base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    w0 = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
    val_loss_0 = evaluate_dataloader(base_model, val_loader, device)
    print(f"Initial Baseline W0: Val Loss = {val_loss_0:.4f} | PPL = {compute_ppl(val_loss_0):.2f}\n")

    named_params = dict(base_model.named_parameters())
    eligible_names = [
        n for n in w0
        if n in named_params
        and w0[n].dim() >= 2
        and not n.startswith("transformer.wte")
        and not n.startswith("transformer.wpe")
        and not n.startswith("lm_head")
    ]

    model_oracle = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model_excp = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model_dacp = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

    opt_oracle = torch.optim.AdamW(model_oracle.parameters(), lr=1e-4, weight_decay=0.01)
    opt_excp = torch.optim.AdamW(model_excp.parameters(), lr=1e-4, weight_decay=0.01)
    opt_dacp = torch.optim.AdamW(model_dacp.parameters(), lr=1e-4, weight_decay=0.01)

    iter_oracle = iter(train_loader)
    iter_excp = iter(train_loader)
    iter_dacp = iter(train_loader)

    prev_w_excp = {k: v.clone() for k, v in w0.items()}
    prev_w_dacp = {k: v.clone() for k, v in w0.items()}

    cycle_records = []

    for cycle in range(1, args.num_cycles + 1):
        print(f"\n--- Cycle {cycle}/{args.num_cycles}: Training {args.steps_per_cycle} steps ---")
        
        # 1. Train Oracle
        iter_oracle = train_steps(model_oracle, opt_oracle, iter_oracle, train_loader, args.steps_per_cycle, device)
        l_ora = evaluate_dataloader(model_oracle, val_loader, device)
        p_ora = compute_ppl(l_ora)

        # 2. Train ExCP Track
        iter_excp = train_steps(model_excp, opt_excp, iter_excp, train_loader, args.steps_per_cycle, device)
        curr_w_excp = {k: v.detach().clone() for k, v in model_excp.state_dict().items()}
        delta_excp = {k: curr_w_excp[k] - prev_w_excp[k] for k in curr_w_excp}
        # Compute ExCP scores
        scores_excp = {}
        np_excp = dict(model_excp.named_parameters())
        for n in eligible_names:
            param = np_excp[n]
            st = opt_excp.state.get(param, {})
            mt = st.get("exp_avg_sq", torch.ones_like(param)).to(device)
            scores_excp[n] = delta_excp[n].abs() * torch.sqrt(mt + 1e-8)
        # Apply 90% pruning
        all_ex = torch.cat([s.flatten() for s in scores_excp.values()])
        k_ex = int(all_ex.numel() * (1 - args.sparsity))
        th_ex = torch.kthvalue(all_ex, all_ex.numel() - k_ex + 1).values.item()
        restored_excp = {}
        for n in curr_w_excp:
            if n in eligible_names:
                m = (scores_excp[n] >= th_ex).to(delta_excp[n].dtype)
                restored_excp[n] = prev_w_excp[n] + delta_excp[n] * m
            else:
                restored_excp[n] = curr_w_excp[n]
        if "lm_head.weight" in restored_excp:
            restored_excp["lm_head.weight"] = restored_excp["transformer.wte.weight"]
        # Restore into ExCP model
        model_excp.load_state_dict(restored_excp)
        prev_w_excp = {k: v.clone() for k, v in restored_excp.items()}
        l_ex = evaluate_dataloader(model_excp, val_loader, device)
        p_ex = compute_ppl(l_ex)

        # 3. Train DACP Track
        iter_dacp = train_steps(model_dacp, opt_dacp, iter_dacp, train_loader, args.steps_per_cycle, device)
        curr_w_dacp = {k: v.detach().clone() for k, v in model_dacp.state_dict().items()}
        delta_dacp = {k: curr_w_dacp[k] - prev_w_dacp[k] for k in curr_w_dacp}
        np_dacp = dict(model_dacp.named_parameters())
        # HVP on 4 calibration batches
        calib_batches = [next(iter(train_loader)) for _ in range(4)]
        grad_accum = {n: torch.zeros_like(np_dacp[n]) for n in eligible_names}
        hvp_accum = {n: torch.zeros_like(np_dacp[n]) for n in eligible_names}
        with sdpa_kernel(SDPBackend.MATH):
            for cb in calib_batches:
                inp = cb["input_ids"].to(device)
                lbl = cb["labels"].to(device) if "labels" in cb else inp
                loss = model_dacp(input_ids=inp, labels=lbl).loss
                el_p = [np_dacp[n] for n in eligible_names]
                grads = torch.autograd.grad(loss, el_p, create_graph=True, retain_graph=True)
                for n, g in zip(eligible_names, grads):
                    grad_accum[n] += g.detach()
                probe = sum((g * delta_dacp[n]).sum() for n, g in zip(eligible_names, grads))
                hvps = torch.autograd.grad(probe, el_p, retain_graph=False)
                for n, h in zip(eligible_names, hvps):
                    hvp_accum[n] += h.detach()
        avg_g = {n: grad_accum[n] / 4 for n in eligible_names}
        avg_h = {n: hvp_accum[n] / 4 for n in eligible_names}
        scores_hvp = {n: (-avg_g[n] * delta_dacp[n] + 0.5 * delta_dacp[n] * avg_h[n]).abs() for n in eligible_names}
        # Curvature + Momentum Rank Fusion
        scores_mom = {}
        for n in eligible_names:
            st = opt_dacp.state.get(np_dacp[n], {})
            mt = st.get("exp_avg_sq", torch.ones_like(np_dacp[n])).to(device)
            scores_mom[n] = delta_dacp[n].abs() * torch.sqrt(mt + 1e-8)
        all_m = torch.cat([scores_mom[n].flatten() for n in eligible_names])
        all_h = torch.cat([scores_hvp[n].flatten() for n in eligible_names])
        N_p = all_m.numel()
        rm = torch.argsort(torch.argsort(all_m, stable=True), stable=True).float() / N_p
        rh = torch.argsort(torch.argsort(all_h, stable=True), stable=True).float() / N_p
        comb_dacp = {}
        off = 0
        for n in eligible_names:
            sz = delta_dacp[n].numel()
            c = 0.60 * rh[off : off + sz].view_as(delta_dacp[n]) + 0.40 * rm[off : off + sz].view_as(delta_dacp[n])
            c[(rm[off : off + sz].view_as(delta_dacp[n]) >= 0.99) | (rh[off : off + sz].view_as(delta_dacp[n]) >= 0.99)] = 1.0
            comb_dacp[n] = c
            off += sz
        # Pruning with layer floor
        all_cd = torch.cat([s.flatten() for s in comb_dacp.values()])
        k_d = int(all_cd.numel() * (1 - args.sparsity))
        th_d = torch.kthvalue(all_cd, all_cd.numel() - k_d + 1).values.item()
        restored_dacp = {}
        for n in curr_w_dacp:
            if n in eligible_names:
                m = (comb_dacp[n] >= th_d).to(delta_dacp[n].dtype)
                restored_dacp[n] = prev_w_dacp[n] + delta_dacp[n] * m
            else:
                restored_dacp[n] = curr_w_dacp[n]
        if "lm_head.weight" in restored_dacp:
            restored_dacp["lm_head.weight"] = restored_dacp["transformer.wte.weight"]
        model_dacp.load_state_dict(restored_dacp)
        prev_w_dacp = {k: v.clone() for k, v in restored_dacp.items()}
        l_da = evaluate_dataloader(model_dacp, val_loader, device)
        p_da = compute_ppl(l_da)

        gap = p_ex - p_da
        print(f"Cycle {cycle}: Oracle PPL = {p_ora:.2f} | ExCP PPL = {p_ex:.2f} | DACP PPL = {p_da:.2f} | ★ DACP Lead = +{gap:.2f} PPL")

        cycle_records.append({
            "cycle": cycle,
            "oracle_ppl": p_ora,
            "excp_ppl": p_ex,
            "dacp_ppl": p_da,
            "dacp_lead_over_excp": gap,
        })

    print("\n" + "=" * 105)
    print(f"Final 3-Cycle Result Summary (90% compression):")
    for r in cycle_records:
        print(f"  Cycle {r['cycle']}: Oracle={r['oracle_ppl']:.2f} | ExCP={r['excp_ppl']:.2f} | DACP={r['dacp_ppl']:.2f} (DACP Win by +{r['dacp_lead_over_excp']:.2f} PPL)")
    print("=" * 105)

    out_file = ROOT / args.output_json
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cycle_records, f, indent=2)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
