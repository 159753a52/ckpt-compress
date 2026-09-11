"""Tuned DACP (Layer-wise Block Taylor HVP) vs ExCP on GPT-2 Fine-Tuning.

Features:
1. Blockwise Taylor-HVP scoring (layer-isolated, zero cross-layer gradient noise).
2. Proper Layer-Wise / Tensor-Wise structural budget allocation (preventing layer starvation).
3. Signed Taylor ranking: retains parameters with highest positive predicted loss contribution.
4. Compare head-to-head against ExCP (ICML '24) across 50%, 70%, 80%, 90%, 95% sparsity.
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


def parse_args():
    parser = argparse.ArgumentParser(description="Tuned DACP vs ExCP")
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--ft-steps", type=int, default=50)
    parser.add_argument("--hvp-batches", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval-batches", type=int, default=30)
    parser.add_argument("--output-json", type=str, default="experiments/results/v100_dacp_vs_excp_tuned.json")
    parser.add_argument("--seed", type=int, default=42)
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
    print("=" * 95)
    print("  Optimized DACP (Block Taylor HVP + Layer Allocation) vs ExCP (ICML '24)")
    print(f"  Model: {args.model_name} | Device: {device} | FT Steps: {args.ft_steps} | HVP Batches: {args.hvp_batches}")
    print("=" * 95)

    torch.manual_seed(args.seed)

    # 1. Load Data
    train_loader = get_wikitext2_dataloader(
        split="train",
        batch_size=args.batch_size,
        seq_length=args.seq_len,
        shuffle=True,
    )
    val_loader = get_wikitext2_dataloader(
        split="validation",
        batch_size=args.batch_size,
        seq_length=args.seq_len,
        shuffle=False,
    )

    # 2. Load Model & Reference W0
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    state_w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    val_loss_w0 = evaluate_dataloader(model, val_loader, device, max_batches=args.eval_batches)
    print(f"Base Pretrained W0 -> Val Loss: {val_loss_w0:.4f} | PPL: {compute_ppl(val_loss_w0):.2f}")

    # 3. Fine-tuning with AdamW
    print(f"\nFine-tuning for {args.ft_steps} steps to collect authentic Delta and Adam states...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_iter = iter(train_loader)
    
    t0 = time.perf_counter()
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
    print(f"Fine-tuning finished in {time.perf_counter() - t0:.2f}s")

    model.eval()
    state_w1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    val_loss_w1 = evaluate_dataloader(model, val_loader, device, max_batches=args.eval_batches)
    val_ppl_w1 = compute_ppl(val_loss_w1)
    print(f"Uncompressed Target W1 -> Val Loss: {val_loss_w1:.4f} | PPL: {val_ppl_w1:.2f}\n")

    # 4. Extract Eligible Layers
    delta = {k: state_w1[k] - state_w0[k] for k in state_w0}
    layers = eligible_layers(model, "gpt2")
    eligible_names = [name for layer in layers for name in layer]
    named_params = dict(model.named_parameters())
    total_eligible = sum(delta[n].numel() for n in eligible_names)
    print(f"Structural layers: {len(layers)} blocks, {len(eligible_names)} tensors ({total_eligible:,} parameters)")

    # 5. Method 1: ExCP (ICML '24)
    # ExCP computes score = |Delta| * sqrt(m_t), with tensor-wise normalization:
    t_excp = time.perf_counter()
    scores_excp = {}
    for name in eligible_names:
        param = named_params[name]
        state = optimizer.state.get(param, {})
        m_t = state.get("exp_avg_sq", torch.ones_like(param)).to(device)
        scores_excp[name] = (delta[name].abs() * torch.sqrt(m_t + 1e-8)).cpu()
    excp_ms = (time.perf_counter() - t_excp) * 1000.0
    print(f"[ExCP] Computed scores in {excp_ms:.2f} ms")

    # 6. Method 2: DACP (Blockwise Taylor HVP with Signed Contribution)
    print("\nComputing DACP Blockwise Taylor HVP scores across structural layers...")
    scoring_batches = [next(iter(train_loader)) for _ in range(args.hvp_batches)]
    # Convert batches to format expected by compute_block_taylor_scores
    formatted_batches = []
    for b in scoring_batches:
        inp = b["input_ids"].to(device)
        lbl = b["labels"].to(device) if "labels" in b else inp
        formatted_batches.append({"input_ids": inp, "labels": lbl})

    t_dacp = time.perf_counter()
    dacp_scores, dacp_metrics = compute_block_taylor_scores(
        model=model,
        batches=formatted_batches,
        layers=layers,
        delta=delta,
        device=device,
        return_components=False,
        aggregation="signed_mean",
        task_type="lm",
        model_family="gpt2",
    )
    dacp_ms = (time.perf_counter() - t_dacp) * 1000.0
    print(f"[DACP] Blockwise Taylor HVP computed in {dacp_ms:.1f} ms")

    # Reconstruct state helper
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
        loss = evaluate_dataloader(model, val_loader, device, max_batches=args.eval_batches)
        return loss, compute_ppl(loss)

    # 7. Evaluate across Sparsity Ratios
    # Compare:
    # A) ExCP (ICML '24)
    # B) DACP Uniform Per-Layer (Signed Taylor)
    # C) DACP Global (Signed Taylor)
    # D) Baseline Magnitude Uniform
    sparsity_levels = [0.50, 0.70, 0.80, 0.90, 0.95]
    records = []

    print("\n" + "=" * 105)
    print(f"{'Sparsity':<9} | {'Magnitude PPL':<15} | {'ExCP (ICML24) PPL':<18} | {'DACP (Ours) PPL':<18} | {'DACP vs ExCP Advantage'}")
    print("=" * 105)

    for sp in sparsity_levels:
        # A) ExCP
        # ExCP does thresholding per tensor or across all:
        all_ex = torch.cat([s.flatten() for s in scores_excp.values()])
        k_ex = int(all_ex.numel() * (1 - sp))
        th_ex = torch.kthvalue(all_ex, all_ex.numel() - k_ex + 1).values.item() if k_ex > 0 else float("inf")
        masks_excp = {name: (scores_excp[name] >= th_ex).to(device) for name in eligible_names}
        loss_excp, ppl_excp = restore_and_eval(masks_excp)

        # B) Magnitude Uniform Per-Layer
        masks_mag_uni = {}
        for layer in layers:
            l_scores = torch.cat([delta[n].abs().cpu().flatten() for n in layer])
            prune_count = int(round(l_scores.numel() * sp))
            l_keep = exact_keep_mask(l_scores, prune_count)
            offset = 0
            for n in layer:
                sz = delta[n].numel()
                masks_mag_uni[n] = l_keep[offset : offset + sz].view_as(delta[n]).to(device)
                offset += sz
        loss_mag, ppl_mag = restore_and_eval(masks_mag_uni)

        # C) DACP Layer-Wise Block Taylor (Signed: prune smallest signed scores first)
        # ascending order prunes coordinates whose reversion is predicted to reduce loss first
        masks_dacp = {}
        for layer in layers:
            l_scores = torch.cat([dacp_scores[n].cpu().flatten() for n in layer])
            prune_count = int(round(l_scores.numel() * sp))
            l_keep = exact_keep_mask(l_scores, prune_count)
            offset = 0
            for n in layer:
                sz = dacp_scores[n].numel()
                masks_dacp[n] = l_keep[offset : offset + sz].view_as(dacp_scores[n]).to(device)
                offset += sz
        loss_dacp, ppl_dacp = restore_and_eval(masks_dacp)

        # Advantage
        ppl_adv = ppl_excp - ppl_dacp
        loss_adv = loss_excp - loss_dacp
        if ppl_adv > 0:
            status = f"★ DACP Lower PPL by +{ppl_adv:.2f} (WIN)"
        elif abs(ppl_adv) < 0.05:
            status = f"Tie (DACP: {ppl_dacp:.2f} vs ExCP: {ppl_excp:.2f})"
        else:
            status = f"ExCP Lower by {-ppl_adv:.2f}"

        print(f"{sp:<9.0%} | {ppl_mag:<15.2f} | {ppl_excp:<18.2f} | {ppl_dacp:<18.2f} | {status}")

        records.append({
            "sparsity": sp,
            "magnitude": {"loss": loss_mag, "ppl": ppl_mag},
            "excp": {"loss": loss_excp, "ppl": ppl_excp},
            "dacp_ours": {"loss": loss_dacp, "ppl": ppl_dacp},
            "dacp_ppl_advantage": ppl_adv,
            "dacp_loss_advantage": loss_adv,
        })

    print("=" * 105)
    print(f"Uncompressed Baseline W1: Loss = {val_loss_w1:.4f} | PPL = {val_ppl_w1:.2f}\n")

    out_file = ROOT / args.output_json
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"metadata": {"model": args.model_name, "ft_steps": args.ft_steps}, "results": records}, f, indent=2)
    print(f"Saved results to: {out_file}")


if __name__ == "__main__":
    main()
