"""Benchmark and Compare Checkpoint Compression: Ours (DACP / Second-Order HVP) vs ExCP (ICML '24).

Evaluates:
1. Ground Truth / Uncompressed Checkpoint (W1)
2. ExCP (ICML '24):
   - Uses AdamW second moment (exp_avg_sq: m_t)
   - ExCP Criterion: |Delta W| * sqrt(m_t)
3. Magnitude Baseline: |Delta W|
4. First-Order Taylor: |g * Delta W|
5. Ours (DACP / Second-Order HVP Probe):
   - Curvature-aware: |-g * Delta W + 0.5 * Delta W * H * Delta W|

Evaluated across compression ratios: 50%, 80%, 90%, 95% on real GPT-2 fine-tuned on WikiText-2.
Outputs structured JSON and Markdown summary table.
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


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Ours vs ExCP Checkpoint Compression")
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--ft-steps", type=int, default=50, help="Number of fine-tuning steps")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval-batches", type=int, default=30)
    parser.add_argument("--output-json", type=str, default="experiments/results/v100_excp_comparison.json")
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
    print("=" * 90)
    print("  Checkpoint Compression Head-to-Head: Ours (DACP / HVP) vs ExCP (ICML '24)")
    print(f"  Model: {args.model_name} | Device: {device} | Dataset: WikiText-2 | FT Steps: {args.ft_steps}")
    print("=" * 90)

    torch.manual_seed(args.seed)

    # 1. Load Data & Model
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

    print(f"Loading pretrained {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    state_w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}

    val_loss_w0 = evaluate_dataloader(model, val_loader, device, max_batches=args.eval_batches)
    val_ppl_w0 = compute_ppl(val_loss_w0)
    print(f"Base Pretrained W0 -> Val Loss: {val_loss_w0:.4f} | PPL: {val_ppl_w0:.2f}")

    # 2. Fine-tune with AdamW to accumulate real optimizer state (exp_avg, exp_avg_sq)
    print(f"\nFine-tuning for {args.ft_steps} steps to obtain authentic Delta W and AdamW state...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_iter = iter(train_loader)
    
    t0 = time.perf_counter()
    model.train()
    for s in range(1, args.ft_steps + 1):
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

    ft_duration = time.perf_counter() - t0
    model.eval()
    print(f"Fine-tuning completed in {ft_duration:.2f}s ({ft_duration / args.ft_steps * 1000:.1f} ms/step)")

    state_w1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    val_loss_w1 = evaluate_dataloader(model, val_loader, device, max_batches=args.eval_batches)
    val_ppl_w1 = compute_ppl(val_loss_w1)
    print(f"Target Checkpoint W1 (Uncompressed) -> Val Loss: {val_loss_w1:.4f} | PPL: {val_ppl_w1:.2f}\n")

    # 3. Extract Eligible Parameters & Delta
    delta = {k: state_w1[k] - state_w0[k] for k in state_w0}
    named_params = dict(model.named_parameters())
    eligible_names = [
        name for name in delta
        if name in named_params
        and delta[name].dim() >= 2
        and not name.startswith("transformer.wte")
        and not name.startswith("transformer.wpe")
        and not name.startswith("lm_head")
    ]
    eligible_params = [named_params[name] for name in eligible_names]
    total_eligible = sum(delta[n].numel() for n in eligible_names)
    print(f"Eligible compressible parameters: {total_eligible:,} across {len(eligible_names)} weight matrices")

    # 4. Extract ExCP Optimizer States
    # ExCP criterion: Score_i = |Delta W_i| * sqrt(m_t_i), where m_t is AdamW exp_avg_sq
    t_excp_start = time.perf_counter()
    scores_excp = {}
    for name in eligible_names:
        param = named_params[name]
        state = optimizer.state.get(param, {})
        if "exp_avg_sq" in state:
            m_t = state["exp_avg_sq"].to(device)
        else:
            # Fallback to ones if parameter had no gradient updates
            m_t = torch.ones_like(param)
        # ExCP score: |Delta| * sqrt(m_t)
        scores_excp[name] = delta[name].abs() * torch.sqrt(m_t + 1e-8)
    excp_time_ms = (time.perf_counter() - t_excp_start) * 1000.0
    print(f"[ExCP] Computed Adam second-moment scores in {excp_time_ms:.2f} ms")

    # 5. Compute Magnitude and First-Order Scores
    scores_mag = {name: delta[name].abs() for name in eligible_names}

    # Calibration gradients & HVP on 2 calibration batches
    calib_iter = iter(train_loader)
    calib_batches = [next(calib_iter) for _ in range(2)]

    t_hvp_start = time.perf_counter()
    grad_accum = {name: torch.zeros_like(named_params[name]) for name in eligible_names}
    hvp_accum = {name: torch.zeros_like(named_params[name]) for name in eligible_names}

    model.load_state_dict(state_w1)
    with sdpa_kernel(SDPBackend.MATH):
        for cb in calib_batches:
            inp = cb["input_ids"].to(device)
            lbl = cb["labels"].to(device) if "labels" in cb else inp
            loss = model(input_ids=inp, labels=lbl).loss
            grads = torch.autograd.grad(loss, eligible_params, create_graph=True, retain_graph=True)
            for name, g in zip(eligible_names, grads):
                grad_accum[name] += g.detach()

            probe = sum((g * delta[name]).sum() for name, g in zip(eligible_names, grads))
            hvps = torch.autograd.grad(probe, eligible_params, retain_graph=False)
            for name, h in zip(eligible_names, hvps):
                hvp_accum[name] += h.detach()

    avg_grads = {name: grad_accum[name] / len(calib_batches) for name in eligible_names}
    avg_hvps = {name: hvp_accum[name] / len(calib_batches) for name in eligible_names}
    hvp_time_ms = (time.perf_counter() - t_hvp_start) * 1000.0
    print(f"[Ours/DACP] Computed 2-batch HVP curvature in {hvp_time_ms:.1f} ms")

    # First order
    scores_fo = {name: (avg_grads[name] * delta[name]).abs() for name in eligible_names}

    # Ours (DACP): True Taylor damage: dropping Delta_i changes loss by -g*Delta + 0.5*Delta*(H*Delta)
    scores_ours = {
        name: (-avg_grads[name] * delta[name] + 0.5 * delta[name] * avg_hvps[name]).abs()
        for name in eligible_names
    }

    all_mag = torch.cat([s.flatten() for s in scores_mag.values()])
    all_excp = torch.cat([s.flatten() for s in scores_excp.values()])
    all_fo = torch.cat([s.flatten() for s in scores_fo.values()])
    all_ours = torch.cat([s.flatten() for s in scores_ours.values()])

    def reconstruct(masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        restored = {}
        for name in state_w1:
            if name in masks:
                m = masks[name].to(delta[name].dtype)
                restored[name] = state_w0[name] + delta[name] * m
            else:
                restored[name] = state_w1[name]
        if "lm_head.weight" in restored and "transformer.wte.weight" in restored:
            restored["lm_head.weight"] = restored["transformer.wte.weight"]
        return restored

    # 6. Evaluate Head-to-Head across Sparsity Ratios
    sparsity_levels = [0.50, 0.70, 0.80, 0.90, 0.95]
    records = []

    print("\n" + "=" * 100)
    print(f"{'Sparsity':<9} | {'Magnitude PPL':<14} | {'ExCP (ICML24) PPL':<18} | {'First-Order PPL':<16} | {'Ours (DACP) PPL':<16} | {'Ours vs ExCP Advantage'}")
    print("=" * 100)

    for sp in sparsity_levels:
        # Magnitude
        km = int(all_mag.numel() * (1 - sp))
        tm = torch.kthvalue(all_mag, all_mag.numel() - km + 1).values.item() if km > 0 else float("inf")
        m_mag = {name: scores_mag[name] >= tm for name in eligible_names}
        model.load_state_dict(reconstruct(m_mag))
        l_mag = evaluate_dataloader(model, val_loader, device, max_batches=args.eval_batches)
        p_mag = compute_ppl(l_mag)

        # ExCP
        k_ex = int(all_excp.numel() * (1 - sp))
        t_ex = torch.kthvalue(all_excp, all_excp.numel() - k_ex + 1).values.item() if k_ex > 0 else float("inf")
        m_excp = {name: scores_excp[name] >= t_ex for name in eligible_names}
        model.load_state_dict(reconstruct(m_excp))
        l_excp = evaluate_dataloader(model, val_loader, device, max_batches=args.eval_batches)
        p_excp = compute_ppl(l_excp)

        # First Order
        k_fo = int(all_fo.numel() * (1 - sp))
        t_fo = torch.kthvalue(all_fo, all_fo.numel() - k_fo + 1).values.item() if k_fo > 0 else float("inf")
        m_fo = {name: scores_fo[name] >= t_fo for name in eligible_names}
        model.load_state_dict(reconstruct(m_fo))
        l_fo = evaluate_dataloader(model, val_loader, device, max_batches=args.eval_batches)
        p_fo = compute_ppl(l_fo)

        # Ours
        k_o = int(all_ours.numel() * (1 - sp))
        t_o = torch.kthvalue(all_ours, all_ours.numel() - k_o + 1).values.item() if k_o > 0 else float("inf")
        m_ours = {name: scores_ours[name] >= t_o for name in eligible_names}
        model.load_state_dict(reconstruct(m_ours))
        l_ours = evaluate_dataloader(model, val_loader, device, max_batches=args.eval_batches)
        p_ours = compute_ppl(l_ours)

        ppl_adv = p_excp - p_ours
        loss_adv = l_excp - l_ours

        tag = f"★ Ours Lower PPL by {ppl_adv:.2f}" if ppl_adv > 0 else f"ExCP Lower by {-ppl_adv:.2f}"
        print(f"{sp:<9.0%} | {p_mag:<14.2f} | {p_excp:<18.2f} | {p_fo:<16.2f} | {p_ours:<16.2f} | {tag}")

        records.append({
            "sparsity": sp,
            "magnitude": {"loss": l_mag, "ppl": p_mag},
            "excp": {"loss": l_excp, "ppl": p_excp},
            "first_order": {"loss": l_fo, "ppl": p_fo},
            "ours_dacp": {"loss": l_ours, "ppl": p_ours},
            "ours_ppl_advantage_over_excp": ppl_adv,
            "ours_loss_advantage_over_excp": loss_adv,
        })

    print("=" * 100)
    print(f"Uncompressed Target W1: Loss = {val_loss_w1:.4f} | PPL = {val_ppl_w1:.2f}\n")

    # 7. Output JSON
    output_data = {
        "metadata": {
            "model": args.model_name,
            "dataset": "WikiText-2",
            "fine_tuning_steps": args.ft_steps,
            "val_loss_w1_uncompressed": val_loss_w1,
            "val_ppl_w1_uncompressed": val_ppl_w1,
            "val_loss_w0_base": val_loss_w0,
            "val_ppl_w0_base": val_ppl_w0,
            "excp_scoring_ms": excp_time_ms,
            "ours_hvp_scoring_ms": hvp_time_ms,
        },
        "results": records,
    }

    out_path = ROOT / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Comparison data saved to: {out_path}\n")


if __name__ == "__main__":
    main()
