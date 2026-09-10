"""Real Fine-Tuning and Checkpoint Compression Experiment on GPT-2 with WikiText-2.

Evaluates:
1. Real fine-tuning trajectory of GPT-2 (124M) on WikiText-2 (100 steps).
2. Checkpoint compression accuracy at step 100 under 50%, 80%, and 90% sparsity:
   - Random Baseline
   - Magnitude Pruning (|Delta|)
   - First-Order Taylor (|g * Delta|)
   - Ours (Second-Order HVP Probe: |-g * Delta + 0.5 * Delta * H * Delta|)
3. Resumed Training Recovery (Step 100 -> Step 150):
   - Continues fine-tuning from:
     (a) Uncompressed checkpoint (ideal upper bound)
     (b) 80% compressed checkpoint via Magnitude
     (c) 80% compressed checkpoint via Ours (HVP)
   - Measures whether the model trajectory heals and converges.

Outputs structured JSON and Markdown summary report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    parser = argparse.ArgumentParser(description="Real Fine-Tuning Checkpoint Compression on WikiText-2")
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--total-steps", type=int, default=100, help="Total initial fine-tuning steps")
    parser.add_argument("--resume-steps", type=int, default=50, help="Steps to train after resumption")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval-batches", type=int, default=30, help="Number of validation batches for fast eval")
    parser.add_argument("--output-json", type=str, default="experiments/results/v100_finetune_recovery.json")
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
    print("  Real GPT-2 Fine-Tuning & Checkpoint Compression on WikiText-2")
    print(f"  Model: {args.model_name} | Device: {device} | Seq-Len: {args.seq_len} | Batch Size: {args.batch_size}")
    print("=" * 90)

    torch.manual_seed(args.seed)

    # 1. Load Data
    print("Loading WikiText-2 train & validation data loaders...")
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
    print(f"Loaded: Train batches = {len(train_loader)}, Val batches = {len(val_loader)}")

    # 2. Load Model
    print(f"Loading pretrained base model '{args.model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    
    # Evaluate initial pretrained state W0
    initial_val_loss = evaluate_dataloader(model, val_loader, device, max_batches=args.eval_batches)
    initial_val_ppl = compute_ppl(initial_val_loss)
    print(f"[Pretrained Base W0] Initial Val Loss: {initial_val_loss:.4f} | PPL: {initial_val_ppl:.2f}\n")

    state_w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 3. Fine-Tuning Phase 1: Train to step 100
    print(f"Starting Phase 1: Fine-tuning for {args.total_steps} optimizer steps...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_iter = iter(train_loader)
    
    t_start = time.perf_counter()
    model.train()
    step_losses = []
    
    for step in range(1, args.total_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device) if "labels" in batch else input_ids
        
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        optimizer.step()
        step_losses.append(loss.item())

        if step % 25 == 0 or step == args.total_steps:
            print(f"  Step {step:3d}/{args.total_steps} | Train Loss: {loss.item():.4f} | Avg (last 25): {sum(step_losses[-25:]) / len(step_losses[-25:]):.4f}")

    train_time = time.perf_counter() - t_start
    print(f"Phase 1 completed in {train_time:.2f}s ({train_time / args.total_steps * 1000:.1f} ms/step)!")

    # Checkpoint W_100
    state_w100 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    val_loss_w100 = evaluate_dataloader(model, val_loader, device, max_batches=args.eval_batches)
    val_ppl_w100 = compute_ppl(val_loss_w100)
    print(f"\n[Checkpoint W100 (Uncompressed)] Val Loss: {val_loss_w100:.4f} | PPL: {val_ppl_w100:.2f} (Loss drop: {val_loss_w100 - initial_val_loss:.4f})")

    # 4. Checkpoint Compression & Scoring Evaluation
    print("\n" + "=" * 90)
    print("  Evaluating Checkpoint Compression at Step 100 (WikiText-2 Validation)")
    print("=" * 90)

    delta = {k: state_w100[k] - state_w0[k] for k in state_w0}
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
    print(f"Compressible tensors: {len(eligible_names)} ({sum(delta[n].numel() for n in eligible_names):,} parameters)")

    # Compute calibration gradients and HVP on 2 calibration batches
    calib_iter = iter(train_loader)
    calib_batches = [next(calib_iter) for _ in range(2)]
    
    t_hvp_start = time.perf_counter()
    grad_accum = {name: torch.zeros_like(named_params[name]) for name in eligible_names}
    hvp_accum = {name: torch.zeros_like(named_params[name]) for name in eligible_names}

    model.load_state_dict(state_w100)
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
    hvp_duration_ms = (time.perf_counter() - t_hvp_start) * 1000.0
    print(f"HVP Curvature calculation completed in {hvp_duration_ms:.1f} ms on 2 batches!")

    # Scores
    scores_mag = {name: delta[name].abs() for name in eligible_names}
    scores_fo = {name: (avg_grads[name] * delta[name]).abs() for name in eligible_names}
    scores_ours = {
        name: (-avg_grads[name] * delta[name] + 0.5 * delta[name] * avg_hvps[name]).abs()
        for name in eligible_names
    }

    all_mag = torch.cat([s.flatten() for s in scores_mag.values()])
    all_fo = torch.cat([s.flatten() for s in scores_fo.values()])
    all_ours = torch.cat([s.flatten() for s in scores_ours.values()])

    def reconstruct_state(masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        restored = {}
        for name in state_w100:
            if name in masks:
                m = masks[name].to(delta[name].dtype)
                restored[name] = state_w0[name] + delta[name] * m
            else:
                restored[name] = state_w100[name]
        if "lm_head.weight" in restored and "transformer.wte.weight" in restored:
            restored["lm_head.weight"] = restored["transformer.wte.weight"]
        return restored

    # Evaluate across 50%, 80%, 90%
    compression_results = []
    masks_80_mag = None
    masks_80_ours = None

    for sp in [0.50, 0.80, 0.90]:
        # Mag
        km = int(all_mag.numel() * (1 - sp))
        tm = torch.kthvalue(all_mag, all_mag.numel() - km + 1).values.item() if km > 0 else float("inf")
        m_mag = {name: scores_mag[name] >= tm for name in eligible_names}

        # First-order
        kfo = int(all_fo.numel() * (1 - sp))
        tfo = torch.kthvalue(all_fo, all_fo.numel() - kfo + 1).values.item() if kfo > 0 else float("inf")
        m_fo = {name: scores_fo[name] >= tfo for name in eligible_names}

        # Ours
        ko = int(all_ours.numel() * (1 - sp))
        to = torch.kthvalue(all_ours, all_ours.numel() - ko + 1).values.item() if ko > 0 else float("inf")
        m_ours = {name: scores_ours[name] >= to for name in eligible_names}

        if sp == 0.80:
            masks_80_mag = m_mag
            masks_80_ours = m_ours

        # Evaluate Mag
        model.load_state_dict(reconstruct_state(m_mag))
        loss_mag = evaluate_dataloader(model, val_loader, device, max_batches=args.eval_batches)
        ppl_mag = compute_ppl(loss_mag)

        # Evaluate FO
        model.load_state_dict(reconstruct_state(m_fo))
        loss_fo = evaluate_dataloader(model, val_loader, device, max_batches=args.eval_batches)
        ppl_fo = compute_ppl(loss_fo)

        # Evaluate Ours
        model.load_state_dict(reconstruct_state(m_ours))
        loss_ours = evaluate_dataloader(model, val_loader, device, max_batches=args.eval_batches)
        ppl_ours = compute_ppl(loss_ours)

        dl_mag = loss_mag - val_loss_w100
        dl_ours = loss_ours - val_loss_w100
        reduction = (dl_mag - dl_ours) / dl_mag * 100.0 if dl_mag > 0 else 0.0

        print(f"  Sparsity {sp:.0%} | Magnitude Loss: {loss_mag:.4f} (PPL: {ppl_mag:.2f}) | Ours Loss: {loss_ours:.4f} (PPL: {ppl_ours:.2f}) | Damage Reduction: {reduction:+.1f}%")
        compression_results.append({
            "sparsity": sp,
            "magnitude": {"loss": loss_mag, "ppl": ppl_mag, "delta_loss": dl_mag},
            "first_order": {"loss": loss_fo, "ppl": ppl_fo, "delta_loss": loss_fo - val_loss_w100},
            "ours": {"loss": loss_ours, "ppl": ppl_ours, "delta_loss": dl_ours},
            "loss_damage_reduction_pct": reduction,
        })

    # 5. Resumed Fine-Tuning from Checkpoint (Step 100 -> Step 150)
    print("\n" + "=" * 90)
    print(f"  Phase 2: Resumed Training from Step 100 to Step 150 ({args.resume_steps} steps)")
    print("  Testing Trajectory Healing: Uncompressed vs 80% Magnitude vs 80% Ours")
    print("=" * 90)

    # Function to train N steps from a given starting state dict
    def train_resumed(init_state: Dict[str, torch.Tensor], label: str) -> Tuple[float, float, List[float]]:
        model.load_state_dict(init_state)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        
        # Reset iterator with fixed seed so all strategies see identical data batches
        torch.manual_seed(args.seed + 999)
        resumed_loader = get_wikitext2_dataloader(
            split="train",
            batch_size=args.batch_size,
            seq_length=args.seq_len,
            shuffle=False,
        )
        r_iter = iter(resumed_loader)
        
        model.train()
        losses = []
        for s in range(1, args.resume_steps + 1):
            b = next(r_iter)
            inp = b["input_ids"].to(device)
            lbl = b["labels"].to(device) if "labels" in b else inp
            opt.zero_grad()
            l = model(input_ids=inp, labels=lbl).loss
            l.backward()
            opt.step()
            losses.append(l.item())

        final_loss = evaluate_dataloader(model, val_loader, device, max_batches=args.eval_batches)
        final_ppl = compute_ppl(final_loss)
        print(f"  [{label:<28}] Final Step 150 Val Loss: {final_loss:.4f} | PPL: {final_ppl:.2f}")
        return final_loss, final_ppl, losses

    res_uncompressed = train_resumed(state_w100, "Uncompressed (Upper Bound)")
    res_mag = train_resumed(reconstruct_state(masks_80_mag), "80% Magnitude Checkpoint")
    res_ours = train_resumed(reconstruct_state(masks_80_ours), "80% Ours (HVP) Checkpoint")

    # 6. Save Structured Results
    record = {
        "metadata": {
            "model": args.model_name,
            "dataset": "Salesforce/wikitext-2-raw-v1",
            "phase1_steps": args.total_steps,
            "phase2_resume_steps": args.resume_steps,
            "initial_w0_val_loss": initial_val_loss,
            "initial_w0_val_ppl": initial_val_ppl,
            "step100_uncompressed_val_loss": val_loss_w100,
            "step100_uncompressed_val_ppl": val_ppl_w100,
            "hvp_calculation_ms": hvp_duration_ms,
        },
        "step100_compression_sweep": compression_results,
        "step150_resumption_recovery": {
            "uncompressed": {"loss": res_uncompressed[0], "ppl": res_uncompressed[1]},
            "magnitude_80pct": {"loss": res_mag[0], "ppl": res_mag[1]},
            "ours_hvp_80pct": {"loss": res_ours[0], "ppl": res_ours[1]},
            "ppl_advantage_over_magnitude": res_mag[1] - res_ours[1],
        },
    }

    out_file = ROOT / args.output_json
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"\nExperimental results successfully saved to: {out_file}\n")


if __name__ == "__main__":
    main()
