"""Comprehensive Extended Benchmarks on Real Pretrained GPT-2 (Tesla V100 GPU).

Experiments Included:
1. Rate-Distortion Sparsity Sweep:
   Evaluates compression ratios: 50%, 70%, 80%, 90%, 95%
   Comparing: Random vs Magnitude vs First-Order vs Ours (Second-Order HVP).

2. Calibration Batch Sensitivity & Overhead Ablation:
   Evaluates calibration batch count B in [1, 2, 4] at 80% compression.
   Measures HVP computation latency (ms) vs resulting PPL recovery.

3. Structural Layer Sensitivity & Budget Allocation Breakdown:
   Analyzes retention ratio across Attention vs MLP layers under Magnitude vs HVP.

Outputs structured JSON and Markdown summary tables for documentation.
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

SAMPLE_TEXTS = [
    (
        "The Apollo program was the third United States human spaceflight program carried out by the "
        "National Aeronautics and Space Administration, which succeeded in landing the first humans on the Moon. "
        "First conceived during Dwight D. Eisenhower's administration as a three-person spacecraft to follow one-person "
        "Project Mercury, Apollo was later dedicated to President John F. Kennedy's national goal of landing a man on the Moon "
        "and returning him safely to the Earth before the end of the 1960s."
    ),
    (
        "Artificial intelligence was founded as an academic discipline in 1956, and in the years since has experienced several "
        "waves of optimism, followed by disappointment and the loss of funding, followed by new approaches, success and renewed funding. "
        "Machine learning is the study of computer algorithms that improve automatically through experience and by the use of data. "
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks."
    ),
    (
        "The history of computing hardware covers the developments from early simple devices to aid calculation to modern computers. "
        "Before the 20th century, most calculations were done by humans. The first mechanical calculators were invented in the 17th century. "
        "The transition from vacuum tubes to solid-state electronics, such as the silicon transistor, in the late 1950s led to smaller, "
        "cheaper, and faster computers."
    ),
    (
        "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the "
        "scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry, quantum field "
        "theory, quantum technology, and quantum information science. Classical physics cannot explain many phenomena at this scale."
    ),
    (
        "Natural language processing is an interdisciplinary subfield of computer science and linguistics. It is primarily concerned "
        "with giving computers the ability to support and manipulate human language. It involves processing natural language datasets, "
        "such as text corpora or speech datasets, using either rule-based or probabilistic machine learning approaches."
    ),
    (
        "General relativity is a theory of gravitation developed by Albert Einstein between 1907 and 1915, with contributions by many others. "
        "According to general relativity, the observed gravitational effect between masses results from their warping of spacetime. "
        "By the beginning of the 20th century, Newton's law of universal gravitation had been accepted for more than two hundred years."
    ),
    (
        "The European Union is a supranational political and economic union of 27 member states that are located primarily in Europe. "
        "The union has a total area of about four million square kilometers and an estimated total population of over 448 million. "
        "An internal single market has been established through a standardized system of laws that apply in all member states."
    ),
    (
        "Photosynthesis is a biological process used by plants and other organisms to convert light energy into chemical energy that, "
        "through cellular respiration, can later be released to fuel the organism's activities. Some of this chemical energy is stored "
        "in carbohydrate molecules, such as sugars and starches, which are synthesized from carbon dioxide and water."
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Extended Benchmarks on Real GPT-2")
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=str, default="experiments/results/v100_extended_benchmarks.json")
    parser.add_argument("--ft-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_ppl(loss_val: float) -> float:
    try:
        return math.exp(min(loss_val, 20.0))
    except OverflowError:
        return float("inf")


def prepare_dataset(tokenizer, batch_size: int, seq_len: int, device: str):
    full_text = " \n\n ".join(SAMPLE_TEXTS * 6)
    tokens = tokenizer.encode(full_text, return_tensors="pt")[0]
    num_tokens = (tokens.numel() // seq_len) * seq_len
    tokens = tokens[:num_tokens].view(-1, seq_len)

    total_samples = tokens.size(0)
    split_idx = max(4, total_samples // 2)
    train_tokens = tokens[:split_idx]
    eval_tokens = tokens[split_idx:]

    train_batches = [
        train_tokens[i : i + batch_size].to(device)
        for i in range(0, train_tokens.size(0) - batch_size + 1, batch_size)
    ]
    eval_batches = [
        eval_tokens[i : i + batch_size].to(device)
        for i in range(0, eval_tokens.size(0) - batch_size + 1, batch_size)
    ]
    return train_batches, eval_batches


def get_hvp_scores(
    model: nn.Module,
    calib_batches: List[torch.Tensor],
    delta: Dict[str, torch.Tensor],
    eligible_names: List[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
    """Compute averaged gradients and HVP probe on calib_batches."""
    t0 = time.perf_counter()
    named_params = dict(model.named_parameters())
    eligible_params = [named_params[name] for name in eligible_names]

    grad_accum = {name: torch.zeros_like(named_params[name]) for name in eligible_names}
    hvp_accum = {name: torch.zeros_like(named_params[name]) for name in eligible_names}

    with sdpa_kernel(SDPBackend.MATH):
        for batch in calib_batches:
            out = model(input_ids=batch, labels=batch)
            loss = out.loss
            grads = torch.autograd.grad(loss, eligible_params, create_graph=True, retain_graph=True)
            for name, g in zip(eligible_names, grads):
                grad_accum[name] += g.detach()

            # Probe: H * delta
            probe = sum((g * delta[name]).sum() for name, g in zip(eligible_names, grads))
            hvps = torch.autograd.grad(probe, eligible_params, retain_graph=False)
            for name, h in zip(eligible_names, hvps):
                hvp_accum[name] += h.detach()

    b_count = len(calib_batches)
    avg_grads = {name: grad_accum[name] / b_count for name in eligible_names}
    avg_hvps = {name: hvp_accum[name] / b_count for name in eligible_names}
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return avg_grads, avg_hvps, elapsed_ms


def main():
    args = parse_args()
    device = args.device
    print("=" * 90)
    print(f"  Running Extended V100 GPU Benchmarks on Real Pretrained GPT-2")
    print(f"  Model: {args.model_name} | Device: {device}")
    print("=" * 90)

    torch.manual_seed(args.seed)

    # 1. Load Pretrained Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    train_batches, eval_batches = prepare_dataset(tokenizer, args.batch_size, args.seq_len, device)

    def eval_loss(m: nn.Module) -> float:
        m.eval()
        total_loss = 0.0
        with torch.no_grad():
            for b in eval_batches:
                loss = m(input_ids=b, labels=b).loss
                total_loss += loss.item()
        return total_loss / len(eval_batches)

    state_w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    loss_w0 = eval_loss(model)
    ppl_w0 = compute_ppl(loss_w0)
    print(f"W0 (Base Model) -> Loss: {loss_w0:.4f}, PPL: {ppl_w0:.2f}")

    # Fine-tuning adaptation steps to create realistic delta
    print(f"Fine-tuning {args.ft_steps} steps to generate realistic Delta W...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    step = 0
    while step < args.ft_steps:
        for b in train_batches:
            if step >= args.ft_steps:
                break
            optimizer.zero_grad()
            loss = model(input_ids=b, labels=b).loss
            loss.backward()
            optimizer.step()
            step += 1

    model.eval()
    state_w1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    loss_w1 = eval_loss(model)
    ppl_w1 = compute_ppl(loss_w1)
    print(f"W1 (Trained Target) -> Loss: {loss_w1:.4f}, PPL: {ppl_w1:.2f}")

    # Delta definition
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
    total_eligible_params = sum(delta[k].numel() for k in eligible_names)
    print(f"Eligible compressible parameters: {total_eligible_params:,} across {len(eligible_names)} tensors\n")

    # Helper to reconstruct and evaluate
    def evaluate_mask(masks: Dict[str, torch.Tensor]) -> Tuple[float, float, float, float]:
        restored = {}
        for name in state_w1:
            if name in masks:
                m = masks[name].to(delta[name].dtype)
                restored[name] = state_w0[name] + delta[name] * m
            else:
                restored[name] = state_w1[name]
        if "lm_head.weight" in restored and "transformer.wte.weight" in restored:
            restored["lm_head.weight"] = restored["transformer.wte.weight"]

        model.load_state_dict(restored)
        l = eval_loss(model)
        p = compute_ppl(l)
        return l, p, l - loss_w1, p - ppl_w1

    benchmark_records = {
        "metadata": {
            "model": args.model_name,
            "device": device,
            "target_w1_loss": loss_w1,
            "target_w1_ppl": ppl_w1,
            "w0_loss": loss_w0,
            "w0_ppl": ppl_w0,
            "total_params": sum(p.numel() for p in model.parameters()),
            "compressible_params": total_eligible_params,
        },
        "experiment_1_sparsity_sweep": {},
        "experiment_2_calibration_sensitivity": {},
        "experiment_3_layer_allocation": {},
    }

    # =========================================================================
    # EXPERIMENT 1: Sparsity Sweep (Rate-Distortion Curve)
    # =========================================================================
    print("=" * 90)
    print("  [Experiment 1] Sparsity Sweep (Rate-Distortion Curve: 50% -> 95%)")
    print("=" * 90)

    model.load_state_dict(state_w1)
    grads_1b, hvps_1b, hvp_ms = get_hvp_scores(model, [train_batches[0]], delta, eligible_names)

    # Score formulations
    scores_mag = {name: delta[name].abs() for name in eligible_names}
    scores_fo = {name: (grads_1b[name] * delta[name]).abs() for name in eligible_names}
    scores_ours = {
        name: (-grads_1b[name] * delta[name] + 0.5 * delta[name] * hvps_1b[name]).abs()
        for name in eligible_names
    }

    all_mag = torch.cat([s.flatten() for s in scores_mag.values()])
    all_fo = torch.cat([s.flatten() for s in scores_fo.values()])
    all_ours = torch.cat([s.flatten() for s in scores_ours.values()])

    sparsity_levels = [0.50, 0.70, 0.80, 0.90, 0.95]
    sweep_results = []

    for sp in sparsity_levels:
        # Random
        torch.manual_seed(args.seed + int(sp * 100))
        masks_rnd = {}
        for name in eligible_names:
            r = torch.rand_like(delta[name])
            k = int(delta[name].numel() * (1 - sp))
            thresh = torch.kthvalue(r.flatten(), delta[name].numel() - k + 1).values.item() if k > 0 else float("inf")
            masks_rnd[name] = r >= thresh

        # Magnitude
        k_m = int(all_mag.numel() * (1 - sp))
        th_m = torch.kthvalue(all_mag, all_mag.numel() - k_m + 1).values.item() if k_m > 0 else float("inf")
        masks_m = {name: scores_mag[name] >= th_m for name in eligible_names}

        # First Order
        k_fo = int(all_fo.numel() * (1 - sp))
        th_fo = torch.kthvalue(all_fo, all_fo.numel() - k_fo + 1).values.item() if k_fo > 0 else float("inf")
        masks_fo = {name: scores_fo[name] >= th_fo for name in eligible_names}

        # Ours
        k_o = int(all_ours.numel() * (1 - sp))
        th_o = torch.kthvalue(all_ours, all_ours.numel() - k_o + 1).values.item() if k_o > 0 else float("inf")
        masks_o = {name: scores_ours[name] >= th_o for name in eligible_names}

        l_rnd, p_rnd, dl_rnd, dp_rnd = evaluate_mask(masks_rnd)
        l_m, p_m, dl_m, dp_m = evaluate_mask(masks_m)
        l_fo, p_fo, dl_fo, dp_fo = evaluate_mask(masks_fo)
        l_o, p_o, dl_o, dp_o = evaluate_mask(masks_o)

        reduction = (dl_m - dl_o) / dl_m * 100.0 if dl_m > 0 else 0.0

        res_row = {
            "sparsity": sp,
            "random": {"loss": l_rnd, "ppl": p_rnd, "delta_loss": dl_rnd},
            "magnitude": {"loss": l_m, "ppl": p_m, "delta_loss": dl_m},
            "first_order": {"loss": l_fo, "ppl": p_fo, "delta_loss": dl_fo},
            "ours": {"loss": l_o, "ppl": p_o, "delta_loss": dl_o},
            "loss_damage_reduction_pct": reduction,
        }
        sweep_results.append(res_row)
        print(
            f"  Sparsity {sp:.0%} | Mag Loss: {l_m:.4f} (PPL: {p_m:.2f}) | "
            f"Ours Loss: {l_o:.4f} (PPL: {p_o:.2f}) | "
            f"Damage Reduction: {reduction:+.1f}%"
        )

    benchmark_records["experiment_1_sparsity_sweep"] = sweep_results

    # =========================================================================
    # EXPERIMENT 2: Calibration Batch Sensitivity (B in [1, 2, 4])
    # =========================================================================
    print("\n" + "=" * 90)
    print("  [Experiment 2] Calibration Batch Sensitivity (B in [1, 2, 4] @ 80% Sparsity)")
    print("=" * 90)

    calib_results = []
    target_sp = 0.80

    for b_size in [1, 2, 4]:
        sub_batches = train_batches[:b_size]
        model.load_state_dict(state_w1)
        g_b, h_b, elapsed = get_hvp_scores(model, sub_batches, delta, eligible_names)
        sc_b = {
            name: (-g_b[name] * delta[name] + 0.5 * delta[name] * h_b[name]).abs()
            for name in eligible_names
        }
        all_sc_b = torch.cat([s.flatten() for s in sc_b.values()])
        k_b = int(all_sc_b.numel() * (1 - target_sp))
        th_b = torch.kthvalue(all_sc_b, all_sc_b.numel() - k_b + 1).values.item() if k_b > 0 else float("inf")
        mask_b = {name: sc_b[name] >= th_b for name in eligible_names}
        l_b, p_b, dl_b, dp_b = evaluate_mask(mask_b)

        row = {
            "batches": b_size,
            "hvp_time_ms": elapsed,
            "loss": l_b,
            "ppl": p_b,
            "delta_loss": dl_b,
        }
        calib_results.append(row)
        print(f"  B = {b_size:<2} | Scoring Latency: {elapsed:6.1f} ms | Loss: {l_b:.4f} | PPL: {p_b:.2f}")

    benchmark_records["experiment_2_calibration_sensitivity"] = calib_results

    # =========================================================================
    # EXPERIMENT 3: Structural Layer Sensitivity & Budget Allocation Breakdown
    # =========================================================================
    print("\n" + "=" * 90)
    print("  [Experiment 3] Attention vs MLP Structural Budget Allocation (@ 80% Sparsity)")
    print("=" * 90)

    # Use 80% masks computed earlier
    attn_names = [n for n in eligible_names if "attn" in n]
    mlp_names = [n for n in eligible_names if "mlp" in n]

    total_attn_params = sum(delta[n].numel() for n in attn_names)
    total_mlp_params = sum(delta[n].numel() for n in mlp_names)

    # Magnitude distribution
    attn_kept_mag = sum(masks_m[n].sum().item() for n in attn_names)
    mlp_kept_mag = sum(masks_m[n].sum().item() for n in mlp_names)
    attn_ratio_mag = attn_kept_mag / total_attn_params
    mlp_ratio_mag = mlp_kept_mag / total_mlp_params

    # Ours (HVP) distribution
    attn_kept_ours = sum(masks_o[n].sum().item() for n in attn_names)
    mlp_kept_ours = sum(masks_o[n].sum().item() for n in mlp_names)
    attn_ratio_ours = attn_kept_ours / total_attn_params
    mlp_ratio_ours = mlp_kept_ours / total_mlp_params

    layer_alloc = {
        "attn_total_params": total_attn_params,
        "mlp_total_params": total_mlp_params,
        "magnitude": {
            "attn_retention_pct": attn_ratio_mag * 100.0,
            "mlp_retention_pct": mlp_ratio_mag * 100.0,
        },
        "ours_hvp": {
            "attn_retention_pct": attn_ratio_ours * 100.0,
            "mlp_retention_pct": mlp_ratio_ours * 100.0,
        },
    }
    benchmark_records["experiment_3_layer_allocation"] = layer_alloc

    print(f"  Component       | Total Params | Magnitude Kept % | Ours (HVP) Kept % | Allocation Shift")
    print(f"  ----------------+--------------+------------------+-------------------+------------------")
    print(
        f"  Self-Attention  | {total_attn_params:12,d} | {attn_ratio_mag * 100:15.1f}% | {attn_ratio_ours * 100:16.1f}% | "
        f"{(attn_ratio_ours - attn_ratio_mag) * 100:+5.1f}%"
    )
    print(
        f"  Feed-Forward MLP| {total_mlp_params:12,d} | {mlp_ratio_mag * 100:15.1f}% | {mlp_ratio_ours * 100:16.1f}% | "
        f"{(mlp_ratio_ours - mlp_ratio_mag) * 100:+5.1f}%"
    )

    # Save JSON results
    out_path = ROOT / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2)
    print(f"\nSaved structured experimental data to: {out_path}")


if __name__ == "__main__":
    main()
