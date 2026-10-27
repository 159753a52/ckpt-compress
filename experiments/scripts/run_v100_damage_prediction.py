"""Predicted versus measured deletion damage of DACP-G and DACP-H masks.

For each deletion rate, a mask is selected by staged rescoring (first-order
score with the variance penalty, as DACP-G, or the Taylor score, as DACP-H).
The damage of the final mask, e = -z*d, is then predicted at the checkpoint on
scoring batches by the first-order model g^T e (what DACP-G's score contains)
and by the Taylor model g^T e + 0.5 e^T H_bd e with layer-wise HVPs (what
DACP-H's score contains), and measured as the loss increase on the
training-calibration batches and on the held-out validation set.
"""
from pathlib import Path
import argparse
import json
import math
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sparsities", default="0.5,0.6,0.7,0.8,0.84")
    ap.add_argument("--config", default="5:8:2", help="K:B:kappa of the selection")
    ap.add_argument("--select", choices=("first_order", "taylor"), default="first_order")
    ap.add_argument("--predict-batches", type=int, default=16)
    args = ap.parse_args()
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from run_residual_sweep_small import REGIMES, load_regime, TINY
    from run_residual_ablation_small import geometric_schedule, grad_moments, run_iterative, _masked_delta
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from experiments.lib.residual_runtime import evaluate_lm, lm_loss, set_seed, write_json
    from experiments.lib.residual_scoring import (_restore_scoring_state, _snapshot_scoring_state,
                                                  _structural_blocks_for_scoring)

    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    ctx = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    sweep, pool = ctx["sweep"], ctx["pool"]
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, 42)
    initial = plan.scoring_batches[0]
    used = plan.selected_pool_indices[len(plan.training_segments[0]):][:16]
    gate_indices = [i for i in range(len(pool)) if i not in set(used)][:64]
    extra = [i for i in range(len(pool)) if i not in set(used + gate_indices)]
    random.Random(42).shuffle(extra)
    scoring = initial + [pool[i] for i in extra[:48]]
    gate = [pool[i] for i in gate_indices]
    dev = sweep.device
    sweep.apply(None)
    base_gate = evaluate_lm(sweep.model, gate, dev)["loss"]
    base_val = ctx["uncompressed"]["val"]
    K, B, kappa = (int(args.config.split(":")[0]), int(args.config.split(":")[1]), float(args.config.split(":")[2]))

    def taylor_moments(d, probe, bs):
        """Batch means of -g*d + 0.5*d*(H_ll probe) and of -g*d, one layer graph at a time."""
        model = sweep.model
        named = dict(model.named_parameters())
        blocks = _structural_blocks_for_scoring(model, sweep.layers, sweep.family, sweep.blocks)
        full = {n: torch.zeros(d[n].shape, dtype=torch.float32, device=dev) for n in sweep.names}
        m2 = {n: torch.zeros(d[n].shape, dtype=torch.float32, device=dev) for n in sweep.names}
        first = {n: torch.zeros(d[n].shape, dtype=torch.float32, device=dev) for n in sweep.names}
        state = _snapshot_scoring_state(model)
        model.eval()
        try:
            for eligible, block in zip(sweep.layers, blocks):
                for name, par in named.items():
                    par.requires_grad_(name in set(block))
                params = [named[n] for n in block]
                dl = {n: d[n].to(dev, torch.float32) for n in eligible}
                pl = {n: probe[n].to(dev, torch.float32) for n in eligible}
                for b, batch in enumerate(bs):
                    with sdpa_kernel(SDPBackend.MATH):
                        loss = lm_loss(model, batch, dev)
                        grads = dict(zip(block, torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)))
                        gp = sum((grads[n] * pl[n]).sum() for n in eligible)
                        hv = dict(zip(block, torch.autograd.grad(gp, params, allow_unused=True)))
                    for n in eligible:
                        f = -grads[n].detach().float() * dl[n]
                        c = f + 0.5 * dl[n] * hv[n].detach().float()
                        diff = c - full[n]
                        full[n].add_(diff, alpha=1.0 / (b + 1))
                        m2[n].add_(diff * (c - full[n]))
                        first[n].add_(f - first[n], alpha=1.0 / (b + 1))
                    del loss, grads, gp, hv
        finally:
            _restore_scoring_state(model, *state)
        return full, m2, first

    def full_quadratic(probe, bs):
        """Mean over batches of probe^T H probe with the whole-model Hessian (one double backward per batch)."""
        model = sweep.model
        named = dict(model.named_parameters())
        state = _snapshot_scoring_state(model)
        model.eval()
        wanted = set(sweep.names)
        for name, par in named.items():
            par.requires_grad_(name in wanted)
        params = [named[n] for n in sweep.names]
        pl = [probe[n].to(dev, torch.float32) for n in sweep.names]
        total = 0.0
        try:
            for batch in bs:
                with sdpa_kernel(SDPBackend.MATH):
                    loss = lm_loss(model, batch, dev)
                    grads = torch.autograd.grad(loss, params, create_graph=True)
                    gp = sum((g * p).sum() for g, p in zip(grads, pl))
                    hv = torch.autograd.grad(gp, params)
                total += float(sum((h.detach().double() * p.double()).sum() for h, p in zip(hv, pl)))
                del loss, grads, gp, hv
        finally:
            _restore_scoring_state(model, *state)
        return total / len(bs)

    rows = []
    for sp in [float(x) for x in args.sparsities.split(",")]:
        set_seed(42)
        sweep.apply(None)
        stage = 0

        def scorer(current, unused, keep_prev, seed):
            nonlocal stage
            bs = scoring[stage * B:(stage + 1) * B]
            stage += 1
            if args.select == "taylor":
                d = _masked_delta(current, keep_prev)
                mean, m2, _ = taylor_moments(d, d, bs)
                return {n: mean[n].abs() + kappa * (m2[n] / (B - 1)).clamp_min(0).sqrt() / math.sqrt(B) for n in current.names}, {}
            mean, m2, _ = grad_moments(current, bs, True)
            return {n: mean[i].abs() + kappa * (m2[i] / (B - 1)).clamp_min(0).sqrt() / math.sqrt(B)
                    for i, n in enumerate(current.names)}, {}

        _, _, keep = run_iterative(sweep, scoring, geometric_schedule(sp, K), scorer, 42, eval_at=[], return_keep=True)
        sweep.apply(keep)
        gate_loss = evaluate_lm(sweep.model, gate, dev)["loss"]
        val_loss = sweep.evaluate()["val"]
        sweep.apply(None)
        # Prediction at the checkpoint for deleting z = ~keep: shares with probe d_z sum to g^T e + 0.5 e^T H_bd e.
        dz = {n: sweep.delta[n] * (~keep[n]).cpu() for n in sweep.names}
        full, _, first = taylor_moments(dz, dz, scoring[:args.predict_batches])
        pred_h = sum(float(full[n].double().sum()) for n in sweep.names)
        pred_g = sum(float(first[n].double().sum()) for n in sweep.names)
        pred_full = pred_g + 0.5 * full_quadratic(dz, scoring[:args.predict_batches])
        rec = {"sparsity": sp, "select": args.select, "config": [K, B, kappa],
               "measured_gate": gate_loss - base_gate, "measured_val": val_loss - base_val,
               "val_ppl_change_percent": 100 * math.expm1(val_loss - base_val),
               "pred_first_order": pred_g, "pred_taylor": pred_h, "pred_taylor_full_hessian": pred_full, "predict_batches": args.predict_batches}
        for k in ("gate", "val"):
            m = rec[f"measured_{k}"]
            rec[f"rel_err_first_order_{k}"] = abs(pred_g - m) / abs(m)
            rec[f"rel_err_taylor_{k}"] = abs(pred_h - m) / abs(m)
        rows.append(rec)
        write_json(args.out, rows)
        print("PRED " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        del keep, dz, full, first
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
