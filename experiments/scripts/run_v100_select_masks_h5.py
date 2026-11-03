"""Select residual keep masks at one deletion rate for DACP-G, ExCP and Inshrinkerator.

DACP-G uses staged first-order scoring with the variance penalty; its stage,
batch and penalty configuration is chosen on the training-calibration (gate)
loss only. With --score taylor and --probe stage, the curvature term of DACP-H at
stage k uses the residual of the entries that stage k deletes, provisionally
chosen by the first-order score with the same penalty (and optionally refined
by the Taylor score), instead of the whole surviving residual. ExCP and Inshrinkerator use their fixed rules (residual magnitude;
block weight-gradient product on eight scoring batches). Held-out validation is
evaluated once per selected mask and is never used for selection.
"""
from pathlib import Path
import argparse
import json
import math
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GRID = [(4, 4, k, False) for k in (0.0, 0.5, 1.0, 2.0)]
GRID += [(4, b, k, False) for b in (8, 16) for k in (0.0, 0.5, 1.0, 2.0)]
GRID += [(8, 4, 1.0, False), (8, 8, 1.0, False), (8, 8, 2.0, False), (16, 4, 1.0, False), (16, 4, 2.0, False)]
# Shared batches: every stage rescores the same B batches (the 16x16 DACP-G setting of the tolerance and 98% experiments).
SHARED = [(k, b, kappa, True) for k, b in ((8, 16), (16, 8), (16, 16)) for kappa in (1.0, 2.0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sparsity", type=float, required=True)
    ap.add_argument("--grid", choices=("disjoint", "shared", "all"), default="all")
    ap.add_argument("--configs", help="comma-separated K:B:kappa[:shared] entries; overrides --grid")
    ap.add_argument("--score", choices=("first_order", "taylor"), default="first_order")
    ap.add_argument("--probe", choices=("surviving", "stage"), default="surviving",
                    help="HVP direction of the Taylor score: whole surviving residual, or the provisional deletions of the current stage")
    ap.add_argument("--refine", type=int, default=0, help="extra rounds that reselect the stage deletions with the Taylor score")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--signed", action="store_true",
                    help="rank by the signed upper bound mean + kappa*SE (shifted to be non-negative) instead of |mean| + kappa*SE")
    ap.add_argument("--fd-eps", type=float, default=0.1, help="step of the finite-difference HVP along the probe residual")
    ap.add_argument("--fd-bf16", action="store_true", help="compute the two finite-difference gradients under BF16 autocast")
    ap.add_argument("--hvp", choices=("block", "full", "fd"), default="block",
                    help="layer-wise Hessian blocks, or the whole-model Hessian including cross-layer blocks")
    ap.add_argument("--penalty-from", choices=("taylor", "first_order"), default="taylor",
                    help="batch spread used by the variance penalty of the Taylor score")
    ap.add_argument("--dacp-only", action="store_true", help="skip the ExCP and Inshrinkerator masks")
    ap.add_argument("--fisher", default="0", help="comma-separated coefficients of 0.5*v*d^2, v = bias-corrected Adam second moment")
    ap.add_argument("--magnitude-caps", default="", help="comma-separated c; DACP-G never deletes entries with |d| above c times the ExCP deletion threshold")
    args = ap.parse_args()
    import torch
    from run_residual_sweep_small import REGIMES, load_regime, global_keep_masks
    from run_residual_ablation_small import geometric_schedule, grad_moments, run_iterative
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, set_seed, sha256_file, write_json
    from experiments.lib.residual_scoring import compute_block_first_order_scores

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
    assert len(scoring) == 64 and not set(used + extra[:48]).intersection(gate_indices)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sweep.apply(None)
    baseline_gate = evaluate_lm(sweep.model, gate, "cuda:0")
    common = {"seed": 42, "checkpoint_sha256": ctx["provenance"]["checkpoint_sha256"],
              "gate_sha256": batch_hash(gate), "heldout_sha256": batch_hash(ctx["eval_batches"]),
              "sparsity": args.sparsity, "script_sha256": sha256_file(Path(__file__))}
    write_json(args.out_dir / "run-manifest.json", {"identity": common, "dacp_grid": {"disjoint": GRID, "shared": SHARED, "all": GRID + SHARED}[args.grid],
               "gate_pool_indices": gate_indices, "scoring_pool_indices": used + extra[:48],
               "baseline_gate": baseline_gate, "baseline_heldout": ctx["uncompressed"],
               "code_commit": commit, "gpu": torch.cuda.get_device_name(0)})

    def finalize(name, keep, fields):
        mask = args.out_dir / f"{name}-mask.pt"
        torch.save({n: x.cpu() for n, x in keep.items()}, mask)
        sweep.apply(keep)
        gate_eval = evaluate_lm(sweep.model, gate, "cuda:0")
        heldout = sweep.evaluate()
        sweep.apply(None)
        rec = {**fields, "status": "completed", "identity_common": common,
               "mask_path": str(mask.resolve()), "mask_sha256": sha256_file(mask),
               "gate": gate_eval, "gate_ppl_change_percent": 100 * math.expm1(gate_eval["loss"] - baseline_gate["loss"]),
               "baseline_gate": baseline_gate, "baseline": ctx["uncompressed"], "evaluation": heldout,
               "validation_ppl_change_percent": 100 * math.expm1(heldout["val"] - ctx["uncompressed"]["val"]),
               "code_commit": commit}
        write_json(args.out_dir / f"{name}.json", rec)
        print("SELECTED " + name + " " + json.dumps({k: rec[k] for k in ("gate_ppl_change_percent", "validation_ppl_change_percent")}), flush=True)

    # DACP-G: staged first-order scores with the variance penalty, configuration chosen on the gate loss.
    grid = {"disjoint": GRID, "shared": SHARED, "all": GRID + SHARED}[args.grid]
    if args.configs:
        grid = [(int(c.split(":")[0]), int(c.split(":")[1]), float(c.split(":")[2]), c.endswith(":shared"))
                for c in args.configs.split(",")]
    from run_residual_ablation_small import _masked_delta
    lambdas = [float(x) for x in args.fisher.split(",")]
    fisher = None
    if any(lam > 0 for lam in lambdas):
        from experiments.lib.residual_runtime import load_training_checkpoint
        ck = load_training_checkpoint(args.checkpoint)
        st = ck.optimizer_state
        ids = [i for g in st["param_groups"] for i in g["params"]]
        order = [n for n, x in ck.model_state.items() if x.dtype.is_floating_point]
        assert len(order) == len(ids) and len(st["param_groups"]) == 1
        beta2 = st["param_groups"][0]["betas"][1]
        wanted = set(sweep.names)
        fisher = {}
        for n, i in zip(order, ids):
            if n in wanted:
                s = st["state"][i]
                assert tuple(s["exp_avg_sq"].shape) == tuple(sweep.delta[n].shape)
                fisher[n] = (s["exp_avg_sq"].float() / (1 - beta2 ** float(s["step"]))).to(sweep.device)
        assert set(fisher) == wanted
        del ck, st
    grid = [(k, b, kappa, shared, lam) for k, b, kappa, shared in grid for lam in lambdas]
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.lib.residual_scoring import (_restore_scoring_state, _snapshot_scoring_state,
                                                  _structural_blocks_for_scoring)
    from experiments.lib.residual_runtime import lm_loss
    from run_residual_sweep_small import TINY

    def taylor_moments(current, bs, keep_prev, probe_keep=None):
        """Per-batch mean and M2 of -g_i d_i + 0.5 d_i (H_ll p_l)_i, one layer graph at a time;
        p is the surviving residual, or its restriction to probe_keep."""
        d = _masked_delta(current, keep_prev)
        p = d if probe_keep is None else {n: d[n] * probe_keep[n].cpu() for n in current.names}
        model, dev = current.model, current.device
        named = dict(model.named_parameters())
        blocks = _structural_blocks_for_scoring(model, current.layers, current.family, current.blocks)
        mean = {n: torch.zeros(d[n].shape, dtype=torch.float32, device=dev) for n in current.names}
        m2 = {n: torch.zeros(d[n].shape, dtype=torch.float32, device=dev) for n in current.names}
        fo = args.penalty_from == "first_order"
        fmean = {n: torch.zeros(d[n].shape, dtype=torch.float32, device=dev) for n in current.names} if fo else None
        fm2 = {n: torch.zeros(d[n].shape, dtype=torch.float32, device=dev) for n in current.names} if fo else None
        state = _snapshot_scoring_state(model)
        model.eval()
        groups = list(zip(current.layers, blocks))
        if args.hvp in ("full", "fd"):
            groups = [([n for layer in current.layers for n in layer], [n for blk in blocks for n in blk])]
        try:
            for eligible, block in groups:
                bset = set(block)
                for name, par in named.items():
                    par.requires_grad_(name in bset)
                params = [named[n] for n in block]
                dl = {n: d[n].to(dev, torch.float32) for n in eligible}
                pl = {n: p[n].to(dev, torch.float32) for n in eligible}
                for b, batch in enumerate(bs):
                    if args.hvp == "fd":
                        # Hv ~ [g(theta + eps p) - g(theta)] / eps: two ordinary backward passes, no second-order graph.
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.fd_bf16):
                            loss = lm_loss(model, batch, dev)
                        grads = dict(zip(block, torch.autograd.grad(loss, params, allow_unused=True)))
                        with torch.no_grad():
                            saved = {n: named[n].detach().clone() for n in eligible}
                            for n in eligible:
                                named[n].add_(pl[n].to(named[n].dtype), alpha=args.fd_eps)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.fd_bf16):
                            loss2 = lm_loss(model, batch, dev)
                        g2 = dict(zip(block, torch.autograd.grad(loss2, params, allow_unused=True)))
                        with torch.no_grad():
                            for n in eligible:
                                named[n].copy_(saved[n])
                        hv = {n: (g2[n].float() - grads[n].float()) / args.fd_eps for n in eligible}
                        gp = None
                        del saved, g2, loss2
                    else:
                      with sdpa_kernel(SDPBackend.MATH):
                        loss = lm_loss(model, batch, dev)
                        grads = dict(zip(block, torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)))
                        gp = sum((grads[n] * pl[n]).sum() for n in eligible)
                        hv = dict(zip(block, torch.autograd.grad(gp, params, allow_unused=True)))
                    for n in eligible:
                        f = -grads[n].detach().float() * dl[n]
                        c = f + 0.5 * dl[n] * hv[n].detach().float()
                        if fo:
                            fd = f - fmean[n]
                            fmean[n].add_(fd, alpha=1.0 / (b + 1))
                            fm2[n].add_(fd * (f - fmean[n]))
                        diff = c - mean[n]
                        mean[n].add_(diff, alpha=1.0 / (b + 1))
                        m2[n].add_(diff * (c - mean[n]))
                    del loss, grads, gp, hv
                del dl, pl
        finally:
            _restore_scoring_state(model, *state)
        if fo:
            del fmean
            return mean, fm2
        return mean, m2

    def shift(scores, keep_prev):
        """Signed scores moved to be non-negative over the surviving entries (order preserved)."""
        low = min(float((s if keep_prev is None else s[keep_prev[n].to(s.device)]).min()) for n, s in scores.items())
        return {n: (s - low).clamp_min(0) for n, s in scores.items()}

    def stage_deletions(current, scores, keep_prev, sp):
        """Entries the stage would delete under these scores (run_iterative's selection rule)."""
        dev = current.device
        masked = {}
        for n in current.names:
            sc = scores[n].to(dev, torch.float32) + TINY
            masked[n] = sc if keep_prev is None else torch.where(keep_prev[n].to(dev), sc, torch.zeros_like(sc))
        (_, keep, _), = list(global_keep_masks(masked, current.names, [sp], dev))
        del masked
        return {n: (~keep[n]) if keep_prev is None else (keep_prev[n].to(keep[n].device) & ~keep[n]) for n in current.names}
    caps = [float(x) for x in args.magnitude_caps.split(",") if x] or [0.0]
    t_excp, mag = None, None
    if any(caps):
        sweep.apply(None)
        mag = {n: sweep.delta[n].abs() for n in sweep.names}
        (_, keep_e, _), = list(global_keep_masks(mag, sweep.names, [args.sparsity], sweep.device))
        t_excp = max(float(mag[n][~keep_e[n].cpu()].max()) for n in sweep.names if bool((~keep_e[n]).any()))
        del keep_e
        print(f"EXCP_THRESHOLD {t_excp:.6e}", flush=True)
    big = torch.finfo(torch.float32).max
    for cap in caps:
        protect = {n: mag[n] > cap * t_excp for n in sweep.names} if cap else None
        candidates, best, best_keep = [], None, None
        for stages, batches, penalty, shared, lam in grid:
            tag = f"K{stages}_B{batches}_k{penalty:g}" + ("_shared" if shared else "") + (f"_f{lam:g}" if lam else "") + args.tag_suffix
            sched = geometric_schedule(args.sparsity, stages)
            set_seed(42)
            sweep.apply(None)
            stage = 0

            def scorer(current, unused, keep_prev, seed):
                nonlocal stage
                bs = scoring[:batches] if shared else scoring[stage * batches:(stage + 1) * batches]
                stage += 1
                if args.score == "taylor":
                    def ucb(mean, m2):
                        out = {n: (mean[n] if args.signed else mean[n].abs()) + (penalty * (m2[n] / (batches - 1)).clamp_min(0).sqrt() / math.sqrt(batches) if penalty else 0)
                               for n in current.names}
                        return shift(out, keep_prev) if args.signed else out
                    probe_keep = None
                    if args.probe == "stage":
                        gm, gm2, _ = grad_moments(current, bs, True)
                        probe_keep = stage_deletions(current, ucb(dict(zip(current.names, gm)), dict(zip(current.names, gm2))),
                                                     keep_prev, sched[stage - 1])
                        del gm, gm2
                    scores = ucb(*taylor_moments(current, bs, keep_prev, probe_keep))
                    for _ in range(args.refine):
                        probe_keep = stage_deletions(current, scores, keep_prev, sched[stage - 1])
                        scores = ucb(*taylor_moments(current, bs, keep_prev, probe_keep))
                    return scores, {"score": "taylor"}
                mean, m2, losses = grad_moments(current, bs, penalty > 0)
                scores = {n: (mean[i] if args.signed else mean[i].abs()) + (penalty * (m2[i] / (batches - 1)).clamp_min(0).sqrt() / math.sqrt(batches) if penalty else 0)
                          for i, n in enumerate(current.names)}
                if args.signed:
                    scores = shift(scores, keep_prev)
                if lam:
                    d = current.delta_for_scoring()
                    for i, n in enumerate(current.names):
                        scores[n] = scores[n] + lam * 0.5 * fisher[n] * current.dev(d[i]).float() ** 2
                    del d
                if protect is not None:
                    for n in current.names:
                        scores[n] = torch.where(protect[n].to(scores[n].device), torch.full_like(scores[n], big), scores[n])
                return scores, {"score_losses": losses}

            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            _, detail, keep = run_iterative(sweep, scoring, sched, scorer, 42, eval_at=[], return_keep=True)
            seconds = time.perf_counter() - began
            sweep.apply(keep)
            gate_eval = evaluate_lm(sweep.model, gate, "cuda:0")
            heldout = sweep.evaluate()
            delta = gate_eval["loss"] - baseline_gate["loss"]
            rec = {"tag": tag, "config": [stages, batches, penalty] + (["shared"] if shared else []) + (["fisher", lam] if lam else []), "backward_budget": stages * batches,
                   "scoring_sha256": batch_hash(scoring[:batches] if shared else scoring[:stages * batches]), "gate": gate_eval,
                   "gate_delta_loss": delta, "gate_ppl_change_percent": 100 * math.expm1(delta), "score_seconds": seconds,
                   "score": args.score, "probe": args.probe if args.score == "taylor" else None, "refine": args.refine, "penalty_from": args.penalty_from, "signed": args.signed, "hvp": args.hvp, "fd_eps": args.fd_eps if args.hvp == "fd" else None, "fd_bf16": args.fd_bf16 if args.hvp == "fd" else None, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "heldout": heldout, "validation_delta_loss": heldout["val"] - ctx["uncompressed"]["val"],
                   "validation_ppl_change_percent": 100 * math.expm1(heldout["val"] - ctx["uncompressed"]["val"])}
            candidates.append(rec)
            write_json(args.out_dir / (f"dacp_candidates_{args.score}" + (f"_{args.probe}" if args.score == "taylor" else "") + args.tag_suffix + (f"_cap{cap:g}" if cap else "") + ".json"), candidates)
            print(f"RESULT dacp {args.score} {tag} gate_ppl={rec['gate_ppl_change_percent']:.4f} val_ppl={rec['validation_ppl_change_percent']:.4f} seconds={seconds:.1f}", flush=True)
            if best is None or delta < best["gate_delta_loss"]:
                best, best_keep = rec, {n: x.cpu() for n, x in keep.items()}
            del keep
            sweep.apply(None)
            sweep.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        finalize(("dacp" if args.score == "first_order" else f"dacp_taylor_{args.probe}") + args.tag_suffix + ("_fisher" if any(lambdas) else "") + (f"_cap{cap:g}" if cap else ""), best_keep, {"method": "iterative_ucb_disjoint", "score": args.score, "config": best["config"],
                 "magnitude_cap": cap or None, "excp_threshold": t_excp,
                 "backward_budget": best["backward_budget"],
                 "selection_rule": "minimum training-calibration loss over the registered grid; heldout validation not used"})
        del best_keep
    if args.dacp_only:
        print("DONE", flush=True)
        return

    # ExCP: residual magnitude, exact global threshold.
    sweep.apply(None)
    scores = {n: sweep.delta[n].abs() for n in sweep.names}
    (_, keep, _), = list(global_keep_masks(scores, sweep.names, [args.sparsity], sweep.device))
    del scores
    finalize("excp", keep, {"method": "excp_style", "config": ["residual-magnitude"], "backward_budget": 0})
    del keep

    # Inshrinkerator: block weight-gradient product on the first eight scoring batches (as in the 70% runs).
    sweep.apply(None)
    scores, _ = compute_block_first_order_scores(sweep.model, initial[:8], sweep.layers, sweep.cur_cpu, sweep.device,
                                                 model_family=sweep.family, block_parameter_names=sweep.blocks)
    (_, keep, _), = list(global_keep_masks(scores, sweep.names, [args.sparsity], sweep.device))
    del scores
    finalize("ins", keep, {"method": "inshrinkerator_style", "config": ["block-weight-gradient"], "backward_budget": 8})
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
