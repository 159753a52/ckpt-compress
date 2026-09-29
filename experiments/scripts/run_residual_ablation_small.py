"""Ablations of iterative re-linearized residual pruning on Pythia-410M.

Builds on ``run_residual_sweep_small.py`` (same regimes, data batches, eligible
matrices, exact global k-smallest masks, immediate restoration, 100 held-out
evaluation batches).  One JSON per (task, regime, seed) with the sweep schema
(``uncompressed``, ``evals[tag][str(p)]``, ``sparsities``, ``residual``,
``full_restore``, ``regime``, ``seed``), written as
``abl-<task>_<regime>_seed<seed>.json`` so the original ``R*_seed*.json`` glob of
the sweep summarizer does not pick these files up.

Iterative re-linearization (all ``iterative_*`` tags): for a stage schedule
0 < P_1 < ... < P_K, stage j sets the model to the partially restored point
theta_{j-1} (coordinates pruned by stages < j restored to the reference), scores
the still-surviving coordinates there, and prunes the floor(P_j N) lowest-scored
coordinates overall; previously pruned coordinates are forced to stay pruned
(score 0, survivors get +1e-38), so the pruned set is monotone.  For a surviving
coordinate i restoring it moves it by -d_i with d = theta_ckpt - theta_ref.

Tasks (``--tasks``):
  sched   schedule ablation.  For target P and K stages the schedule is geometric
          in the kept fraction: P_j = 1 - (1 - P)^(j/K), j = 1..K (kept fraction
          shrinks by the constant factor (1-P)^(1/K) per stage; K = 1 is one-shot
          first order with the averaged full gradient).  Score |g(theta_{j-1}) d|.
          Tags iterative_K{K}; each target is a separate run, evaluated only at P.
          ``gradient_passes`` = K (each pass = one backward per scoring batch).
  combo   combined scorers on the sweep's list schedule 0.5,0.8,0.9,0.95,0.98,0.99,
          0.995 (identical to iterative_first_order of the sweep; evaluated at every
          stage):
            iterative_first_order  |g d| (re-run, reproducibility check)
            iterative_ucb          |mean_b c_b| + std_b(c_b)/sqrt(B), c_b = -g_b d
            iterative_taylor       compute_block_taylor_scores(abs_mean) with
                                   delta = d * survivors at theta_{j-1}
            iterative_probe        compute_probe_gn_scores(topk, m=1), same delta
          plus the same scorers on the geometric K=8 schedule per target
          (iterative_ucb_K8, iterative_probe_K8) when ``--combo-k`` is set.
  batches number of scoring batches B in {2,4,8,16}: the scoring set is
          partition_repeated_seed_batches(pool, 100, 1, 16, seed).scoring_batches[0]
          truncated to its first B batches (the first 8 are exactly the sweep's
          scoring batches; the 16 are disjoint from the evaluation batches, which
          come from the validation loader).  Tags first_order_B{B},
          first_order_ucb_B{B} (one pass of per-batch full gradients) and
          iterative_first_order_B{B} (list schedule, evaluated at 0.9,0.98,0.995).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_residual_sweep_small import (  # noqa: E402
    REGIMES,
    SPARSITIES,
    TINY,
    Sweep,
    _jsonable,
    global_keep_masks,
    load_regime,
    regime_dataset,
    regime_model,
    now,
)

TensorDict = Dict[str, torch.Tensor]
SCHED_TARGETS = [0.9, 0.95, 0.98, 0.99, 0.995]
SCHED_KS = [1, 2, 4, 8, 16]
BATCH_COUNTS = [2, 4, 8, 16]
BATCH_TARGETS = [0.9, 0.98, 0.995]


def geometric_schedule(target: float, k: int) -> List[float]:
    """P_j = 1 - (1 - P)^(j/K); the last stage is exactly the target."""
    sched = [1.0 - (1.0 - target) ** (j / k) for j in range(1, k + 1)]
    sched[-1] = target
    return sched


def list_schedule(target: float) -> List[float]:
    return [p for p in SPARSITIES if p < target - 1e-12] + [target]


# ------------------------------------------------------------------ scorers at theta_{j-1}
# Each scorer is called with the model already at the partial point and returns
# non-negative scores for every eligible coordinate (device or CPU tensors) and meta.

def grad_moments(s: Sweep, batches, with_std: bool):
    """Per-coordinate mean (and M2) of c_b = -g_b * d over the batches at the current point."""
    d = s.delta_for_scoring()
    mean = [torch.zeros_like(x, device=s.device) for x in d]
    m2 = [torch.zeros_like(x, device=s.device) for x in d] if with_std else None

    def visit(b: int, grads: List[torch.Tensor]) -> None:
        for i, g in enumerate(grads):
            c = -g * s.dev(d[i])
            diff = c - mean[i]
            mean[i].add_(diff, alpha=1.0 / (b + 1))
            if m2 is not None:
                m2[i].add_(diff * (c - mean[i]))

    losses = s.batch_gradients(batches, visit)
    del d
    return mean, m2, losses


def score_first_order(s: Sweep, batches, keep_prev, seed):
    mean, _, losses = grad_moments(s, batches, False)
    return {n: mean[i].abs() for i, n in enumerate(s.names)}, {"score_losses": losses}


def score_ucb(s: Sweep, batches, keep_prev, seed):
    B = len(batches)
    mean, m2, losses = grad_moments(s, batches, True)
    out = {}
    for i, n in enumerate(s.names):
        out[n] = mean[i].abs() + (m2[i] / (B - 1)).clamp_min(0).sqrt() / math.sqrt(B)
    return out, {"score_losses": losses}


def _masked_delta(s: Sweep, keep_prev) -> TensorDict:
    if keep_prev is None:
        return s.delta
    return {n: s.delta[n] * keep_prev[n].cpu() for n in s.names}


def score_taylor(s: Sweep, batches, keep_prev, seed):
    from experiments.lib.residual_scoring import compute_block_taylor_scores

    sc, meta = compute_block_taylor_scores(
        s.model, batches, s.layers, _masked_delta(s, keep_prev), s.device, aggregation="abs_mean",
        task_type="lm", model_family=s.family, block_parameter_names=s.blocks)
    return sc, {"score_kind": meta.get("score_kind"), "total_seconds": meta.get("total_seconds")}


def score_probe(s: Sweep, batches, keep_prev, seed):
    from experiments.lib.residual_probe_scoring import compute_probe_gn_scores

    sc, meta = compute_probe_gn_scores(
        s.model, batches, s.layers, _masked_delta(s, keep_prev), s.device, estimator="topk",
        rounds=1, seed=seed, task_type="lm", model_family=s.family, block_parameter_names=s.blocks)
    return sc, {"total_seconds": meta.get("total_seconds")}


SCORERS: Dict[str, Callable] = {"first_order": score_first_order, "ucb": score_ucb,
                                "taylor": score_taylor, "probe": score_probe}


def run_iterative(s: Sweep, batches, schedule: Sequence[float], scorer: Callable, seed: int,
                  eval_at: Optional[Sequence[float]] = None, return_keep: bool = False):
    """Monotone staged pruning; evaluates after the stages whose sparsity is in eval_at
    (all stages if None).  Returns evals keyed by str(sparsity) and per-stage records
    (plus the final keep masks if return_keep)."""
    evals, stages = {}, []
    keep_prev: Optional[TensorDict] = None
    eval_keys = None if eval_at is None else {round(p, 12) for p in eval_at}
    t_all = now()
    for j, sp in enumerate(schedule):
        t0 = now()
        s.apply(keep_prev)
        raw, meta = scorer(s, batches, keep_prev, seed)
        s.apply(None)
        score_s = now() - t0
        scores = {}
        for n in s.names:
            sc = raw[n].to(s.device, torch.float32) + TINY
            if keep_prev is not None:
                sc = torch.where(keep_prev[n], sc, torch.zeros_like(sc))
            scores[n] = sc
        del raw
        (_, keep, sel_s), = list(global_keep_masks(scores, s.names, [sp], s.device))
        del scores
        if keep_prev is not None:
            for n in s.names:
                if bool((~keep_prev[n] & keep[n]).any()):
                    raise RuntimeError("iterative pruned set is not monotone")
        rec = {"stage": j + 1, "sparsity": sp, "score_s": round(score_s, 2), "select_s": round(sel_s, 3)}
        losses = meta.get("score_losses")
        if losses:
            rec["score_loss_at_partial_point"] = sum(losses) / len(losses)
        if eval_keys is None or round(sp, 12) in eval_keys:
            s.apply(keep)
            ev = s.evaluate()
            s.apply(None)
            ev.update({"select_s": round(sel_s, 3), "gradient_passes": j + 1, "stages": len(schedule),
                       "cumulative_s": round(now() - t_all, 2)})
            evals[str(sp)] = ev
            rec["val"] = ev["val"]
        stages.append(rec)
        keep_prev = keep
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if return_keep:
        return evals, stages, keep_prev
    del keep_prev
    return evals, stages


# ------------------------------------------------------------------ tasks

def task_sched(s: Sweep, batches, seed, dv, log) -> dict:
    evals, meta = {}, {}
    for k in SCHED_KS:
        tag = f"iterative_K{k}"
        evals[tag], meta[tag] = {}, {}
        for target in SCHED_TARGETS:
            sched = geometric_schedule(target, k)
            ev, stages = run_iterative(s, batches, sched, score_first_order, seed, eval_at=[target])
            evals[tag][str(target)] = ev[str(target)]
            meta[tag][str(target)] = {"schedule": sched, "stages": stages}
            log(f"{tag} P={target}: dval {dv(ev[str(target)]):.1f}e-3 ({k} gradient passes)")
    return {"evals": evals, "iterative_stages": meta, "sparsities": SCHED_TARGETS}


def task_combo(s: Sweep, batches, seed, dv, log, scorers: Sequence[str], combo_k: int) -> dict:
    evals, meta, timing = {}, {}, {}
    names = {"first_order": "iterative_first_order", "ucb": "iterative_ucb",
             "taylor": "iterative_taylor", "probe": "iterative_probe"}
    for key in scorers:
        tag = names[key]
        t0 = now()
        evals[tag], stages = run_iterative(s, batches, SPARSITIES, SCORERS[key], seed)
        meta[tag] = {"schedule": SPARSITIES, "stages": stages}
        timing[tag] = {"total_s": round(now() - t0, 2)}
        log(f"{tag}: " + ", ".join(f"{p}:{dv(v):.1f}" for p, v in evals[tag].items()) + f" ({timing[tag]['total_s']} s)")
    if combo_k:
        for key in [k for k in scorers if k in ("ucb", "probe")]:
            tag = f"{names[key]}_K{combo_k}"
            evals[tag], meta[tag] = {}, {}
            t0 = now()
            for target in SCHED_TARGETS:
                sched = geometric_schedule(target, combo_k)
                ev, stages = run_iterative(s, batches, sched, SCORERS[key], seed, eval_at=[target])
                evals[tag][str(target)] = ev[str(target)]
                meta[tag][str(target)] = {"schedule": sched, "stages": stages}
            timing[tag] = {"total_s": round(now() - t0, 2)}
            log(f"{tag}: " + ", ".join(f"{p}:{dv(v):.1f}" for p, v in evals[tag].items()))
    return {"evals": evals, "iterative_stages": meta, "timing": timing, "sparsities": SPARSITIES}


def task_batches(s: Sweep, batches16, seed, dv, log) -> dict:
    evals, meta = {}, {}
    for B in BATCH_COUNTS:
        bs = batches16[:B]
        t0 = now()
        s.apply(None)
        mean, m2, _ = grad_moments(s, bs, True)  # one pass of per-batch gradients serves both scores
        fo = {n: mean[i].abs() for i, n in enumerate(s.names)}
        raw = {n: mean[i].abs() + (m2[i] / (B - 1)).clamp_min(0).sqrt() / math.sqrt(B)
               for i, n in enumerate(s.names)}
        del mean, m2
        score_s = now() - t0
        for tag, sc in ((f"first_order_B{B}", fo), (f"first_order_ucb_B{B}", raw)):
            evals[tag] = s.sweep_global(sc, BATCH_TARGETS, extra={"gradient_passes": 1, "scoring_batches": B})
            log(f"{tag}: " + ", ".join(f"{p}:{dv(v):.1f}" for p, v in evals[tag].items()))
        meta[f"first_order_B{B}"] = {"score_s_both": round(score_s, 2)}
        del raw, fo
        tag = f"iterative_first_order_B{B}"
        evals[tag], stages = run_iterative(s, bs, SPARSITIES, score_first_order, seed, eval_at=BATCH_TARGETS)
        for ev in evals[tag].values():
            ev["scoring_batches"] = B
        meta[tag] = {"schedule": SPARSITIES, "stages": stages}
        log(f"{tag}: " + ", ".join(f"{p}:{dv(v):.1f}" for p, v in evals[tag].items()))
        torch.cuda.empty_cache()
    return {"evals": evals, "iterative_stages": meta, "sparsities": BATCH_TARGETS}


BEST_CONFIGS = (("first_order", 16), ("ucb", 8), ("ucb", 16))


def task_best(s: Sweep, batches16, seed, dv, log, targets: Sequence[float] = SCHED_TARGETS,
              configs: Sequence = BEST_CONFIGS) -> dict:
    """Combine the best settings of the other tasks: geometric K = 16 schedule to each target,
    (scorer, B) in ``configs`` (default first_order B16, ucb B8, ucb B16)."""
    evals, meta = {}, {}
    for key, B in configs:
        tag = f"iterative_{'first_order' if key == 'first_order' else 'ucb'}_K16_B{B}"
        evals[tag], meta[tag] = {}, {}
        for target in targets:
            sched = geometric_schedule(target, 16)
            ev, stages = run_iterative(s, batches16[:B], sched, SCORERS[key], seed, eval_at=[target])
            ev[str(target)]["scoring_batches"] = B
            evals[tag][str(target)] = ev[str(target)]
            meta[tag][str(target)] = {"schedule": sched, "stages": stages}
        log(f"{tag}: " + ", ".join(f"{p}:{dv(v):.1f}" for p, v in evals[tag].items()))
    return {"evals": evals, "iterative_stages": meta, "sparsities": list(targets)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=sorted(REGIMES), required=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--tasks", default="combo,sched,batches")
    ap.add_argument("--combo-scorers", default="first_order,ucb,probe,taylor")
    ap.add_argument("--combo-k", type=int, default=0)
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--targets", default="", help="task best: target sparsities (default 0.9,...,0.995)")
    ap.add_argument("--best-configs", default="",
                    help="task best: scorer:B pairs, e.g. ucb:16,first_order:16 (default first_order:16,ucb:8,ucb:16)")
    ap.add_argument("--prefix", default="abl-", help="output file name prefix (<prefix><task>_<regime>_seed<seed>.json)")
    args = ap.parse_args()

    from experiments.lib.residual_runtime import batch_hash, set_seed, write_json
    from experiments.lib.residual_training import partition_repeated_seed_batches

    tasks = [t for t in args.tasks.split(",") if t]
    if set(tasks) - {"sched", "combo", "batches", "best"}:
        raise ValueError(f"unknown tasks {tasks}")
    scorers = [c for c in args.combo_scorers.split(",") if c]
    if set(scorers) - set(SCORERS):
        raise ValueError(f"unknown combo scorers {scorers}")
    seeds = [int(x) for x in args.seeds.split(",")]
    targets = [float(x) for x in args.targets.split(",") if x] or SCHED_TARGETS
    best_configs = BEST_CONFIGS
    if args.best_configs:
        best_configs = [(c.split(":")[0], int(c.split(":")[1])) for c in args.best_configs.split(",") if c]
        if {k for k, _ in best_configs} - {"first_order", "ucb"} or any(not 2 <= b <= 16 for _, b in best_configs):
            raise ValueError(f"bad --best-configs {args.best_configs}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_regime(args.regime, args.checkpoint_root, args.data_root, args.device)
    s, pool, eval_batches = ctx["sweep"], ctx["pool"], ctx["eval_batches"]
    unc = ctx["uncompressed"]
    spec = REGIMES[args.regime]

    def dv(ev: dict) -> float:
        return (ev["val"] - unc["val"]) * 1e3

    for seed in seeds:
        for task in tasks:
            path = args.out_dir / f"{args.prefix}{task}_{args.regime}_seed{seed}.json"
            if path.exists():
                print(f"skip existing {path}", flush=True)
                continue

            def log(msg: str) -> None:
                print(f"[{args.regime} seed {seed} {task}] {msg}", flush=True)

            set_seed(seed)
            plan8 = partition_repeated_seed_batches(pool, 100, 1, 8, seed)
            batches = plan8.scoring_batches[0]
            plan16 = partition_repeated_seed_batches(pool, 100, 1, 16, seed)
            batches16 = plan16.scoring_batches[0]
            if batch_hash(batches16[:8]) != batch_hash(batches):
                raise RuntimeError("first 8 of the 16 scoring batches differ from the sweep's scoring batches")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            s.apply(None)
            if task == "sched":
                out = task_sched(s, batches, seed, dv, log)
            elif task == "combo":
                out = task_combo(s, batches, seed, dv, log, scorers, args.combo_k)
            elif task == "best":
                out = task_best(s, batches16, seed, dv, log, targets, best_configs)
            else:
                out = task_batches(s, batches16, seed, dv, log)
            s.apply(None)
            used = batches16 if task in ("batches", "best") else batches
            res = {
                "task": task, "seed": seed, "regime": args.regime, "regime_description": spec["description"],
                "uncompressed": unc, "full_restore": ctx["full_restore"], "residual": ctx["residual"],
                **out,
                "methods": list(out["evals"]),
                "scoring_batches": {"count": len(used), "sha256": batch_hash(used),
                                    "first8_sha256": batch_hash(batches),
                                    "selected_pool_indices": list(plan16.selected_pool_indices if task in ("batches", "best")
                                                                  else plan8.selected_pool_indices)},
                "model": regime_model(args.regime), "model_family": ctx["family"], "dataset": regime_dataset(args.regime),
                "batch_size": 2, "seq_length": 128,
                "eval_batches": {"count": len(eval_batches), "sha256": batch_hash(eval_batches)},
                "training_pool": {"count": len(pool), "sha256": batch_hash(pool)},
                "provenance": ctx["provenance"],
                "protocol": "immediate restoration (no recovery training); exact global k-smallest keep masks; "
                            "iterative stages monotone; see run_residual_ablation_small.py docstring",
                "seconds": round(time.time() - t0, 1),
                "setup_seconds": round(ctx["setup_seconds"], 1),
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            }
            write_json(path, _jsonable(res))
            print(f"wrote {path} ({res['seconds']} s, peak {res['peak_gib']} GiB)", flush=True)


if __name__ == "__main__":
    main()
