"""Repeated compression-recovery chains on Pythia-410M with the reference fixed
to the base (pretrained) weights.

Small-model counterpart of ``run_base_chains_32b.py``.  Starting from the
step-1000 Alpaca checkpoint of a regime (parameters and AdamW moments; ``R1``
by default), cycle c = 1..C

  1. takes the current parameters theta, the residual d = theta - base on the
     eligible matrices, and scores it with one method,
  2. restores: the floor(P N) lowest-scored coordinates are set to their base
     values (residual sparsity P; everything else, including every non-eligible
     tensor, is kept losslessly),
  3. keeps the AdamW moments unchanged (stored losslessly),
  4. evaluates immediately on the sweep's 100 held-out batches,
  5. trains R steps on recovery chunk c and evaluates again.

The reference is never advanced: every cycle compresses the whole delta to the
base weights (ExCP's residual-to-pretrained setting).

Methods (scoring code of run_residual_sweep_small.py / run_residual_ablation_small.py)
  excp_style            |d|, exact global k-smallest
  inshrinkerator_style  |g * theta| (compute_block_first_order_scores), global
  dacp                  block HVP score (abs_mean) with per-layer Weibull allocation
  first_order_ucb       one-shot |mean_b c_b| + std_b(c_b)/sqrt(B), c_b = -g_b d
  iterative_ucb         the same score re-linearized over K stages of the geometric
                        schedule P_j = 1 - (1 - P)^(j/K) (run_iterative, monotone)
  no_compression        training only (reference trajectory)

Scoring batches depend on the seed exactly as in the sweep/ablation scripts: the
one-shot gradient baselines use the sweep's 8 scoring batches
``partition_repeated_seed_batches(pool, 100, 1, 8, seed)``, the UCB methods the
first B of the 16 batches of ``partition_repeated_seed_batches(pool, 100, 1, 16, seed)``
(the first 8 of which are the 8 above).  They are fixed across the cycles of a chain.

Recovery training mirrors the fine-tuning recipe (finetune_pythia_410m.py:
AdamW, weight decay 0.01, batch 4 x seq 512 from the Alpaca train split, token-mean
loss via its ``train_step_accum``) at a constant learning rate.  The saved cosine
schedule has decayed to 0 at step 1000; restarting at the fine-tune's peak 2e-5
raises the held-out loss of the uncompressed chain by about 0.15 nats within 500
steps, so the default is 2e-6 (0.1x peak), at which the uncompressed chain keeps
improving slowly and compressed chains recover within a cycle.  AMP is off
(no saved GradScaler state; fp32 avoids skipped steps that would differ across
methods).  The recovery chunks are the train loader batches (generator seed 0)
that follow the samples of the 140-batch scoring pool, so they are disjoint from
the scoring batches and from the validation split; chunk c is identical for all
methods and all seeds.

One JSON per (method, P, seed): ``<out-dir>/<id>_seed<seed>.json``, ids as in the
32B script (``no_compression``, ``excp_style_0.95``, ``first_order_ucb_B16_0.95``,
``iterative_ucb_B16_K16_0.95``, ...).  Existing results are skipped, so a queue
can be restarted; per-cycle records are also written under ``partial/``.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[1]
for _p in (str(REPO), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_residual_ablation_small import geometric_schedule, run_iterative, score_ucb  # noqa: E402
from run_residual_sweep_small import (  # noqa: E402
    MAX_LAYER_RATIO,
    REGIMES,
    Sweep,
    _jsonable,
    global_keep_masks,
    load_regime,
    now,
    regime_model,
)

TensorDict = Dict[str, torch.Tensor]
METHODS = ("no_compression", "excp_style", "inshrinkerator_style", "dacp", "first_order_ucb", "iterative_ucb")
POOL_BATCHES, POOL_BATCH_SIZE = 140, 2  # scoring pool of load_regime (140 batches of 2 sequences)


def chain_id(method: str, sparsity: float, b: int, k: int) -> str:
    if method == "no_compression":
        return method
    if method == "iterative_ucb":
        return f"iterative_ucb_B{b}_K{k}_{sparsity}"
    if method == "first_order_ucb":
        return f"first_order_ucb_B{b}_{sparsity}"
    return f"{method}_{sparsity}"


def evaluate(s: Sweep) -> dict:
    ev = s.evaluate()
    return {"val": {"loss": ev["val"], "ppl": ev["ppl"], "tokens": ev["tokens"], "eval_s": ev["eval_s"]}}


@torch.no_grad()
def refresh_current(s: Sweep, ref_cpu: Mapping[str, torch.Tensor]) -> None:
    """Make the model's current eligible parameters the Sweep's checkpoint point."""
    if s.offload:
        raise NotImplementedError("offloaded Sweep states are not supported by the chains")
    for n in s.names:
        p = s.params[n].detach()
        s.cur_dev[n].copy_(p)
        s.cur_cpu[n] = p.float().cpu()
        s.delta[n] = s.cur_cpu[n] - ref_cpu[n]


def select_keep(s: Sweep, method: str, sparsity: float, batches8, batches16, seed: int, b: int,
                k: int) -> Tuple[TensorDict, dict]:
    """Keep masks (True = keep the residual coordinate) for one method at the current point."""
    from experiments.lib.residual_methods import build_weibull_mask
    from experiments.lib.residual_scoring import compute_block_first_order_scores, compute_block_taylor_scores

    common = dict(task_type="lm", model_family=s.family, block_parameter_names=s.blocks)
    s.apply(None)
    info: Dict[str, object] = {}
    if method in ("first_order_ucb", "iterative_ucb"):
        schedule = [sparsity] if method == "first_order_ucb" else geometric_schedule(sparsity, k)
        _, stages, keep = run_iterative(s, batches16[:b], schedule, score_ucb, seed, eval_at=[], return_keep=True)
        info.update({"schedule": schedule, "stages": stages, "scoring_batches": b})
    elif method == "dacp":
        t0 = now()
        sc, meta = compute_block_taylor_scores(s.model, batches8, s.layers, s.delta, s.device,
                                               aggregation="abs_mean", **common)
        info["score_s"] = round(now() - t0, 2)
        t1 = now()
        keep, alloc = build_weibull_mask(s.layers, sc, sparsity, max_layer_ratio=MAX_LAYER_RATIO)
        info.update({"select_s": round(now() - t1, 3), "target_pruned": alloc["target_pruned"],
                     "weibull_layer_counts": alloc["weibull_layer_counts"], "max_layer_ratio": MAX_LAYER_RATIO,
                     "scoring_batches": len(batches8)})
        del sc
    elif method in ("excp_style", "inshrinkerator_style"):
        t0 = now()
        if method == "excp_style":
            sc = {n: s.delta[n].abs() for n in s.names}
        else:
            sc, _ = compute_block_first_order_scores(s.model, batches8, s.layers, s.cur_cpu, s.device, **common)
            info["scoring_batches"] = len(batches8)
        info["score_s"] = round(now() - t0, 2)
        (_, keep, sel_s), = list(global_keep_masks(sc, s.names, [sparsity], s.device))
        info["select_s"] = round(sel_s, 3)
        del sc
    else:
        raise ValueError(f"no scorer for {method}")
    s.apply(None)
    s.model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return keep, info


@torch.no_grad()
def restore(s: Sweep, keep: TensorDict) -> dict:
    """Set pruned coordinates to the base values; returns kept-count statistics."""
    s.apply(keep)
    kept_layer, kept_l2, total_l2 = [], 0.0, 0.0
    for layer in s.layers:
        c = 0
        for n in layer:
            m = keep[n].to(s.params[n].device)
            if not torch.equal(s.params[n][~m], s.ref_dev[n][~m]):
                raise RuntimeError(f"pruned coordinates of {n} are not at the reference")
            c += int(m.sum())
            d = s.delta[n].to(m.device)
            kept_l2 += float((d * m).double().square().sum())
            total_l2 += float(d.double().square().sum())
        kept_layer.append(c)
    return {"kept": sum(kept_layer), "kept_per_layer": kept_layer, "kept_residual_l2": math.sqrt(kept_l2),
            "kept_residual_l2_fraction": math.sqrt(kept_l2 / total_l2) if total_l2 > 0 else None}


def run_chain(s: Sweep, method: str, sparsity: float, cycles: int, recovery_chunks: Sequence[Sequence],
              optimizer: torch.optim.Optimizer, train_fn: Callable, batches8, batches16, seed: int, b: int, k: int,
              ref_cpu: Mapping[str, torch.Tensor], start_step: int, log: Callable[[str], None],
              on_cycle: Optional[Callable[[dict], None]] = None, train_seed: int = 10_000) -> List[dict]:
    """C compression-recovery cycles from the model's current state (moments kept as they are)."""
    from experiments.lib.residual_runtime import set_seed

    if len(recovery_chunks) < cycles:
        raise ValueError("not enough recovery chunks")
    total = sum(s.delta[n].numel() for n in s.names)
    out = []
    step = start_step
    for c in range(1, cycles + 1):
        tcyc = now()
        refresh_current(s, ref_cpu)
        rec: Dict[str, object] = {"cycle": c, "step": step, "residual": s.residual_stats()}
        if method != "no_compression":
            set_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = now()
            keep, info = select_keep(s, method, sparsity, batches8, batches16, seed, b, k)
            rec["score"] = info
            rec["score_total_s"] = round(now() - t0, 2)
            t0 = now()
            rec["restore"] = restore(s, keep)
            rec["restore"]["target_kept"] = total - math.floor(sparsity * total)
            rec["restore_s"] = round(now() - t0, 2)
            del keep
            if torch.cuda.is_available():
                rec["score_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        rec["immediate"] = evaluate(s)
        chunk = recovery_chunks[c - 1]
        set_seed(train_seed + c)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = now()
        losses, lrs = [], []
        for batch in chunk:
            lrs.append(float(optimizer.param_groups[0]["lr"]))
            losses.append(float(train_fn(s.model, optimizer, batch)))
        s.model.zero_grad(set_to_none=True)
        rec["recovery_train_s"] = round(now() - t0, 2)
        if torch.cuda.is_available():
            rec["recovery_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        rec["train_log"] = {"steps": len(chunk), "train_losses": losses, "learning_rates": lrs}
        step += len(chunk)
        rec["final"] = evaluate(s)
        rec["cycle_s"] = round(now() - tcyc, 2)
        out.append(rec)
        if on_cycle is not None:
            on_cycle(rec)
        log(f"cycle {c}: immediate {rec['immediate']['val']['loss']:.5f} final {rec['final']['val']['loss']:.5f} "
            f"score {rec.get('score_total_s', 0)} s cycle {rec['cycle_s']} s")
    refresh_current(s, ref_cpu)
    return out


def recovery_data(model_name: str, data_root: Path, batch_size: int, seq_length: int, count: int,
                  pool: Sequence[Mapping[str, torch.Tensor]]) -> Tuple[list, dict]:
    """Train-loader batches following the samples of the scoring pool (same generator-seeded order)."""
    from experiments.lib.data import cache_batches, get_data_loaders

    pool_samples = POOL_BATCHES * POOL_BATCH_SIZE
    if pool_samples % batch_size:
        raise ValueError("recovery batch size must divide the scoring pool's sample count")
    skip = pool_samples // batch_size
    loader, _, task_type = get_data_loaders(model_name, "alpaca", batch_size, seq_length, data_dir=str(data_root))
    cached = cache_batches(loader, skip + count, task_type)
    if len(cached) != skip + count:
        raise RuntimeError("train loader returned too few batches")
    # the skipped batches hold the pool's samples: compare the first 128 tokens of each sequence
    head = torch.cat([bt["input_ids"] for bt in cached[:skip]])
    pool_ids = torch.cat([bt["input_ids"] for bt in pool])
    width = pool_ids.shape[1]
    aligned = bool(head.shape[0] == pool_ids.shape[0] and torch.equal(head[:, :width], pool_ids))
    return cached[skip:], {"skipped_batches": skip, "pool_alignment_verified": aligned,
                           "batch_size": batch_size, "seq_length": seq_length, "count": count}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=sorted(REGIMES), default="R1")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--sparsity", type=float, default=0.95)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--recovery", type=int, default=50, help="recovery steps per cycle")
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--train-batch-size", type=int, default=4)
    ap.add_argument("--train-seq-length", type=int, default=512)
    ap.add_argument("--micro-batch-size", type=int, default=2)
    ap.add_argument("--b", type=int, default=16, help="scoring batches of the UCB methods")
    ap.add_argument("--k", type=int, default=16, help="stages of iterative_ucb")
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from experiments.lib.residual_runtime import batch_hash, load_training_checkpoint, sha256_file, write_json
    from experiments.lib.residual_training import build_optimizer, partition_repeated_seed_batches
    from experiments.scripts.finetune.finetune_pythia_410m import train_step_accum

    methods = [m for m in args.methods.split(",") if m]
    if set(methods) - set(METHODS):
        raise ValueError(f"unknown methods {methods}")
    if not 2 <= args.b <= 16:
        raise ValueError("--b must be in [2, 16]")
    seeds = [int(x) for x in args.seeds.split(",")]
    todo = [(seed, m) for seed in seeds for m in methods
            if not (args.out_dir / f"{chain_id(m, args.sparsity, args.b, args.k)}_seed{seed}.json").exists()]
    if not todo:
        print("all results exist", flush=True)
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "partial").mkdir(exist_ok=True)

    t_setup = time.time()
    ctx = load_regime(args.regime, args.checkpoint_root, args.data_root, args.device)
    s, pool, eval_batches = ctx["sweep"], ctx["pool"], ctx["eval_batches"]
    ckpt_path = args.checkpoint_root / REGIMES[args.regime]["checkpoint"]
    ckpt = load_training_checkpoint(ckpt_path)
    ckpt_state = {n: v.detach().clone() for n, v in ckpt.model_state.items()}
    ckpt_opt, step0 = ckpt.optimizer_state, ckpt.step
    del ckpt
    ref_cpu = {n: s.ref_dev[n].float().cpu() for n in s.names}
    params = list(s.model.parameters())
    shapes = [tuple(ckpt_opt["state"][i]["exp_avg"].shape) for i in range(len(params))]
    if shapes != [tuple(p.shape) for p in params]:
        raise RuntimeError("optimizer state does not match the model's parameter order")
    saved_group = {k: v for k, v in ckpt_opt["param_groups"][0].items() if k != "params"}
    rec_batches, rec_meta = recovery_data(regime_model(args.regime), args.data_root, args.train_batch_size,
                                          args.train_seq_length, args.cycles * args.recovery, pool)
    chunks = [rec_batches[i * args.recovery:(i + 1) * args.recovery] for i in range(args.cycles)]
    rec_meta["chunk_sha256"] = [batch_hash(ch) for ch in chunks]
    setup_s = time.time() - t_setup
    print(f"setup {setup_s:.1f} s; recovery data {rec_meta}", flush=True)

    def train_fn(model, opt, batch):
        return train_step_accum(model, opt, None, batch, args.device, args.micro_batch_size, use_amp=False)

    for seed, m in todo:
        cid = chain_id(m, args.sparsity, args.b, args.k)
        path = args.out_dir / f"{cid}_seed{seed}.json"
        if path.exists():
            continue
        stamp = time.strftime("%Y%m%dT%H%M%S")

        def log(msg: str) -> None:
            print(f"[{cid} seed {seed}] {msg}", flush=True)

        plan8 = partition_repeated_seed_batches(pool, 100, 1, 8, seed)
        plan16 = partition_repeated_seed_batches(pool, 100, 1, 16, seed)
        batches8, batches16 = plan8.scoring_batches[0], plan16.scoring_batches[0]
        if batch_hash(batches16[:8]) != batch_hash(batches8):
            raise RuntimeError("first 8 of the 16 scoring batches differ from the sweep's scoring batches")
        t0 = time.time()
        s.model.load_state_dict(ckpt_state, strict=True)
        refresh_current(s, ref_cpu)
        opt = build_optimizer(s.model, ckpt_opt, learning_rate=args.lr)
        unc = evaluate(s)
        log(f"start: uncompressed val {unc['val']['loss']:.5f}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def on_cycle(rec: dict) -> None:
            write_json(args.out_dir / "partial" / f"{cid}_seed{seed}_{stamp}_c{rec['cycle']}.json", _jsonable(rec))

        cycles = run_chain(s, m, args.sparsity, args.cycles, chunks, opt, train_fn, batches8, batches16, seed,
                           args.b, args.k, ref_cpu, step0, log, on_cycle)
        used = batches16[:args.b] if m in ("first_order_ucb", "iterative_ucb") else batches8
        res = {
            "id": cid, "method": m, "sparsity": None if m == "no_compression" else args.sparsity,
            "seed": seed, "reference": "base", "regime": args.regime,
            "recovery_steps": args.recovery, "cycles_n": args.cycles,
            "b": args.b if m in ("first_order_ucb", "iterative_ucb") else None,
            "k": args.k if m == "iterative_ucb" else None,
            "schedule": (geometric_schedule(args.sparsity, args.k) if m == "iterative_ucb"
                         else [args.sparsity] if m == "first_order_ucb" else None),
            "weibull_cap": MAX_LAYER_RATIO if m == "dacp" else None,
            "start_step": step0, "uncompressed": unc, "cycles": cycles,
            "train": {"lr": args.lr, "constant_lr": True, "batch_size": args.train_batch_size,
                      "seq_length": args.train_seq_length, "micro_batch_size": args.micro_batch_size,
                      "amp": False, "optimizer": "AdamW (checkpoint moments)", "saved_param_group": saved_group,
                      "recovery_data": rec_meta},
            "scoring_batches": None if m in ("no_compression", "excp_style") else
            {"count": len(used), "sha256": batch_hash(used)},
            "eval_batches": {"count": len(eval_batches), "sha256": batch_hash(eval_batches)},
            "training_pool": {"count": len(pool), "sha256": batch_hash(pool)},
            "model": regime_model(args.regime), "model_family": ctx["family"],
            "provenance": ctx["provenance"], "residual_start": ctx["residual"],
            "chain_s": round(time.time() - t0, 1), "setup_s": round(setup_s, 1),
            "peak_gib": max((c.get(k_, 0.0) for c in cycles for k_ in ("score_peak_gib", "recovery_peak_gib")),
                            default=None),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "script_sha256": sha256_file(Path(__file__).resolve()),
        }
        write_json(path, _jsonable(res))
        log(f"wrote {path} ({res['chain_s']} s, peak {res['peak_gib']} GiB)")
        del opt
        s.model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
