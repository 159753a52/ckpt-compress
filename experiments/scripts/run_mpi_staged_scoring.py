"""MPI execution of staged DACP scoring: one stage per rank as in Algorithm 2 of the paper.

Every MPI rank is one replica group. It holds the model, the checkpoint and the
reference, and computes the attributions c = -g * d (DACP-G), or with the
finite-difference curvature term (DACP-H), on its own share of the scoring
batches of a stage. The ranks then communicate in two ways only:

* two weight-sized all-reduces merge the batch statistics: the batch mean of the
  attributions, and their second moment centred at that mean (the centred form
  of the sums u and v; it avoids the cancellation in v - u*u/B);
* scalar collectives select the deletions: the eligible entries are split into
  contiguous shards, one owner per shard; each owner counts its entries whose
  score is at most a candidate threshold, the counts are summed with an
  all-reduce of one integer, and bisection on the 32-bit pattern of the score
  finds the exact global threshold. Ties at the threshold are resolved in global
  index order with one prefix sum over the owners.

The ranks here are data-parallel replicas that hold the whole model, so every
rank applies the threshold and the tie quotas to all shards to restore the next
partial model; with model-parallel shards only the owner would. No score, mask or
index array is ever communicated.

With one rank the script is the serial algorithm. --check-serial reruns the
selection with the serial library code (run_iterative with global_keep_masks)
on rank 0 and reports the entries on which the two masks differ.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="result JSON written by rank 0")
    ap.add_argument("--sparsity", type=float, required=True)
    ap.add_argument("--stages", type=int, default=8)
    ap.add_argument("--batches", type=int, default=8, help="scoring batches per stage, over all ranks")
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--shared", action="store_true", help="every stage rescoring the same batches")
    ap.add_argument("--variant", choices=("g", "h"), default="g")
    ap.add_argument("--eta", type=float, default=0.1, help="step of the finite-difference HVP of DACP-H")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--check-serial", action="store_true")
    ap.add_argument("--save-mask", type=Path)
    ap.add_argument("--reference-mask", type=Path, help="mask of another run to compare with")
    args = ap.parse_args()

    import numpy as np
    import torch
    from mpi4py import MPI
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from run_residual_ablation_small import geometric_schedule, grad_moments, run_iterative
    from run_residual_sweep_small import REGIMES, TINY, _count_true, load_regime, now
    from experiments.lib.residual_runtime import batch_hash, lm_loss, set_seed, sha256_file, write_json
    from experiments.lib.residual_training import partition_repeated_seed_batches

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    dev = args.device
    B, K, kappa = args.batches, args.stages, args.kappa
    if B < max(2, size):
        raise SystemExit("--batches must be at least 2 and at least the number of ranks")

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    t_setup = time.perf_counter()
    ctx = load_regime("R1", args.checkpoint.parent, args.data_root, dev)
    sweep, pool = ctx["sweep"], ctx["pool"]
    names = sweep.names

    # Scoring batches: the construction of run_v100_select_masks_h5.py (16 initial batches, then 48 further ones).
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, 42)
    used = plan.selected_pool_indices[len(plan.training_segments[0]):][:16]
    rest = [i for i in range(len(pool)) if i not in set(used)]
    extra = rest[64:]
    random.Random(42).shuffle(extra)
    scoring = plan.scoring_batches[0] + [pool[i] for i in extra[:48]]
    need_batches = B if args.shared else K * B
    if need_batches > len(scoring):
        raise SystemExit(f"{need_batches} scoring batches requested, {len(scoring)} available")

    # Contiguous shards of the eligible entries, one owner each, of nearly equal size.
    shapes = {n: sweep.delta[n].shape for n in names}
    sizes = {n: sweep.delta[n].numel() for n in names}
    N = sum(sizes.values())
    shard_names = [[] for _ in range(size)]
    seen = 0
    for n in names:
        shard_names[min(size - 1, int((seen + sizes[n] / 2) * size / N))].append(n)
        seen += sizes[n]
    sched = geometric_schedule(args.sparsity, K)
    setup_seconds = comm.allreduce(time.perf_counter() - t_setup, op=MPI.MAX)
    log(f"ranks={size} N={N} tensors={len(names)} shard_entries={[sum(sizes[n] for n in s) for s in shard_names]} "
        f"variant={args.variant} K={K} B={B} kappa={kappa:g} shared={args.shared} setup={setup_seconds:.1f}s")

    def stage_batches(j: int):
        return scoring[:B] if args.shared else scoring[j * B:(j + 1) * B]

    # ---------------------------------------------------------------- local statistics

    def local_moments(bs, keep_prev):
        """Running mean and centred second moment of c over this rank's batches at the current model."""
        d = sweep.delta_for_scoring()
        mean = [torch.zeros_like(x, device=dev) for x in d]
        m2 = [torch.zeros_like(x, device=dev) for x in d]

        def accumulate(b: int, i: int, c: torch.Tensor) -> None:
            diff = c - mean[i]
            mean[i].add_(diff, alpha=1.0 / (b + 1))
            m2[i].add_(diff * (c - mean[i]))

        if args.variant == "g":
            def visit(b, grads):
                for i, g in enumerate(grads):
                    accumulate(b, i, -g * sweep.dev(d[i]))

            sweep.batch_gradients(bs, visit)
            return mean, m2

        # DACP-H: one more gradient at theta + eta * d_S, where d_S is the surviving residual.
        model, params = sweep.model, [sweep.params[n] for n in names]
        saved = {n: p.requires_grad for n, p in sweep.params.items()}
        eligible = set(names)
        for n, p in sweep.params.items():
            p.requires_grad_(n in eligible)
        was_training = model.training
        model.eval()
        try:
            for b, batch in enumerate(bs):
                with sdpa_kernel(SDPBackend.MATH):
                    g = torch.autograd.grad(lm_loss(model, batch, dev), params)
                with torch.no_grad():
                    for i, n in enumerate(names):
                        ds = sweep.dev(d[i]) if keep_prev is None else sweep.dev(d[i]) * keep_prev[n]
                        params[i].add_(ds.to(params[i].dtype), alpha=args.eta)
                with sdpa_kernel(SDPBackend.MATH):
                    g2 = torch.autograd.grad(lm_loss(model, batch, dev), params)
                sweep.apply(keep_prev)  # exact return to the partial model
                for i in range(len(names)):
                    di = sweep.dev(d[i])
                    accumulate(b, i, -g[i].float() * di + 0.5 * di * (g2[i].float() - g[i].float()) / args.eta)
                del g, g2
        finally:
            for n, p in sweep.params.items():
                p.requires_grad_(saved[n])
            model.train(was_training)
        return mean, m2

    # ---------------------------------------------------------------- weight-sized all-reduces

    def allreduce_sum(parts):
        """Element-wise sum over the ranks of a list of device tensors; returns device tensors and bytes sent."""
        out, sent = [], 0
        for x in parts:
            if size == 1:
                out.append(x)
                continue
            send = np.ascontiguousarray(x.detach().cpu().numpy())
            recv = np.empty_like(send)
            comm.Allreduce(send, recv, op=MPI.SUM)
            sent += send.nbytes
            out.append(torch.from_numpy(recv).to(dev))
        return out, sent

    def merged_scores(mean, m2, n_local: int, keep_prev):
        weight = n_local / B
        gmean, bytes1 = allreduce_sum([m * weight for m in mean])
        centred = [m2[i] + n_local * (mean[i] - gmean[i]).square() for i in range(len(names))]
        gm2, bytes2 = allreduce_sum(centred)
        scores = {}
        for i, n in enumerate(names):
            sc = gmean[i].abs() + (kappa * (gm2[i] / (B - 1)).clamp_min(0).sqrt() / math.sqrt(B) if kappa else 0)
            sc = sc.to(torch.float32) + TINY
            scores[n] = sc if keep_prev is None else torch.where(keep_prev[n], sc, torch.zeros_like(sc))
        return scores, bytes1 + bytes2

    # ---------------------------------------------------------------- selection with scalar collectives

    def shard_keys(scores, r: int) -> torch.Tensor:
        """Monotone int32 keys of the non-negative float32 scores of shard r, in global index order."""
        flat = torch.cat([scores[n].reshape(-1) for n in shard_names[r]])
        flat.add_(0.0)
        return flat.view(torch.int32)

    def select(scores, k: int):
        """Keep masks that delete the k lowest-scoring entries; returns the masks and the collective count."""
        mine = shard_keys(scores, rank)
        if not bool(torch.isfinite(mine.view(torch.float32)).all()) or bool((mine < 0).any()):
            raise ValueError("scores must be finite and non-negative")
        lo, hi, scalars = 0, 2 ** 31 - 1, 0
        while lo < hi:  # smallest key t with at least k entries of key <= t
            mid = (lo + hi) // 2
            below = comm.allreduce(_count_true(mine <= mid), op=MPI.SUM)
            scalars += 1
            if below >= k:
                hi = mid
            else:
                lo = mid + 1
        t = lo
        need = k - comm.allreduce(_count_true(mine < t), op=MPI.SUM)
        ties = comm.allgather(_count_true(mine == t))  # prefix sum of the tie counts over the owners
        scalars += 2
        quota = []
        for r in range(size):
            quota.append(max(0, min(ties[r], need)))
            need -= quota[-1]
        if need:
            raise RuntimeError("tie resolution missed the deletion budget")
        keep, removed = {}, 0
        for r in range(size):
            if not shard_names[r]:
                continue
            keys = mine if r == rank else shard_keys(scores, r)
            remove = keys < t
            if quota[r]:
                remove[torch.nonzero(keys == t, as_tuple=False).flatten()[:quota[r]]] = True
            removed += _count_true(remove)
            for n, part in zip(shard_names[r], remove.split([sizes[n] for n in shard_names[r]])):
                keep[n] = (~part).view(shapes[n])
            del keys, remove
        if removed != k:
            raise RuntimeError("selection missed the deletion budget")
        return keep, scalars, t

    # ---------------------------------------------------------------- staged run

    def staged_parallel():
        keep_prev, records = None, []
        comm.Barrier()
        t_all = now()
        for j, sp in enumerate(sched):
            k = math.floor(sp * N)
            bs = stage_batches(j)[rank::size]
            comm.Barrier()
            t0 = now()
            sweep.apply(keep_prev)
            mean, m2 = local_moments(bs, keep_prev)
            local_s = now() - t0
            comm.Barrier()
            t1 = now()
            scores, sent = merged_scores(mean, m2, len(bs), keep_prev)
            del mean, m2
            comm.Barrier()
            t2 = now()
            keep, scalars, t = select(scores, k)
            del scores
            comm.Barrier()
            t3 = now()
            if keep_prev is not None and any(bool((~keep_prev[n] & keep[n]).any()) for n in names):
                raise RuntimeError("staged deletion is not monotone")
            keep_prev = keep
            records.append({
                "stage": j + 1, "sparsity": sp, "deleted": k, "threshold_key": t,
                "gradient_seconds": t1 - t0, "gradient_seconds_local_max": comm.allreduce(local_s, op=MPI.MAX),
                "gradient_seconds_local_min": comm.allreduce(local_s, op=MPI.MIN),
                "allreduce_seconds": t2 - t1, "selection_seconds": t3 - t2,
                "allreduce_bytes_per_rank": sent, "scalar_collectives": scalars,
                "scalar_bytes_per_rank": 8 * scalars, "batches_this_rank": len(bs)})
            log("STAGE " + json.dumps(records[-1]))
        comm.Barrier()
        return keep_prev, records, now() - t_all

    set_seed(42)
    torch.cuda.reset_peak_memory_stats()
    keep, records, seconds = staged_parallel()
    peak = comm.allreduce(torch.cuda.max_memory_allocated() / 2 ** 30, op=MPI.MAX)

    if rank != 0:
        del keep
        sweep.model.to("cpu")  # leave the device to rank 0 when the ranks share one GPU
        sweep.cur_dev.clear()
        sweep.ref_dev.clear()
        torch.cuda.empty_cache()
        comm.Barrier()  # wait until rank 0 has finished its evaluation and the optional serial check
        return

    sweep.apply(keep)
    heldout = sweep.evaluate()
    sweep.apply(None)
    result = {
        "ranks": size, "variant": args.variant, "stages": K, "batches": B, "kappa": kappa, "shared": args.shared,
        "eta": args.eta if args.variant == "h" else None, "sparsity": args.sparsity, "eligible_entries": N,
        "shard_entries": [sum(sizes[n] for n in s) for s in shard_names],
        "seconds": seconds, "setup_seconds": setup_seconds, "peak_gib_per_rank": peak,
        "gradient_seconds": sum(r["gradient_seconds"] for r in records),
        "allreduce_seconds": sum(r["allreduce_seconds"] for r in records),
        "selection_seconds": sum(r["selection_seconds"] for r in records),
        "allreduce_bytes_per_rank": sum(r["allreduce_bytes_per_rank"] for r in records),
        "scalar_collectives": sum(r["scalar_collectives"] for r in records),
        "gathered_scores_bytes_per_stage": 4 * N,
        "baseline": ctx["uncompressed"], "heldout": heldout,
        "validation_delta_loss": heldout["val"] - ctx["uncompressed"]["val"],
        "validation_ppl_change_percent": 100 * math.expm1(heldout["val"] - ctx["uncompressed"]["val"]),
        "checkpoint_sha256": ctx["provenance"]["checkpoint_sha256"],
        "scoring_sha256": batch_hash(scoring[:need_batches]), "script_sha256": sha256_file(Path(__file__)),
        "gpu": torch.cuda.get_device_name(0), "mpi": MPI.Get_library_version().splitlines()[0],
        "stage_records": records}

    def differing(a, b) -> int:
        return sum(_count_true(a[n].to(dev) != b[n].to(dev)) for n in names)

    if args.save_mask:
        args.save_mask.parent.mkdir(parents=True, exist_ok=True)
        torch.save({n: x.cpu() for n, x in keep.items()}, args.save_mask)
        result["mask_sha256"] = sha256_file(args.save_mask)
    if args.reference_mask:
        ref = torch.load(args.reference_mask, map_location="cpu")
        result["reference_mask"] = {"path": str(args.reference_mask), "differing_entries": differing(keep, ref)}
        del ref
    if args.check_serial:
        if args.variant != "g":
            raise SystemExit("--check-serial covers DACP-G")
        mine = {n: x.cpu() for n, x in keep.items()}
        del keep
        torch.cuda.empty_cache()
        stage = 0

        def scorer(current, unused, keep_prev, seed):
            nonlocal stage
            bs = stage_batches(stage)
            stage += 1
            mean, m2, losses = grad_moments(current, bs, kappa > 0)
            return ({n: mean[i].abs() + (kappa * (m2[i] / (B - 1)).clamp_min(0).sqrt() / math.sqrt(B) if kappa else 0)
                     for i, n in enumerate(current.names)}, {"score_losses": losses})

        set_seed(42)
        began = now()
        _, _, serial = run_iterative(sweep, scoring, sched, scorer, 42, eval_at=[], return_keep=True)
        serial_seconds = now() - began
        sweep.apply(serial)
        serial_eval = sweep.evaluate()
        sweep.apply(None)
        result["serial_check"] = {
            "differing_entries": differing(mine, serial), "deleted_entries": math.floor(args.sparsity * N),
            "serial_seconds": serial_seconds, "serial_heldout": serial_eval,
            "serial_validation_ppl_change_percent": 100 * math.expm1(serial_eval["val"] - ctx["uncompressed"]["val"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    log("RESULT " + json.dumps({k: v for k, v in result.items() if k != "stage_records"}))
    comm.Barrier()


if __name__ == "__main__":
    main()
