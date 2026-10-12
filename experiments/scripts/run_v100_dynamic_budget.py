"""Three recovery cycles of the existing damage-budget/feedback controller.

Uses block Taylor scores, an exact empirical-CDF allocation, a 95% block cap,
one candidate validation per checkpoint, and lossless fallback. The optimizer
is preserved in FP32. This tests pruning/control, not the quantized archive.
"""
from pathlib import Path
import argparse
import json
import math
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--legacy-mask-record", type=Path, required=True)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--cycles", type=int, default=3)
    args = ap.parse_args()
    if not 1 <= args.batches <= 16 or not 1 <= args.cycles <= 3:
        ap.error("scoring batches must be 1..16 and cycles 1..3")
    import torch
    from run_residual_sweep_small import REGIMES, load_regime, global_keep_masks
    from run_base_chains_small import evaluate, refresh_current, restore, recovery_data
    from experiments.lib.residual_runtime import batch_hash, load_training_checkpoint, evaluate_lm, set_seed, sha256_file, write_json
    from experiments.lib.residual_training import partition_repeated_seed_batches, build_optimizer
    from experiments.lib.residual_scoring import compute_block_taylor_scores
    from experiments.lib.residual_dynamic import budget_keep, update_scale
    from experiments.scripts.finetune.finetune_pythia_410m import train_step_accum
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"dynamic_B{args.batches}_seed42"
    path = args.out_dir / f"{name}.json"
    identity = {"checkpoint_sha256": sha256_file(args.checkpoint), "script_sha256": sha256_file(Path(__file__)),
                "batches": args.batches, "cycles": args.cycles, "recovery_steps": 100, "lr": 2e-6,
                "seed": 42, "target_ppl": .005, "acceptance_ppl": .01, "pmax": .9, "block_cap": .95, "rho": .5,
                "dynamic_library_sha256": sha256_file(ROOT / "experiments/lib/residual_dynamic.py")}
    identity["input_files"] = {str(p.relative_to(args.data_root)): sha256_file(p)
                               for p in sorted((args.data_root / "models/pythia-410m").glob("*")) if p.is_file()}
    identity["input_files"]["alpaca/alpaca_data.json"] = sha256_file(args.data_root / "alpaca/alpaca_data.json")
    if path.exists():
        old = json.loads(path.read_text())
        if {k: old["identity"].get(k) for k in identity} != identity or old["status"] != "completed":
            raise RuntimeError("existing dynamic result identity conflict")
        print("SKIP " + name, flush=True)
        return
    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    ctx = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    s, pool = ctx["sweep"], ctx["pool"]
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, 42)
    scoring = plan.scoring_batches[0][:args.batches]
    offset = len(plan.training_segments[0])
    excluded = set(plan.selected_pool_indices[offset:offset + 16])
    gate_indices = [i for i in range(len(pool)) if i not in excluded][:64]
    gate = [pool[i] for i in gate_indices]
    assert not excluded.intersection(gate_indices)
    identity.update(scoring_sha256=batch_hash(scoring), gate_sha256=batch_hash(gate),
                    heldout_sha256=batch_hash(ctx["eval_batches"]))
    checkpoint = load_training_checkpoint(args.checkpoint)
    opt = build_optimizer(s.model, checkpoint.optimizer_state, learning_rate=2e-6)
    ref = {n: s.ref_dev[n].float().cpu() for n in s.names}
    training, training_meta = recovery_data("pythia-410m", args.data_root, 4, 512, 100 * args.cycles, pool)
    assert training_meta["pool_alignment_verified"]
    identity["recovery_batch_hashes"] = [batch_hash(training[i * 100:(i + 1) * 100]) for i in range(args.cycles)]
    baseline = json.loads((ROOT / "results/v100_dense_recovery_20261011/no_compression_p0.98_seed42.json").read_text())
    if baseline["protocol"]["checkpoint_sha256"] != identity["checkpoint_sha256"] or baseline["identity"]["recovery"]["chunk_sha256"][:args.cycles] != identity["recovery_batch_hashes"]:
        raise RuntimeError("cached uncompressed continuation differs")
    legacy = json.loads(args.legacy_mask_record.read_text())
    old_cal = json.loads((ROOT / "results/v100_quality_calibration_20261011/staged_K4_B4_p0.7_seed42.json").read_text())
    if legacy["identity"]["gate_sha256"] != identity["gate_sha256"]:
        raise RuntimeError("warm calibration input differs")
    warm_mask = torch.load(args.legacy_mask_record.parent / legacy["mask"]["file"], map_location="cpu", weights_only=True)
    record = {"status": "running", "identity": identity, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "allocation": "empirical-CDF fallback with exact proxy/count check", "state_scope": "residual pruning; FP32 optimizer retained losslessly; no quantization",
              "cycles": [], "baseline_reused": baseline["code_commit"], "gpu": torch.cuda.get_device_name(0)}
    resume = args.out_dir / f"{name}-resume.pt"
    scale = None
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["record"]["identity"] != identity:
            raise RuntimeError("resume identity conflict")
        s.model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        record, scale = saved["record"], saved["scale"]
        del saved
    eps, eps_max = math.log1p(.005), math.log1p(.01)
    started = time.perf_counter()
    for cycle in range(len(record["cycles"]), args.cycles):
        set_seed(42)
        refresh_current(s, ref)
        gate_original = evaluate_lm(s.model, gate, "cuda:0")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        scores, scoring_meta = compute_block_taylor_scores(s.model, scoring, s.layers, s.delta, "cuda:0",
                             aggregation="abs_mean", model_family=s.family, block_parameter_names=s.blocks)
        score_seconds = time.perf_counter() - t0
        if scale is None:
            warm_q = sum(float((scores[n] * (~warm_mask[n])).double().sum()) for n in s.names)
            scale = max(1e-12, max(0., old_cal["gate_delta_loss"]) / max(warm_q, 1e-20))
            record["warm_start"] = {"source": str(args.legacy_mask_record), "proxy": warm_q, "scale": scale,
                                    "observed_delta_reused": old_cal["gate_delta_loss"]}
        allowed = {}
        for layer in s.layers:
            (_, top_keep, _), = list(global_keep_masks(scores, layer, [.95], "cuda:0"))
            allowed.update({n: ~top_keep[n] for n in layer})
        t0 = time.perf_counter()
        keep, allocation = budget_keep(scores, s.names, eps / scale, .9, allowed, "cuda:0")
        select_seconds = time.perf_counter() - t0
        del scores, allowed
        s.apply(keep)
        candidate = evaluate_lm(s.model, gate, "cuda:0")
        delta = candidate["loss"] - gate_original["loss"]
        accepted = math.isfinite(delta) and delta <= eps_max and allocation["pruned"] > 0
        next_scale = update_scale(scale, allocation["proxy_cost"], delta, .5, 1e-12, eps_max)
        if next_scale is None:
            accepted = False
            next_scale = max(scale * 2, 1e-12)
        if not accepted:
            s.apply(None)
        saved_rate = allocation["sparsity"] if accepted else 0.
        step = checkpoint.step + cycle * 100
        rec = {"cycle": cycle + 1, "step": step, "allocation": allocation, "scale": scale, "next_scale": next_scale,
               "gate_original": gate_original, "gate_candidate": candidate, "gate_delta_loss": delta,
               "gate_ppl_change_percent": 100 * math.expm1(delta), "accepted": accepted,
               "saved_sparsity": saved_rate, "score_seconds": score_seconds, "select_seconds": select_seconds,
               "score_kind": scoring_meta["score_kind"], "validation_trace": [], "train_log": []}
        # Commit the chosen checkpoint and feedback together before recovery.
        snapshot = args.out_dir / f"{name}-chosen.pt"
        temporary = snapshot.with_suffix(".tmp")
        torch.save({"model": {n: x.detach().cpu() for n, x in s.model.state_dict().items()}, "optimizer": opt.state_dict(),
                    "step": step, "controller": {"scale": next_scale, "cycle": cycle + 1},
                    "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, temporary)
        temporary.replace(snapshot)
        rec["checkpoint_commit"] = {"bytes": snapshot.stat().st_size, "sha256": sha256_file(snapshot), "format": "full decoded snapshot; bytes are not a compressed-archive ratio"}
        scale = next_scale
        immediate = evaluate(s)
        rec["validation_trace"].append({"step": step, **immediate})
        rec["immediate"] = immediate
        set_seed(42)
        for offset, batch in enumerate(training[cycle * 100:(cycle + 1) * 100], 1):
            loss = float(train_step_accum(s.model, opt, None, batch, "cuda:0", 2, use_amp=False))
            if not math.isfinite(loss):
                raise RuntimeError("nonfinite recovery loss")
            rec["train_log"].append({"step": step + offset, "loss": loss})
            if offset % 10 == 0:
                value = evaluate(s)
                rec["validation_trace"].append({"step": step + offset, **value})
                write_json(args.out_dir / f"{name}-partial.json", {**record, "current_cycle": rec})
                print(f"TRACE cycle={cycle + 1} step={step + offset} saved_P={saved_rate:.4f} val={value['val']['loss']:.6f}", flush=True)
        rec["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        rec["final"] = {"val": rec["validation_trace"][-1]["val"]}
        record["cycles"].append(rec)
        temporary = resume.with_suffix(".tmp")
        torch.save({"record": record, "scale": scale, "model": {n: x.detach().cpu() for n, x in s.model.state_dict().items()}, "optimizer": opt.state_dict()}, temporary)
        temporary.replace(resume)
        write_json(args.out_dir / f"{name}-cycle{cycle + 1}.json", record)
        print(f"CYCLE {cycle + 1} accepted={accepted} gate_delta={delta:.6f} next_scale={scale:.6g}", flush=True)
    record.update(status="completed", elapsed_seconds_this_process=time.perf_counter() - started)
    write_json(path, record)


if __name__ == "__main__":
    main()
