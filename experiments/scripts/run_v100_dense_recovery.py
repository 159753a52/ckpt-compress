"""Dense validation measurements for checkpoint recovery on one V100.

Reuses the saved Pythia-410M checkpoint, data order, selectors and AdamW
recovery implementation. New measurement protocol: three 100-step cycles,
validation every ten steps, one scoring seed. Completed identities are skipped;
the latest completed cycle saves model and optimizer state for restart.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sparsity", type=float, default=0.98)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--recovery", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--methods", default="no_compression,excp_style,inshrinkerator_style,dacp,first_order_ucb,iterative_ucb_disjoint")
    ap.add_argument("--initial-mask-record", type=Path)
    ap.add_argument("--initial-archive-record", type=Path)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    methods = args.methods.split(",")
    allowed = {"no_compression", "excp_style", "inshrinkerator_style", "dacp", "first_order_ucb", "iterative_ucb_disjoint"}
    if set(methods) - allowed or len(set(methods)) != len(methods):
        ap.error("unknown or repeated method")
    if not 0 < args.sparsity < 1 or min(args.cycles, args.recovery, args.eval_every) < 1:
        ap.error("invalid sparsity or step count")
    config = {key: getattr(args, key) for key in ("seed", "sparsity", "cycles", "recovery", "eval_every", "lr")}
    config.update(train_batch_size=4, train_seq_length=512, micro_batch_size=2, eval_batches=100,
                  scoring_pool_batches=16, gradient_baseline_batches=8,
                  staged_groups="four disjoint groups of four", amp=False)
    if args.plan_only:
        print(json.dumps({"protocol": config, "methods": methods, "purpose": "new dense recovery measurements"}))
        return
    if any(getattr(args, key) is None for key in ("checkpoint", "data_root", "out_dir")):
        ap.error("--checkpoint, --data-root and --out-dir are required")

    import torch
    from run_base_chains_small import evaluate, recovery_data, refresh_current, restore, select_keep
    from run_residual_ablation_small import geometric_schedule, grad_moments, run_iterative
    from run_residual_sweep_small import REGIMES, load_regime, _jsonable
    from experiments.lib.residual_runtime import batch_hash, load_training_checkpoint, set_seed, sha256_file, write_json
    from experiments.lib.residual_training import build_optimizer, partition_repeated_seed_batches
    from experiments.scripts.finetune.finetune_pythia_410m import train_step_accum

    args.out_dir.mkdir(parents=True, exist_ok=True)
    partial = args.out_dir / "partial"
    partial.mkdir(exist_ok=True)
    source_files = [Path(__file__), ROOT / "experiments/scripts/run_base_chains_small.py",
                    ROOT / "experiments/scripts/run_residual_ablation_small.py",
                    ROOT / "experiments/scripts/run_residual_sweep_small.py"]
    protocol = {"config": config, "checkpoint_sha256": sha256_file(args.checkpoint),
                "sources": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_files},
                "input_files": {str(p.relative_to(args.data_root)): sha256_file(p)
                                for p in sorted((args.data_root / "models/pythia-410m").glob("*")) if p.is_file()}}
    alpaca = args.data_root / "alpaca/alpaca_data.json"
    protocol["input_files"]["alpaca/alpaca_data.json"] = sha256_file(alpaca)
    mask_record = None
    if args.initial_mask_record is not None:
        mask_record = json.loads(args.initial_mask_record.read_text())
        info = mask_record["identity"]
        if info["checkpoint_sha256"] != protocol["checkpoint_sha256"] or info["sparsity"] != args.sparsity or info["seed"] != args.seed:
            raise RuntimeError("initial mask does not match the checkpoint, sparsity or seed")
        protocol["initial_mask_record_sha256"] = sha256_file(args.initial_mask_record)
    archive_record = None
    if args.initial_archive_record is not None:
        if len(methods) != 1 or methods[0] not in {"iterative_ucb_disjoint", "excp_style"} or args.cycles != 1:
            ap.error("archive continuation requires one selector and one cycle")
        archive_record = json.loads(args.initial_archive_record.read_text())
        if archive_record["identity"]["checkpoint_sha256"] != protocol["checkpoint_sha256"] or archive_record["sparsity"] != args.sparsity or not archive_record["passed_gate"]:
            raise RuntimeError("archive does not match the calibrated operating point")
        archive_path = args.initial_archive_record.parent / archive_record["archive"]["file"]
        if sha256_file(archive_path) != archive_record["archive"]["sha256"]:
            raise RuntimeError("archive hash differs")
        protocol["initial_archive_record_sha256"] = sha256_file(args.initial_archive_record)
        protocol["initial_archive_sha256"] = archive_record["archive"]["sha256"]
    todo = []
    for method in methods:
        path = args.out_dir / f"{method}_p{args.sparsity}_seed{args.seed}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("protocol") != protocol or old.get("status") != "completed":
                raise RuntimeError(f"Existing result identity conflict: {path}")
            print("SKIP " + str(path), flush=True)
        else:
            todo.append((method, path))
    if not todo:
        return

    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    context = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    sweep, pool = context["sweep"], context["pool"]
    checkpoint = load_training_checkpoint(args.checkpoint)
    state, optimizer_state, start_step = checkpoint.model_state, checkpoint.optimizer_state, checkpoint.step
    ref_cpu = {name: sweep.ref_dev[name].float().cpu() for name in sweep.names}
    scoring = partition_repeated_seed_batches(pool, 100, 1, 16, args.seed).scoring_batches[0]
    recovery, data_meta = recovery_data("pythia-410m", args.data_root, 4, 512, args.cycles * args.recovery, pool)
    if not data_meta["pool_alignment_verified"]:
        raise RuntimeError("Recovery data do not follow the scoring pool")
    data_meta["chunk_sha256"] = [batch_hash(recovery[i * args.recovery:(i + 1) * args.recovery]) for i in range(args.cycles)]
    identity = {"scoring_sha256": batch_hash(scoring), "evaluation_sha256": batch_hash(context["eval_batches"]),
                "training_pool_sha256": batch_hash(pool), "recovery": data_meta}
    if mask_record is not None and (mask_record["identity"]["scoring_sha256"] != identity["scoring_sha256"] or mask_record["identity"]["heldout_sha256"] != identity["evaluation_sha256"]):
        raise RuntimeError("initial mask scoring/evaluation data differ")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(args.out_dir / "run-manifest.json", {"protocol": protocol, "identity": identity,
               "code_commit": commit, "methods": methods, "gpu": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "source_provenance": context["provenance"]})
    print("SETUP " + json.dumps(identity), flush=True)
    resume_path = args.out_dir / "latest-cycle.pt"

    for method, path in todo:
        began = time.time()
        sweep.model.load_state_dict(state, strict=True)
        opt = build_optimizer(sweep.model, optimizer_state, learning_rate=args.lr)
        record = {"status": "running", "purpose": "new dense V100 recovery measurements",
                  "protocol": protocol, "identity": identity, "method": method,
                  "sparsity": None if method == "no_compression" else args.sparsity,
                  "seed": args.seed, "model": "pythia-410m", "dataset": "alpaca", "reference": "base",
                  "start_step": start_step, "cycles": [], "source_provenance": context["provenance"],
                  "code_commit": commit, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
        if resume_path.exists():
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            if saved["record"]["method"] == method:
                if saved["record"]["protocol"] != protocol or saved["record"]["identity"] != identity:
                    raise RuntimeError("Resume identity conflict")
                sweep.model.load_state_dict(saved["model"], strict=True)
                opt.load_state_dict(saved["optimizer"])
                record = saved["record"]
                print(f"RESUME {method} after cycle {len(record['cycles'])}", flush=True)
            del saved
        refresh_current(sweep, ref_cpu)
        if not record["cycles"]:
            record["uncompressed"] = evaluate(sweep)
        for cycle in range(len(record["cycles"]) + 1, args.cycles + 1):
            refresh_current(sweep, ref_cpu)
            step = start_step + (cycle - 1) * args.recovery
            rec = {"cycle": cycle, "step": step, "validation_trace": [], "train_log": []}
            set_seed(args.seed)
            torch.cuda.reset_peak_memory_stats()
            if method != "no_compression":
                t0 = time.perf_counter()
                reuse_initial = method == "iterative_ucb_disjoint" and cycle == 1 and mask_record is not None
                if archive_record is not None:
                    rec["score"] = {"reused_from": str(args.initial_archive_record),
                                    "backward_budget": archive_record.get("backward_budget"),
                                    "note": "decoded archive contains the selected weights; no scoring repeated"}
                elif reuse_initial:
                    mask_path = args.initial_mask_record.parent / mask_record["mask"]["file"]
                    if sha256_file(mask_path) != mask_record["mask"]["sha256"]:
                        raise RuntimeError("initial mask artifact hash differs")
                    keep = torch.load(mask_path, map_location="cpu", weights_only=True)
                    rec["score"] = {"stages": mask_record["stage_records"], "backward_budget": 16,
                                    "unique_scoring_batches": 16, "reused_from": str(args.initial_mask_record),
                                    "reused_source_sha256": protocol["initial_mask_record_sha256"]}
                elif method == "iterative_ucb_disjoint":
                    stage = 0

                    def scorer(current, unused, keep_prev, seed):
                        nonlocal stage
                        selected = scoring[stage * 4:(stage + 1) * 4]
                        stage += 1
                        mean, m2, losses = grad_moments(current, selected, True)
                        scores = {name: mean[i].abs() + (m2[i] / 3).clamp_min(0).sqrt() / 2
                                  for i, name in enumerate(current.names)}
                        return scores, {"score_losses": losses}

                    _, stages, keep = run_iterative(sweep, scoring, geometric_schedule(args.sparsity, 4),
                                                    scorer, args.seed, eval_at=[], return_keep=True)
                    rec["score"] = {"stages": stages, "backward_budget": 16, "unique_scoring_batches": 16}
                    sweep.apply(None)
                else:
                    keep, rec["score"] = select_keep(sweep, method, args.sparsity, scoring[:8], scoring, args.seed, 16, 4)
                rec["score_total_s"] = time.perf_counter() - t0
                rec["restore"] = restore(sweep, keep)
                del keep
            if archive_record is not None:
                import io
                import zlib
                from transformers import GPTNeoXForCausalLM
                from run_system_bytes_comparison import excp_dequant
                payload = torch.load(io.BytesIO(zlib.decompress(archive_path.read_bytes())), map_location="cpu", weights_only=True)
                base_model = GPTNeoXForCausalLM.from_pretrained(args.data_root / "models/pythia-410m", dtype=torch.float32, local_files_only=True)
                base = dict(base_model.named_parameters())
                params = dict(sweep.model.named_parameters())
                decoded = {"param_groups": payload["optimizer_metadata"]["param_groups"], "state": {}}
                with torch.no_grad():
                    for index, name in enumerate(payload["parameter_names"]):
                        shape = base[name].shape
                        params[name].copy_(base[name].to("cuda:0") + excp_dequant(payload["model"][name], shape, "cuda:0").float())
                        moments = {key: (payload[key][name].float() if payload["codec"] == "fp16_lossless_moments"
                                         else excp_dequant(payload[key][name], shape, "cpu")) for key in ("exp_avg", "exp_avg_sq")}
                        decoded["state"][index] = {**payload["optimizer_metadata"]["state"][index], **moments}
                sweep.model.load_state_dict({**sweep.model.state_dict(), **payload["buffers"]}, strict=True)
                opt.load_state_dict(decoded)
                for group in opt.param_groups:
                    group["lr"] = args.lr
                value = archive_record["validation"]
                rec["immediate"] = {"val": {"loss": value["loss"], "ppl": value["perplexity"],
                                             "tokens": value["tokens"], "eval_s": value["seconds"]}}
                rec["immediate_reused_from"] = str(args.initial_archive_record)
                rec["decoded_archive"] = archive_record["archive"]
                del payload, decoded, base, base_model
            elif method == "iterative_ucb_disjoint" and cycle == 1 and mask_record is not None:
                value = mask_record["evaluation"]
                rec["immediate"] = {"val": {"loss": value["val"], "ppl": value["ppl"],
                                             "tokens": value["tokens"], "eval_s": value["eval_s"]}}
                rec["immediate_reused_from"] = str(args.initial_mask_record)
            else:
                rec["immediate"] = evaluate(sweep)
            rec["validation_trace"].append({"step": step, **rec["immediate"]})
            train_seconds = 0.0
            chunk = recovery[(cycle - 1) * args.recovery:cycle * args.recovery]
            set_seed(args.seed)
            for offset, batch in enumerate(chunk, 1):
                t0 = time.perf_counter()
                loss = float(train_step_accum(sweep.model, opt, None, batch, "cuda:0", 2, use_amp=False))
                train_seconds += time.perf_counter() - t0
                if not math.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                rec["train_log"].append({"step": step + offset, "loss": loss})
                if offset % args.eval_every == 0 or offset == args.recovery:
                    value = evaluate(sweep)
                    if not math.isfinite(value["val"]["loss"]):
                        raise RuntimeError("Nonfinite validation loss")
                    rec["validation_trace"].append({"step": step + offset, **value})
                    write_json(partial / f"{method}_current.json", _jsonable({**record, "current_cycle": rec}))
                    print(f"TRACE {method} cycle={cycle} step={step + offset} val={value['val']['loss']:.6f}", flush=True)
            rec["final"] = {"val": rec["validation_trace"][-1]["val"]}
            rec["recovery_train_s"] = train_seconds
            rec["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            record["cycles"].append(rec)
            tmp_state = resume_path.with_suffix(".tmp")
            torch.save({"record": record, "model": {n: p.detach().cpu() for n, p in sweep.model.state_dict().items()},
                        "optimizer": opt.state_dict()}, tmp_state)
            tmp_state.replace(resume_path)
            write_json(partial / f"{method}_cycle{cycle}.json", _jsonable(record))
        states = args.out_dir / "final-states"
        states.mkdir(exist_ok=True)
        final_state = states / f"{method}_cycle{args.cycles}.pt"
        if final_state.exists() and not os.path.samefile(final_state, resume_path):
            raise RuntimeError(f"Final state already exists without its completed result: {final_state}")
        if not final_state.exists():
            os.link(resume_path, final_state)
        record.update(status="completed", elapsed_this_process_s=time.time() - began,
                      final_state={"path": str(final_state), "bytes": final_state.stat().st_size})
        write_json(path, _jsonable(record))
        print("RESULT " + str(path), flush=True)
        del opt
        sweep.model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
