"""Measure real archives at a training-calibrated quality-preserving point.

Reuse the saved staged mask. Compare magnitude selection in the same codec,
using one seed, a fixed calibration gate and explicit weights/Adam state scope.
Completed cases are skipped; failed quality gates remain in the raw record.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--mask-record", type=Path, required=True)
    ap.add_argument("--calibration-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from transformers import GPTNeoXForCausalLM
    from experiments.lib.data import cache_batches, get_data_loaders
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, sha256_file, set_seed, write_json
    from run_residual_sweep_small import global_keep_masks
    from run_system_bytes_comparison import excp_quant, excp_dequant, named_moments

    os.environ["DACP_DATA_ROOT"] = str(args.data_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_info = json.loads(args.mask_record.read_text())
    calibration = json.loads(args.calibration_manifest.read_text())
    mask_path = args.mask_record.parent / mask_info["mask"]["file"]
    identity = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "mask_record_sha256": sha256_file(args.mask_record), "mask_sha256": sha256_file(mask_path),
        "calibration_sha256": sha256_file(args.calibration_manifest),
        "sources": {p.name: sha256_file(p) for p in (Path(__file__), Path(__file__).with_name("run_system_bytes_comparison.py"))},
        "seed": 42, "gate_ppl_tolerance": 0.01,
        "codec_grid": ["kmeans4_joint", "kmeans8_joint", "fp16_lossless_moments"],
        "magnitude_sparsities": [0.7, 0.5],
        "reference_scope": "pretrained base supplied separately; not in archive bytes",
        "state_scope": "all model parameters and buffers, both Adam moments, steps and param_groups",
    }
    source = args.data_root / "models/pythia-410m"
    identity["base_files"] = {p.name: sha256_file(p) for p in sorted(source.glob("*")) if p.is_file()}
    if identity["checkpoint_sha256"] != mask_info["identity"]["checkpoint_sha256"] or identity["mask_sha256"] != mask_info["mask"]["sha256"]:
        raise RuntimeError("mask and input checkpoint differ")
    final_path = args.out_dir / "completed.json"
    if final_path.exists():
        old = json.loads(final_path.read_text())
        if old["identity"] != identity or old["status"] != "completed":
            raise RuntimeError("completed archive experiment identity conflict")
        print("SKIP " + str(final_path), flush=True)
        return
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    set_seed(0)
    train, val, task_type = get_data_loaders("pythia-410m", "alpaca", 2, 128, data_dir=str(args.data_root))
    pool, heldout = cache_batches(train, 140, task_type), cache_batches(val, 100, task_type)
    gate = [pool[i] for i in calibration["gate_pool_indices"]]
    del train, val, pool
    if batch_hash(gate) != mask_info["identity"]["gate_sha256"] or batch_hash(heldout) != mask_info["identity"]["heldout_sha256"]:
        raise RuntimeError("calibration or validation data differ")
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = GPTNeoXForCausalLM.from_pretrained(source, dtype=torch.float32, local_files_only=True)
    base = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    current = raw["model_state_dict"]
    model.load_state_dict(current, strict=True)
    model.to("cuda:0").eval()
    optimizer = raw["optimizer_state_dict"]
    m, v = named_moments(model, optimizer)
    names = list(base)
    metadata = {"param_groups": optimizer["param_groups"],
                "state": {i: {k: x for k, x in state.items() if k not in ("exp_avg", "exp_avg_sq")}
                          for i, state in optimizer["state"].items()}}
    buffers = {n: x for n, x in current.items() if n not in base}
    staged = torch.load(mask_path, map_location="cpu", weights_only=True)
    records, selected = [], {}
    eligible_names = list(staged)
    for method in ("iterative_ucb_disjoint", "excp_style"):
        for sparsity in ([0.7] if method == "iterative_ucb_disjoint" else identity["magnitude_sparsities"]):
            if method == "iterative_ucb_disjoint":
                keep = staged
                score_s = sum(x["score_s"] + x["select_s"] for x in mask_info["stage_records"])
            else:
                began = time.perf_counter()
                scores = {n: (current[n].float() - base[n]).abs() for n in eligible_names}
                (_, keep, _), = list(global_keep_masks(scores, eligible_names, [sparsity], "cuda:0"))
                keep = {n: x.cpu() for n, x in keep.items()}
                score_s = time.perf_counter() - began
                del scores
            for mode in identity["codec_grid"]:
                case = f"{method}_p{sparsity}_{mode}"
                record_path = args.out_dir / f"{case}.json"
                archive_path = args.out_dir / f"{case}.zlib"
                if record_path.exists():
                    rec = json.loads(record_path.read_text())
                    if rec["identity"] != identity or rec["status"] != "completed" or sha256_file(archive_path) != rec["archive"]["sha256"]:
                        raise RuntimeError("archive case identity conflict")
                    print("SKIP " + case, flush=True)
                else:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    payload = {"model": {}, "exp_avg": {}, "exp_avg_sq": {}, "buffers": buffers,
                               "optimizer_metadata": metadata, "parameter_names": names, "step": raw["step"],
                               "codec": mode, "identity": identity}
                    for i, n in enumerate(names):
                        residual = (current[n].float() - base[n]).to("cuda:0")
                        if n in keep:
                            residual.mul_(keep[n].to("cuda:0"))
                        if mode == "fp16_lossless_moments":
                            payload["model"][n] = {"raw": residual.half().cpu()}
                            payload["exp_avg"][n] = m[n]
                            payload["exp_avg_sq"][n] = v[n]
                        else:
                            bits = 4 if mode == "kmeans4_joint" else 8
                            payload["model"][n] = excp_quant(residual, 42 * 100003 + 3 * i, bits)
                            moment = m[n].to("cuda:0")
                            joint = (moment.abs() > 2 * moment.abs().mean()) & (residual != 0)
                            payload["exp_avg"][n] = excp_quant(moment * joint, 42 * 100003 + 3 * i + 1)
                            payload["exp_avg_sq"][n] = excp_quant(v[n].to("cuda:0") * joint, 42 * 100003 + 3 * i + 2)
                        del residual
                    stream = io.BytesIO()
                    torch.save(payload, stream)
                    del payload
                    encoded = zlib.compress(stream.getvalue(), 6)
                    serialized_bytes = stream.tell()
                    del stream
                    temporary = archive_path.with_suffix(".tmp")
                    temporary.write_bytes(encoded)
                    temporary.replace(archive_path)
                    del encoded
                    torch.cuda.synchronize()
                    encode_s = time.perf_counter() - began
                    began = time.perf_counter()
                    payload = torch.load(io.BytesIO(zlib.decompress(archive_path.read_bytes())), weights_only=True, map_location="cpu")
                    params = dict(model.named_parameters())
                    decoded_optimizer = {"param_groups": payload["optimizer_metadata"]["param_groups"], "state": {}}
                    with torch.no_grad():
                        for i, n in enumerate(names):
                            residual = excp_dequant(payload["model"][n], base[n].shape, "cuda:0").float()
                            params[n].copy_(base[n].to("cuda:0") + residual)
                            moments = {k: (payload[k][n].float() if mode == "fp16_lossless_moments"
                                           else excp_dequant(payload[k][n], base[n].shape, "cpu"))
                                       for k in ("exp_avg", "exp_avg_sq")}
                            if any(not bool(x.isfinite().all()) for x in moments.values()) or bool((moments["exp_avg_sq"] < 0).any()):
                                raise RuntimeError("invalid decoded optimizer state")
                            decoded_optimizer["state"][i] = {**payload["optimizer_metadata"]["state"][i], **moments}
                            del residual
                    model.load_state_dict({**model.state_dict(), **payload["buffers"]}, strict=True)
                    torch.cuda.synchronize()
                    decode_s = time.perf_counter() - began
                    gate_result = evaluate_lm(model, gate, "cuda:0")
                    delta = gate_result["loss"] - calibration["baseline_gate"]["loss"]
                    passed = math.isfinite(delta) and delta <= math.log1p(0.01)
                    validation = evaluate_lm(model, heldout, "cuda:0") if passed else None
                    rec = {"status": "completed", "identity": identity, "code_commit": commit,
                           "method": method, "sparsity": sparsity, "codec": mode, "seed": 42,
                           "quantization": {"weight_bits": 4 if mode == "kmeans4_joint" else 8 if mode == "kmeans8_joint" else 16,
                                            "moment_bits": 32 if mode == "fp16_lossless_moments" else 4,
                                            "joint_moment_beta": None if mode == "fp16_lossless_moments" else 2},
                           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                           "baseline_gate_reused": calibration["baseline_gate"],
                           "baseline_validation_reused": mask_info["baseline"],
                           "gate": gate_result, "gate_relative_ppl_change": math.expm1(delta), "passed_gate": passed,
                           "validation": validation,
                           "validation_relative_ppl_change": math.expm1(validation["loss"] - mask_info["baseline"]["val"]) if validation else None,
                           "archive": {"file": archive_path.name, "bytes": archive_path.stat().st_size,
                                       "sha256": sha256_file(archive_path), "torch_serialized_bytes": serialized_bytes,
                                       "original_checkpoint_bytes": args.checkpoint.stat().st_size,
                                       "byte_compression": args.checkpoint.stat().st_size / archive_path.stat().st_size},
                           "timing": {"selection_s": score_s, "selection_reused": method == "iterative_ucb_disjoint",
                                      "encode_and_write_s": encode_s, "read_and_decode_s": decode_s},
                           "eligible_kept": sum(int(x.sum()) for x in keep.values()),
                           "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                           "decoded_optimizer_states": len(decoded_optimizer["state"])}
                    write_json(record_path, rec)
                    del payload, decoded_optimizer
                    print(f"RESULT {case} gate_ppl={rec['gate_relative_ppl_change']:.6f} bytes={rec['archive']['bytes']}", flush=True)
                records.append({"file": record_path.name, "sha256": sha256_file(record_path)})
                if rec["passed_gate"]:
                    selected[method] = records[-1]
                    break
            if method in selected:
                break
    write_json(final_path, {"status": "completed", "identity": identity, "code_commit": commit,
                           "records": records, "selected": selected})
    print("COMPLETED " + str(final_path), flush=True)


if __name__ == "__main__":
    main()
