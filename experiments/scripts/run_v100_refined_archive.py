"""Encode the calibrated new mask while reusing verified BF16 optimizer states."""
from pathlib import Path
import argparse
import io
import json
import math
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--legacy-archive", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--weight-bits", choices=(4, 8), type=int, default=4)
    ap.add_argument("--dense-weight-bits", choices=(4, 8), type=int)
    ap.add_argument("--lossless-vectors", action="store_true")
    ap.add_argument("--shared-state", type=Path, help="store common optimizer state once; every checkpoint counts its full size")
    ap.add_argument("--method", choices=("iterative_ucb_disjoint", "excp_style", "inshrinkerator_style"), default="iterative_ucb_disjoint")
    args = ap.parse_args()
    import torch
    from run_residual_sweep_small import REGIMES, load_regime
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, sha256_file, write_json
    from run_system_bytes_comparison import excp_quant, excp_dequant
    chosen = json.loads(args.selected.read_text())
    old = json.loads(args.legacy_archive.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.out_dir / f"{args.method}_p0.7_bf16mom.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        quant = record["quantization"]
        if quant["weight_bits"] != args.weight_bits or quant.get("dense_weight_bits") != args.dense_weight_bits or quant.get("lossless_vectors", False) != args.lossless_vectors:
            raise RuntimeError("existing archive encoding configuration differs")
        if record["selection_sha256"] != sha256_file(args.selected) or sha256_file(args.out_dir / record["archive"]["file"]) != record["archive"]["sha256"]:
            raise RuntimeError("existing refined archive differs")
        print("SKIP refined archive", flush=True)
        return
    source = args.legacy_archive.parent / old["archive"]["file"]
    if sha256_file(source) != old["archive"]["sha256"] or old["identity"]["checkpoint_sha256"] != sha256_file(args.checkpoint):
        raise RuntimeError("legacy archive identity differs")
    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    ctx = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    s, pool = ctx["sweep"], ctx["pool"]
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, 42)
    offset = len(plan.training_segments[0])
    used = set(plan.selected_pool_indices[offset:offset + 16])
    gate = [pool[i] for i in range(len(pool)) if i not in used][:64]
    if batch_hash(gate) != chosen["identity_common"]["gate_sha256"] or batch_hash(ctx["eval_batches"]) != chosen["identity_common"]["heldout_sha256"]:
        raise RuntimeError("selected calibration and validation inputs differ")
    mask_path = Path(chosen["mask_path"])
    if not mask_path.is_absolute():
        mask_path = args.selected.parent / mask_path.name
    if sha256_file(mask_path) != chosen["mask_sha256"]:
        raise RuntimeError("selected mask hash differs")
    keep = torch.load(mask_path, map_location="cpu", weights_only=True)
    payload = torch.load(io.BytesIO(zlib.decompress(source.read_bytes())), map_location="cpu", weights_only=True)
    from transformers import GPTNeoXForCausalLM
    base_model = GPTNeoXForCausalLM.from_pretrained(args.data_root / "models/pythia-410m", dtype=torch.float32, local_files_only=True)
    base = dict(base_model.named_parameters())
    params = dict(s.model.named_parameters())
    started = time.perf_counter()
    with torch.no_grad():
        for index, name in enumerate(payload["parameter_names"]):
            if name in keep or args.weight_bits == 8 or args.dense_weight_bits is not None or args.lossless_vectors:
                residual = (s.delta[name].to("cuda:0") * keep[name].to("cuda:0") if name in keep
                            else params[name].detach() - base[name].to("cuda:0"))
                if args.lossless_vectors and params[name].ndim < 2:
                    payload["model"][name] = {"raw": residual.cpu()}
                else:
                    bits = args.weight_bits if name in keep else (args.dense_weight_bits or args.weight_bits)
                    payload["model"][name] = excp_quant(residual, 42 * 100003 + 3 * index, bits)
    payload["identity"] = {"checkpoint_sha256": sha256_file(args.checkpoint), "selection_sha256": sha256_file(args.selected)}
    payload["eligible_parameter_names"] = s.names
    if args.shared_state is not None:
        import os
        args.shared_state.parent.mkdir(parents=True, exist_ok=True)
        if not args.shared_state.exists():
            state = {k: v for k, v in payload.items() if k not in {"model", "identity", "eligible_parameter_names"}}
            stream = io.BytesIO()
            torch.save(state, stream)
            args.shared_state.write_bytes(zlib.compress(stream.getvalue(), 6))
            write_json(args.shared_state.with_suffix(".identity.json"),
                       {"checkpoint_sha256": sha256_file(args.checkpoint), "source_archive_sha256": old["archive"]["sha256"]})
            del stream, state
        shared_identity = json.loads(args.shared_state.with_suffix(".identity.json").read_text())
        if shared_identity != {"checkpoint_sha256": sha256_file(args.checkpoint), "source_archive_sha256": old["archive"]["sha256"]}:
            raise RuntimeError("shared optimizer state source differs")
        shared = args.out_dir / "optimizer_state.zlib"
        if not shared.exists():
            os.link(args.shared_state, shared)
        weights = args.out_dir / "weights.zlib"
        stream = io.BytesIO()
        torch.save({"model": payload["model"], "identity": payload["identity"], "eligible_parameter_names": s.names}, stream)
        weights.write_bytes(zlib.compress(stream.getvalue(), 6))
        archive = args.out_dir / f"{args.method}_p0.7.manifest.json"
        write_json(archive, {"format": "checkpoint-components-v1",
                   "state": {"file": shared.name, "sha256": sha256_file(shared)},
                   "weights": {"file": weights.name, "sha256": sha256_file(weights)}})
        archive_bytes = shared.stat().st_size + weights.stat().st_size + archive.stat().st_size
    else:
        stream = io.BytesIO()
        torch.save(payload, stream)
        archive = args.out_dir / f"{args.method}_p0.7_bf16mom.zlib"
        archive.write_bytes(zlib.compress(stream.getvalue(), 6))
        archive_bytes = archive.stat().st_size
    encode_seconds = time.perf_counter() - started
    # Reopen the saved bytes; quality measurements use decoded weights.
    del payload, stream
    from experiments.lib.calibrated_archive import load_payload
    payload = load_payload(archive)
    with torch.no_grad():
        for name, encoded in payload["model"].items():
            params[name].copy_(base[name].to("cuda:0") + excp_dequant(encoded, base[name].shape, "cuda:0").float())
    s.model.load_state_dict({**s.model.state_dict(), **payload["buffers"]}, strict=True)
    calibration = evaluate_lm(s.model, gate, "cuda:0")
    validation = evaluate_lm(s.model, ctx["eval_batches"], "cuda:0")
    old_gate = json.loads((ROOT / "results/v100_quality_calibration_20261011/staged_K4_B4_p0.7_seed42.json").read_text())["baseline_gate"]
    delta = calibration["loss"] - old_gate["loss"]
    record = {"status": "completed", "identity": payload["identity"], "method": args.method, "sparsity": .7, "seed": 42,
              "selection_sha256": sha256_file(args.selected), "config": chosen["config"], "backward_budget": chosen["backward_budget"],
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "archive": {"file": archive.name, "bytes": archive_bytes, "sha256": sha256_file(archive),
                          "format": "components; optimizer, weights and manifest all counted" if args.shared_state else "single zlib"},
              "byte_compression": args.checkpoint.stat().st_size / archive_bytes,
              "gate": calibration, "gate_ppl_change_percent": 100 * math.expm1(delta), "passed_gate": delta <= math.log1p(.01),
              "validation": validation, "validation_ppl_change_percent": 100 * math.expm1(validation["loss"] - ctx["uncompressed"]["val"]),
              "quantization": {"weight_bits": args.weight_bits, "moment_bits": 16, "moment_dtype": "bfloat16", "all_moments_retained": True},
              "optimizer_payload_reused": {"source": str(args.legacy_archive), "sha256": old["archive"]["sha256"]},
              "encode_seconds": encode_seconds, "baseline_validation": ctx["uncompressed"]}
    write_json(record_path, record)
    if args.dense_weight_bits is not None or args.lossless_vectors:
        record["quantization"].update(dense_weight_bits=args.dense_weight_bits, lossless_vectors=args.lossless_vectors)
        write_json(record_path, record)
    write_json(args.out_dir / "completed.json", {"status": "completed", "selected": {args.method: {"file": record_path.name, "sha256": sha256_file(record_path)}}})
    print("ARCHIVE " + json.dumps({k: record[k] for k in ("byte_compression", "validation_ppl_change_percent", "passed_gate")}), flush=True)


if __name__ == "__main__":
    main()
