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
    args = ap.parse_args()
    import torch
    from run_residual_sweep_small import REGIMES, load_regime
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, sha256_file, write_json
    from run_system_bytes_comparison import excp_quant, excp_dequant
    chosen = json.loads(args.selected.read_text())
    old = json.loads(args.legacy_archive.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.out_dir / "iterative_ucb_disjoint_p0.7_bf16mom.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
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
    if sha256_file(mask_path) != chosen["mask_sha256"]:
        raise RuntimeError("selected mask hash differs")
    keep = torch.load(mask_path, map_location="cpu", weights_only=True)
    payload = torch.load(io.BytesIO(zlib.decompress(source.read_bytes())), map_location="cpu", weights_only=True)
    started = time.perf_counter()
    for index, name in enumerate(payload["parameter_names"]):
        if name in keep:
            residual = s.delta[name].to("cuda:0") * keep[name].to("cuda:0")
            payload["model"][name] = excp_quant(residual, 42 * 100003 + 3 * index, 4)
    payload["identity"] = {"checkpoint_sha256": sha256_file(args.checkpoint), "selection_sha256": sha256_file(args.selected)}
    stream = io.BytesIO()
    torch.save(payload, stream)
    archive = args.out_dir / "iterative_ucb_disjoint_p0.7_bf16mom.zlib"
    archive.write_bytes(zlib.compress(stream.getvalue(), 6))
    encode_seconds = time.perf_counter() - started
    # Reopen the saved bytes; quality measurements use decoded weights.
    del payload, stream
    payload = torch.load(io.BytesIO(zlib.decompress(archive.read_bytes())), map_location="cpu", weights_only=True)
    from transformers import GPTNeoXForCausalLM
    base_model = GPTNeoXForCausalLM.from_pretrained(args.data_root / "models/pythia-410m", dtype=torch.float32, local_files_only=True)
    base = dict(base_model.named_parameters())
    with torch.no_grad():
        for name, encoded in payload["model"].items():
            s.params[name].copy_(base[name].to("cuda:0") + excp_dequant(encoded, base[name].shape, "cuda:0").float())
    s.model.load_state_dict({**s.model.state_dict(), **payload["buffers"]}, strict=True)
    calibration = evaluate_lm(s.model, gate, "cuda:0")
    validation = evaluate_lm(s.model, ctx["eval_batches"], "cuda:0")
    old_gate = json.loads((ROOT / "results/v100_quality_calibration_20261011/staged_K4_B4_p0.7_seed42.json").read_text())["baseline_gate"]
    delta = calibration["loss"] - old_gate["loss"]
    record = {"status": "completed", "identity": payload["identity"], "method": "iterative_ucb_disjoint", "sparsity": .7, "seed": 42,
              "selection_sha256": sha256_file(args.selected), "config": chosen["config"], "backward_budget": chosen["backward_budget"],
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "archive": {"file": archive.name, "bytes": archive.stat().st_size, "sha256": sha256_file(archive)},
              "byte_compression": args.checkpoint.stat().st_size / archive.stat().st_size,
              "gate": calibration, "gate_ppl_change_percent": 100 * math.expm1(delta), "passed_gate": delta <= math.log1p(.01),
              "validation": validation, "validation_ppl_change_percent": 100 * math.expm1(validation["loss"] - ctx["uncompressed"]["val"]),
              "quantization": {"weight_bits": 4, "moment_bits": 16, "moment_dtype": "bfloat16", "all_moments_retained": True},
              "optimizer_payload_reused": {"source": str(args.legacy_archive), "sha256": old["archive"]["sha256"]},
              "encode_seconds": encode_seconds, "baseline_validation": ctx["uncompressed"]}
    write_json(record_path, record)
    write_json(args.out_dir / "completed.json", {"status": "completed", "selected": {"iterative_ucb_disjoint": {"file": record_path.name, "sha256": sha256_file(record_path)}}})
    print("ARCHIVE " + json.dumps({k: record[k] for k in ("byte_compression", "validation_ppl_change_percent", "passed_gate")}), flush=True)


if __name__ == "__main__":
    main()
