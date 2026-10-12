"""Write and validate the task-calibrated residual gain in the physical codebook."""
from pathlib import Path
import argparse
import io
import json
import math
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from experiments.lib.residual_runtime import evaluate_lm, batch_hash, sha256_file, write_json
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from run_residual_sweep_small import REGIMES, load_regime
    from run_system_bytes_comparison import excp_dequant
    cal = json.loads(args.calibration.read_text())
    chosen = cal["selected"]
    if chosen is None:
        raise RuntimeError("no gain satisfies the calibration quality gate")
    source_record = Path(cal["source_record"])
    if sha256_file(source_record) != cal["source_record_sha256"]:
        raise RuntimeError("source record differs")
    source = json.loads(source_record.read_text())
    source_archive = source_record.parent / source["archive"]["file"]
    if sha256_file(source_archive) != cal["identity"]["source_archive_sha256"]:
        raise RuntimeError("source archive differs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gain = chosen["gain"]
    method = cal["method"]
    payload = torch.load(io.BytesIO(zlib.decompress(source_archive.read_bytes())), map_location="cpu", weights_only=True)
    for encoded in payload["model"].values():
        key = "raw" if "raw" in encoded else "codebook"
        encoded[key] = encoded[key].float() * gain
    payload["gain_calibration"] = {"gain": gain, "sha256": sha256_file(args.calibration)}
    stream = io.BytesIO()
    torch.save(payload, stream)
    archive = args.out_dir / f"{method}_gain{gain:g}.zlib"
    archive.write_bytes(zlib.compress(stream.getvalue(), 6))
    del payload, stream
    payload = torch.load(io.BytesIO(zlib.decompress(archive.read_bytes())), map_location="cpu", weights_only=True)
    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    ctx = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    s = ctx["sweep"]
    plan = partition_repeated_seed_batches(ctx["pool"], 100, 1, 16, 42)
    offset = len(plan.training_segments[0])
    used = set(plan.selected_pool_indices[offset:offset + 16])
    gate = [ctx["pool"][i] for i in range(len(ctx["pool"])) if i not in used][:64]
    if batch_hash(gate) != cal["identity"]["gate_sha256"]:
        raise RuntimeError("gain calibration gate differs")
    baseline_gate = evaluate_lm(s.model, gate, s.device)
    from transformers import GPTNeoXForCausalLM
    base_model = GPTNeoXForCausalLM.from_pretrained(args.data_root / "models/pythia-410m", dtype=torch.float32, local_files_only=True)
    base = dict(base_model.named_parameters())
    params = dict(s.model.named_parameters())
    with torch.no_grad():
        for n, value in payload["model"].items():
            params[n].copy_(base[n].to(s.device) + excp_dequant(value, base[n].shape, s.device).float())
    s.model.load_state_dict({**s.model.state_dict(), **payload["buffers"]}, strict=True)
    actual_gate = evaluate_lm(s.model, gate, s.device)
    if abs(actual_gate["loss"] - chosen["gate"]["loss"]) > 1e-6:
        raise RuntimeError("written codebooks differ from gain calibration")
    validation = evaluate_lm(s.model, ctx["eval_batches"], s.device)
    delta = actual_gate["loss"] - baseline_gate["loss"]
    record = {**source, "identity": {"checkpoint_sha256": sha256_file(args.checkpoint),
              "gain_calibration_sha256": sha256_file(args.calibration)}, "status": "completed", "method": method,
              "gain": gain, "gate": actual_gate, "passed_gate": delta <= math.log1p(.01),
              "gate_ppl_change_percent": 100*math.expm1(delta), "validation": validation,
              "validation_ppl_change_percent": 100*math.expm1(validation["loss"]-ctx["uncompressed"]["val"]),
              "archive": {"file": archive.name, "bytes": archive.stat().st_size, "sha256": sha256_file(archive),
                          "original_checkpoint_bytes": args.checkpoint.stat().st_size},
              "byte_compression": args.checkpoint.stat().st_size/archive.stat().st_size,
              "quantization": {**source["quantization"], "codebook_dtype": "float32", "gain_applied_to_codes": gain},
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    path = args.out_dir / f"{method}_gain{gain:g}.json"
    write_json(path, record)
    write_json(args.out_dir / "completed.json", {"status": "completed", "selected": {method: {"file": path.name, "sha256": sha256_file(path)}}})
    print("ARCHIVE " + json.dumps({k:record[k] for k in ["gain","byte_compression","validation_ppl_change_percent"]}), flush=True)


if __name__ == "__main__":
    main()
