"""Reuse encoded weights and retain every Adam moment in BF16.

Weight quality and task accuracy can be reused exactly: weight codes and
codebooks are copied unchanged. Measure physical archives and full tensor
decode, then use the recovery runner to check actual optimizer continuation.
"""
from __future__ import annotations

import argparse
import io
import json
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
    ap.add_argument("--encoded-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from experiments.lib.residual_runtime import sha256_file, write_json
    from run_system_bytes_comparison import excp_dequant
    previous = json.loads((args.encoded_dir / "completed.json").read_text())
    identity = {**previous["identity"], "moment_storage": "all Adam entries retained in bfloat16",
                "transform_source_sha256": sha256_file(Path(__file__))}
    if sha256_file(args.checkpoint) != identity["checkpoint_sha256"]:
        raise RuntimeError("original optimizer checkpoint differs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final = args.out_dir / "completed.json"
    if final.exists():
        old = json.loads(final.read_text())
        if old["identity"] != identity or old["status"] != "completed":
            raise RuntimeError("retained-moment experiment identity conflict")
        print("SKIP " + str(final), flush=True)
        return
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = original["optimizer_state_dict"]["state"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    records, selected = [], {}
    for method, item in previous["selected"].items():
        source_path = args.encoded_dir / item["file"]
        if sha256_file(source_path) != item["sha256"]:
            raise RuntimeError("source archive record changed")
        source = json.loads(source_path.read_text())
        source_archive = args.encoded_dir / source["archive"]["file"]
        if sha256_file(source_archive) != source["archive"]["sha256"]:
            raise RuntimeError("source archive hash differs")
        path = args.out_dir / f"{method}_p0.7_bf16mom.json"
        archive = path.with_suffix(".zlib")
        if path.exists():
            record = json.loads(path.read_text())
            if record["identity"] != identity or record["status"] != "completed" or sha256_file(archive) != record["archive"]["sha256"]:
                raise RuntimeError("retained-moment result identity conflict")
            print("SKIP " + str(path), flush=True)
        else:
            payload = torch.load(io.BytesIO(zlib.decompress(source_archive.read_bytes())), map_location="cpu", weights_only=True)
            began = time.perf_counter()
            payload["codec"] = "retained_moments_bf16"
            for i, name in enumerate(payload["parameter_names"]):
                for key in ("exp_avg", "exp_avg_sq"):
                    payload[key][name] = {"raw": state[i][key].to(torch.bfloat16)}
            payload["identity"] = identity
            stream = io.BytesIO()
            torch.save(payload, stream)
            serialized = stream.tell()
            encoded = zlib.compress(stream.getvalue(), 6)
            del stream, payload
            temporary = archive.with_suffix(".tmp")
            temporary.write_bytes(encoded)
            temporary.replace(archive)
            del encoded
            encode_s = time.perf_counter() - began
            began = time.perf_counter()
            decoded = torch.load(io.BytesIO(zlib.decompress(archive.read_bytes())), map_location="cpu", weights_only=True)
            for i, name in enumerate(decoded["parameter_names"]):
                shape = original["model_state_dict"][name].shape
                residual = excp_dequant(decoded["model"][name], shape, "cpu").float()
                if not bool(residual.isfinite().all()):
                    raise RuntimeError("invalid decoded residual")
                for key in ("exp_avg", "exp_avg_sq"):
                    value = excp_dequant(decoded[key][name], shape, "cpu").float()
                    if not torch.equal(value, state[i][key].to(torch.bfloat16).float()):
                        raise RuntimeError("decoded Adam entry differs from its retained BF16 value")
                del residual, value
            decode_s = time.perf_counter() - began
            record = {**source, "identity": identity, "code_commit": commit,
                      "codec": "retained_moments_bf16", "quantization": {**source["quantization"], "moment_bits": 16, "joint_moment_beta": None},
                      "weight_quality_reused": {"file": str(source_path), "sha256": item["sha256"],
                                                "verification": "weight payload copied without modification"},
                      "weight_payload_reused_from_archive_sha256": source["archive"]["sha256"],
                      "archive": {"file": archive.name, "bytes": archive.stat().st_size, "sha256": sha256_file(archive),
                                  "torch_serialized_bytes": serialized, "original_checkpoint_bytes": args.checkpoint.stat().st_size,
                                  "byte_compression": args.checkpoint.stat().st_size / archive.stat().st_size},
                      "timing": {"selection_s": source["timing"]["selection_s"], "selection_reused": True,
                                 "encode_and_write_s": encode_s, "read_and_decode_s": decode_s,
                                 "scope": "CPU moment conversion/serialization/write and full CPU tensor read/decode; encoded weights reused"}}
            del decoded
            write_json(path, record)
            print("RESULT " + str(path) + f" bytes={record['archive']['bytes']}", flush=True)
        entry = {"file": path.name, "sha256": sha256_file(path)}
        records.append(entry)
        selected[method] = entry
    write_json(final, {"status": "completed", "identity": identity, "code_commit": commit,
                      "records": records, "selected": selected})
    print("COMPLETED " + str(final), flush=True)


if __name__ == "__main__":
    main()
