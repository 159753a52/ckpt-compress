"""Fit 96 decoder gains to checkpoint logits; select only on calibration KL."""
from pathlib import Path
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def self_check():
    import torch
    base = torch.tensor([[.2, -.4], [.1, .6]], dtype=torch.float64)
    delta = torch.tensor([[.3, .2], [-.2, .1]], dtype=torch.float64)
    x = torch.tensor([[.7, -.8]], dtype=torch.float64)
    gain = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)
    direct = ((x @ (base + gain * delta).T) ** 2).sum()
    expected, = torch.autograd.grad(direct, gain)
    weight = (base + gain.detach() * delta).requires_grad_()
    ((x @ weight.T) ** 2).sum().backward()
    actual = (weight.grad * delta).sum()
    assert torch.allclose(actual, expected, atol=1e-12)
    eps = 1e-5
    values = [((x @ (base + (gain.detach() + sign * eps) * delta).T) ** 2).sum() for sign in (1, -1)]
    assert torch.allclose(actual, (values[0] - values[1]) / (2 * eps), atol=1e-9)
    print("Matrix-gain chain rule agrees with autograd and finite differences")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--source-record", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from transformers import GPTNeoXForCausalLM
    from experiments.lib.calibrated_archive import load_payload
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, set_seed, sha256_file, write_json
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from run_residual_sweep_small import REGIMES, load_regime
    from run_system_bytes_comparison import excp_dequant

    source = json.loads(args.source_record.read_text())
    archive = args.source_record.parent / source["archive"]["file"]
    source_manifest = json.loads(archive.read_text())
    assert source_manifest["format"] == "checkpoint-components-v1"
    assert "parameter_gains" not in source_manifest
    assert sha256_file(archive) == source["archive"]["sha256"]
    assert sha256_file(args.checkpoint) == source["identity"]["checkpoint_sha256"]
    if (args.out_dir / "completed.json").exists():
        raise RuntimeError("completed calibration exists; use its recorded result")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    ctx = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    s, pool = ctx["sweep"], ctx["pool"]
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, 42)
    offset = len(plan.training_segments[0])
    used = set(plan.selected_pool_indices[offset:offset + 16])
    gate_indices = [i for i in range(len(pool)) if i not in used][:64]
    fit_indices = [i for i in range(len(pool)) if i not in used | set(gate_indices)][:32]
    assert len(fit_indices) == 32 and not set(fit_indices).intersection(gate_indices)
    gate, fit = [pool[i] for i in gate_indices], [pool[i] for i in fit_indices]
    baseline_gate = evaluate_lm(s.model, gate, s.device)
    identity = {"checkpoint_sha256": source["identity"]["checkpoint_sha256"],
                "source_record_sha256": sha256_file(args.source_record), "source_archive_sha256": sha256_file(archive),
                "script_sha256": sha256_file(Path(__file__)), "seed": 42, "fit_sha256": batch_hash(fit),
                "gate_sha256": batch_hash(gate), "fit_indices": fit_indices, "gate_indices": gate_indices,
                "steps": 32, "learning_rate": .02, "gain_bounds": [.5, 1.5], "ppl_gate_percent": .5}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(args.out_dir / "run-manifest.json", {"identity": identity, "code_commit": commit,
               "method": source["method"], "source_record": str(args.source_record), "baseline_gate": baseline_gate,
               "selection": "minimum gate KL among configurations with gate PPL change <= 0.5%; earliest step on ties"})

    def log_probs(batch):
        values = {k: v.to(s.device) for k, v in batch.items()}
        mask = values["labels"][:, 1:] != -100
        logits = s.model(input_ids=values["input_ids"], attention_mask=values.get("attention_mask"), use_cache=False).logits
        return F.log_softmax(logits[:, :-1][mask].float(), dim=-1), values["labels"][:, 1:][mask]

    s.model.eval()
    with torch.no_grad():
        teacher_fit = [log_probs(b)[0].cpu() for b in fit]
        teacher_gate = [log_probs(b)[0].cpu() for b in gate]
    base_model = GPTNeoXForCausalLM.from_pretrained(args.data_root / "models/pythia-410m", dtype=torch.float32, local_files_only=True)
    base = {n: p.detach().to(s.device) for n, p in base_model.named_parameters()}
    # Decode the unscaled codes: the learned gains are absolute, applied exactly once.
    unscaled = dict(source_manifest)
    initial_gain = unscaled.pop("gain", 1.)
    targets = unscaled.pop("gain_parameter_names", s.names)
    assert set(targets) == set(s.names)
    component_bytes = 0
    for key in ("state", "weights"):
        item = unscaled[key]
        original = (archive.parent / item["file"]).resolve()
        assert original.parent == archive.parent.resolve() and sha256_file(original) == item["sha256"]
        dest = args.out_dir / item["file"]
        if not dest.exists():
            if original.stat().st_dev == args.out_dir.stat().st_dev:
                os.link(original, dest)
            else:
                shutil.copyfile(original, dest)
        assert sha256_file(dest) == item["sha256"]
        component_bytes += dest.stat().st_size
    raw_path = args.out_dir / "unscaled.manifest.json"
    write_json(raw_path, unscaled)
    payload = load_payload(raw_path)
    residual = {n: excp_dequant(v, base[n].shape, s.device).float() for n, v in payload["model"].items()}
    params = dict(s.model.named_parameters())
    s.model.load_state_dict({**s.model.state_dict(), **payload["buffers"]}, strict=True)
    for n, p in params.items():
        p.requires_grad_(n in s.names)
    del payload, base_model
    gains = torch.nn.Parameter(torch.full((len(s.names),), initial_gain, device=s.device))
    opt = torch.optim.Adam([gains], lr=.02)

    def apply():
        lookup = dict(zip(s.names, gains.detach()))
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(base[n] + lookup.get(n, 1.) * residual[n])

    def measure():
        kl, ce, tokens = 0., 0., 0
        with torch.no_grad():
            for b, target in zip(gate, teacher_gate):
                logp, labels = log_probs(b)
                kl += F.kl_div(logp, target.to(s.device), reduction="sum", log_target=True).item()
                ce += F.nll_loss(logp, labels, reduction="sum").item()
                tokens += labels.numel()
        return {"kl": kl / tokens, "loss": ce / tokens, "tokens": tokens,
                "ppl_change_percent": 100 * math.expm1(ce / tokens - baseline_gate["loss"])}

    set_seed(42)
    began = time.perf_counter()
    points, best = [], None
    for step in range(33):
        fit_kl = None
        if step:
            b, target = fit[step - 1], teacher_fit[step - 1]
            s.model.zero_grad(set_to_none=True)
            opt.zero_grad(set_to_none=True)
            logp, labels = log_probs(b)
            loss = F.kl_div(logp, target.to(s.device), reduction="sum", log_target=True) / labels.numel()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite reconstruction KL")
            loss.backward()
            # Exact chain rule for W = W_base + gain * decoded_residual.
            gains.grad = torch.stack([(params[n].grad * residual[n]).sum() for n in s.names])
            if not torch.isfinite(gains.grad).all():
                raise RuntimeError("nonfinite gain gradient")
            opt.step()
            with torch.no_grad():
                gains.clamp_(.5, 1.5)
            fit_kl = loss.item()
            del loss, logp
        apply()
        point = {"step": step, "gate": measure(), "fit_kl": fit_kl,
                 "gains": {n: float(g) for n, g in zip(s.names, gains.detach().cpu())}}
        if step == 0:
            assert abs(point["gate"]["loss"] - source["gate"]["loss"]) <= 1e-6
        points.append(point)
        if point["gate"]["ppl_change_percent"] <= .5 and (best is None or point["gate"]["kl"] < best["gate"]["kl"]):
            best = point
        write_json(args.out_dir / "trace.json", {"identity": identity, "points": points, "best_step": best["step"] if best else None})
        print(f"TRACE {source['method']} step={step} KL={point['gate']['kl']:.8g} PPL={point['gate']['ppl_change_percent']:.5f}", flush=True)
    if best is None:
        raise RuntimeError("no feasible configuration, including the original")
    manifest = {**unscaled, "parameter_gains": best["gains"]}
    manifest_path = args.out_dir / "matrix_gains.manifest.json"
    write_json(manifest_path, manifest)
    with torch.no_grad():
        gains.copy_(torch.tensor([best["gains"][n] for n in s.names], device=s.device))
    apply()
    expected_gate = evaluate_lm(s.model, gate, s.device)
    payload = load_payload(manifest_path)
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(base[n] + excp_dequant(payload["model"][n], p.shape, s.device).float())
    del payload
    actual_gate = evaluate_lm(s.model, gate, s.device)
    assert abs(actual_gate["loss"] - expected_gate["loss"]) <= 1e-6
    assert abs(actual_gate["loss"] - best["gate"]["loss"]) <= 1e-6
    validation = evaluate_lm(s.model, ctx["eval_batches"], s.device)
    size = component_bytes + manifest_path.stat().st_size
    record = {**source, "code_commit": commit, "gate": actual_gate, "passed_gate": True,
              "gate_ppl_change_percent": 100 * math.expm1(actual_gate["loss"] - baseline_gate["loss"]),
              "validation": validation, "validation_ppl_change_percent": 100 * math.expm1(validation["loss"] - ctx["uncompressed"]["val"]),
              "archive": {"file": manifest_path.name, "bytes": size, "sha256": sha256_file(manifest_path)},
              "byte_compression": args.checkpoint.stat().st_size / size,
              "matrix_gain_calibration": {"identity": identity, "selected_step": best["step"], "selected_gate_kl": best["gate"]["kl"],
                                          "elapsed_s": time.perf_counter() - began, "source_record_sha256": sha256_file(args.source_record)}}
    # The source scalar gain has been replaced by the explicit per-matrix map.
    record.pop("gain", None)
    write_json(args.out_dir / "archive.json", record)
    write_json(args.out_dir / "completed.json", {"status": "completed", "identity": identity,
               "selected": {source["method"]: {"file": "archive.json", "sha256": sha256_file(args.out_dir / "archive.json")}}})
    print("SELECTED", source["method"], best["step"], record["validation_ppl_change_percent"], record["byte_compression"], flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
    else:
        main()
