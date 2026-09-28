"""Longer 32B fine-tuning trajectory with residual scoring at several checkpoints.

Trains the layer-pipelined model for ``--steps`` steps on WikiText-2 or Alpaca
text and, every ``--every`` steps, evaluates immediate restoration of the
checkpoint residual against the previous checkpoint (and, at the last
checkpoint, against the base weights) for all scorers of
``run_probe_sweep_32b.run_regime``.  Checkpoints are kept in host memory only;
the optimizer state is moved to the host while scoring and restored afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.lib.qwen_pipeline import train_step  # noqa: E402
from experiments.scripts import run_probe_sweep_32b as sweep  # noqa: E402
from experiments.scripts.run_probe_coverage_32b import Context, now, write_json_new  # noqa: E402


def alpaca_ids(tok, path: Path) -> torch.Tensor:
    rows = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for r in rows:
        inp = f"\n\n### Input:\n{r['input']}" if r.get("input") else ""
        parts.append(f"### Instruction:\n{r['instruction']}{inp}\n\n### Response:\n{r['output']}")
    return torch.tensor(tok.encode("\n\n".join(parts)), dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "experiments/configs/probe_coverage_32b.json"))
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--dataset", choices=["wikitext", "alpaca"], default="alpaca")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--every", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--sparsities", default="0.5,0.8,0.9,0.95,0.98,0.99")
    ap.add_argument("--taylor-sparsities", default="0.9,0.98")
    ap.add_argument("--override", default="{}", help="JSON merged into the config (smoke tests)")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    for key, value in json.loads(args.override).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    cfg["train"]["lr"] = args.lr
    cfg["sparsities"] = [float(x) for x in args.sparsities.split(",")]
    cfg["weibull_cap"] = 0.999
    sweep.SPARSITIES = cfg["sparsities"]
    sweep.TAYLOR_SPARSITIES = [float(x) for x in args.taylor_sparsities.split(",") if x]
    sweep.SAMPLE_MS = (1,)
    sweep.TOPK_MS = (1, 4)
    sweep.HEADTAIL_MS = (2,)
    work = Path(args.work_dir)
    ctx = Context(cfg, work)
    if args.dataset == "alpaca":
        from transformers import AutoTokenizer

        cache = work / "tokens_alpaca.pt"
        if cache.exists():
            ids = torch.load(cache, weights_only=True)
        else:
            ids = alpaca_ids(AutoTokenizer.from_pretrained(cfg["model_path"]), Path.home() / "models/alpaca_data.json")
            torch.save(ids, cache)
        L = cfg["seq_len"]
        chunks = ids[: (ids.numel() // L) * L].view(-1, L)
        n = chunks.shape[0]
        # Fixed split: last 32 chunks validation, 8 before them gate, 4 before them scoring.
        ctx.val_ids = chunks[n - 32 :]
        ctx.gate_ids = chunks[n - 40 : n - 32]
        ctx.score_ids = chunks[n - 44 : n - 40]
        ctx.train_chunks = chunks[: n - 44]
        ctx.data_meta["alpaca"] = {"chunks": int(n), "train_chunks": int(n - 44), "val": [n - 32, n], "gate": [n - 40, n - 32], "score": [n - 44, n - 40]}
    if args.steps > ctx.train_chunks.shape[0]:
        raise SystemExit("not enough training chunks")
    ctx.log("trajectory_start", dataset=args.dataset, steps=args.steps, every=args.every, lr=args.lr, data=ctx.data_meta.get("alpaca"))
    names = ctx.pipe.eligible_names
    base = ctx.snapshot(names)
    opt = ctx.new_optimizer()
    opt.init_state()
    out = work / "trajectory"
    out.mkdir(exist_ok=True)
    record = {"dataset": args.dataset, "lr": args.lr, "steps": args.steps, "every": args.every, "train_log": [], "evals": {}, "checkpoints": {}}
    record["evals"]["0"] = ctx.evaluate()
    prev = base
    t0 = now()
    for step in range(1, args.steps + 1):
        rec = train_step(ctx.pipe, opt, ctx.train_chunks[step - 1 : step])
        record["train_log"].append(rec)
        if step % 20 == 0:
            ctx.log("train", **rec, pace_s=round((now() - t0) / step, 3))
        if step % args.every == 0:
            record["evals"][str(step)] = ctx.evaluate()
            theta = ctx.snapshot(ctx.pipe.trainable_names)
            opt.state_to(False)
            torch.cuda.empty_cache()
            regimes = [(f"step{step}_vs_step{step - args.every}", prev)]
            if step == args.steps:
                regimes.append((f"step{step}_vs_base", base))
            for tag, ref in regimes:
                path = out / f"{tag}.json"
                ts = time.time()
                res = sweep.run_regime(ctx, tag, {n: ref[n] for n in names}, theta, out / f"{tag}_work")
                res["seconds"] = round(time.time() - ts, 1)
                res["sparsities"] = cfg["sparsities"]
                res["provenance"] = ctx.prov
                res["gpu"] = ctx.gpu
                res["dataset"] = args.dataset
                res["step"] = step
                write_json_new(path, res)
                ctx.log("regime_done", tag=tag, seconds=res["seconds"])
            ctx.load_params(theta)
            prev = {n: theta[n] for n in names}
            del theta
            opt.state_to(True)
    record["train_seconds_total"] = round(now() - t0, 1)
    write_json_new(out / f"trajectory_{args.dataset}.json", record)


if __name__ == "__main__":
    main()
