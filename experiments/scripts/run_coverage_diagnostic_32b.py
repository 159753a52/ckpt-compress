"""Exact coverage curves of the model's next-token distributions (forward only).

For every predicted position with distribution p, the expected mass covered
after m independent draws with replacement is sum_y p_y (1 - (1 - p_y)^m); the
mass covered by the m most probable tokens is the sum of the top-m
probabilities.  No sampling is involved, so these are exact expectations.

States: base weights, the saved reference (step 100) and checkpoint (step 200).
Data: WikiText-2 scoring chunks, WikiText-2 validation chunks, Alpaca text.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.lib.qwen_pipeline import PipelinedQwen  # noqa: E402

MS = [1, 2, 3, 4, 5, 8, 16, 32, 64]


def alpaca_text(path: Path, limit: int) -> str:
    rows = json.loads(path.read_text(encoding="utf-8"))[:limit]
    parts = []
    for r in rows:
        inp = f"\n\n### Input:\n{r['input']}" if r.get("input") else ""
        parts.append(f"### Instruction:\n{r['instruction']}{inp}\n\n### Response:\n{r['output']}")
    return "\n\n".join(parts)


@torch.no_grad()
def stats(pipe: PipelinedQwen, chunks: torch.Tensor) -> dict:
    acc = {"positions": 0, "nll": 0.0, "entropy": 0.0, "top1": 0.0, "top1_ge_0.9": 0, "top1_lt_0.5": 0}
    rep = {m: 0.0 for m in MS}
    top = {m: 0.0 for m in MS}
    rep_q = {m: [] for m in (1, 8, 64)}
    need_rep = {0.9: [], 0.95: []}
    for i in range(chunks.shape[0]):
        ids = chunks[i : i + 1]
        logits = pipe.forward(ids)
        flat, target = pipe.shifted(logits, ids)
        logp = torch.log_softmax(flat.float(), dim=-1)
        p = logp.exp()
        n = p.shape[0]
        acc["positions"] += n
        acc["nll"] += float(-logp.gather(1, target[:, None]).sum())
        acc["entropy"] += float(-(p * logp).sum())
        t1 = p.max(dim=1).values
        acc["top1"] += float(t1.sum())
        acc["top1_ge_0.9"] += int((t1 >= 0.9).sum())
        acc["top1_lt_0.5"] += int((t1 < 0.5).sum())
        log1m = torch.log1p(-p.clamp(max=1 - 1e-7))
        sorted_p = p.sort(dim=1, descending=True).values
        csum = sorted_p.cumsum(dim=1)
        for m in MS:
            q = (p * (1 - torch.exp(m * log1m))).sum(dim=1)
            rep[m] += float(q.sum())
            top[m] += float(csum[:, m - 1].sum())
            if m in rep_q:
                rep_q[m].append(q.cpu())
        # deterministic rounds needed for each position to reach tau
        for tau in need_rep:
            k = (csum < tau).sum(dim=1) + 1
            need_rep[tau].append(k.cpu())
        del logits, flat, logp, p, log1m, sorted_p, csum
    n = acc["positions"]
    out = {
        "positions": n,
        "mean_nll": acc["nll"] / n,
        "mean_entropy_nats": acc["entropy"] / n,
        "mean_top1": acc["top1"] / n,
        "frac_top1_ge_0.9": acc["top1_ge_0.9"] / n,
        "frac_top1_lt_0.5": acc["top1_lt_0.5"] / n,
        "expected_mean_coverage_with_replacement": {m: rep[m] / n for m in MS},
        "mean_coverage_topk": {m: top[m] / n for m in MS},
    }
    for m, qs in rep_q.items():
        q = torch.cat(qs)
        out[f"with_replacement_m{m}_quantiles"] = {
            str(a): float(torch.quantile(q, a)) for a in (0.05, 0.1, 0.5)
        }
    for tau, ks in need_rep.items():
        k = torch.cat(ks).float()
        out[f"topk_rounds_to_{tau}"] = {
            "median": float(k.median()),
            "p90": float(torch.quantile(k, 0.9)),
            "frac_le_5": float((k <= 5).float().mean()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "experiments/configs/probe_coverage_32b.json"))
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-count", type=int, default=16)
    ap.add_argument("--alpaca-count", type=int, default=16)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "5,6,7":
        raise SystemExit("CUDA_VISIBLE_DEVICES must be 5,6,7")
    cfg = json.loads(Path(args.config).read_text())
    work = Path(args.work_dir)
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model_path"])
    L = cfg["seq_len"]
    train = torch.load(work / "tokens_train.pt", weights_only=True)
    val = torch.load(work / "tokens_val.pt", weights_only=True)
    tr = train[: (train.numel() // L) * L].view(-1, L)
    va = val[: (val.numel() // L) * L].view(-1, L)
    c = cfg["chunks"]
    al_ids = torch.tensor(tok.encode(alpaca_text(Path.home() / "models/alpaca_data.json", 400)), dtype=torch.long)
    al = al_ids[: (al_ids.numel() // L) * L].view(-1, L)[: args.alpaca_count]
    data = {
        "wikitext_scoring": tr[c["score_start"] : c["score_start"] + c["score_count"]],
        "wikitext_val": va[: args.val_count],
        "alpaca": al,
    }
    model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.eval()
    pipe = PipelinedQwen(model, ["cuda:0", "cuda:1", "cuda:2"], cfg["layer_split"])
    meta = json.loads((work / "ckpt" / "meta.json").read_text())
    results = {"ms": MS, "states": {}, "data_chunks": {k: int(v.shape[0]) for k, v in data.items()}}
    for state in ("base", "step100", "step200"):
        if state != "base":
            name = "theta_ref" if state == "step100" else "theta"
            sd = load_file(meta["files"][name]["path"])
            with torch.no_grad():
                for n, v in sd.items():
                    pipe.param(n).copy_(v.to(pipe.param(n).device))
            del sd
        t0 = time.time()
        results["states"][state] = {k: stats(pipe, v) for k, v in data.items()}
        results["states"][state]["seconds"] = round(time.time() - t0, 1)
        print(json.dumps({state: {k: (v["mean_nll"], v["expected_mean_coverage_with_replacement"][8], v["mean_coverage_topk"][8]) for k, v in results["states"][state].items() if isinstance(v, dict)}}), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
