#!/usr/bin/env python
"""Long training runs with simulated failures and compressed residual recovery.

Two settings share this script.

``--mode finetune`` (study A): Pythia-410M is fine-tuned on the full Alpaca data
starting from the pretrained model, which is also the fixed reference of every
residual (the base-reference setting of the paper).

``--mode pretrain`` (study B): a GPT-NeoX model with the shape of Pythia-160M is
trained from scratch on FineWeb-Edu tokens; the reference is the run's own full
checkpoint written every ``--ref-every`` steps (step 0 is the initialization).

At every failure step the current weights are compressed: the residual to the
reference on the eligible matrices (the attention and MLP weight matrices of the
Transformer layers) is pruned at deletion rate P by one selection rule, the model
is restored as reference plus kept residual entries, and training continues.  The
AdamW moments and all other tensors are kept losslessly, so only the selection
rule differs between trajectories.  All trajectories of one seed share the
initialization, the data order, the scoring batches and the evaluation data.

Selection rules (``--method``):
  nocomp    no compression (the reference trajectory)
  excp      keep the residual entries of largest magnitude, one global threshold
  inshrink  keep the entries of largest |g_bar * theta| (8 scoring batches)
  dacp      DACP-G: K stages of the geometric schedule, every stage rescoring the
            same B batches at the partially restored model, score
            |mean_j(-g_j d)| + kappa * std_j(-g_j d) / sqrt(B), exact global
            threshold with ties broken by global index
  restart   delete the whole residual (resume from the reference itself)

The run is resumable: the full state is written atomically after every failure
and every ``--save-every`` steps, and restarting the same command continues from
it.  ``DONE`` marks a finished run; ``summary.json`` holds the results and
``metrics.jsonl`` every evaluation and failure record.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ELIGIBLE_SUFFIXES = (
    "attention.query_key_value.weight",
    "attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
)
F32_INF_BITS = 0x7F800000


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def parse_int_list(text: str):
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("finetune", "pretrain"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method", choices=("nocomp", "excp", "excp_official", "inshrink", "dacp", "restart", "random"),
                    required=True)
    ap.add_argument("--ref-mode", choices=("fixed", "chain"), default="fixed",
                    help="chain: every --ckpt-every steps the residual to the previous reconstructed checkpoint is "
                         "compressed (training continues from the uncompressed weights), and every --break-every "
                         "steps training resumes from the reconstructed checkpoint (the ExCP protocol)")
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--break-every", type=int, default=0)
    ap.add_argument("--match-official", action="store_true",
                    help="delete as many entries as the official ExCP threshold at every checkpoint")
    ap.add_argument("--excp-alpha", type=float, default=5e-5)
    ap.add_argument("--sparsity", type=float, default=0.0, help="deletion rate P")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--stages", type=int, default=16)
    ap.add_argument("--score-batches", type=int, default=16)
    ap.add_argument("--score-batch-size", type=int, default=2)
    ap.add_argument("--decision", type=Path, default=None,
                    help="JSON with 'sparsity' and 'kappa' that override the two options (pilot decision)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--total-steps", type=int, required=True)
    ap.add_argument("--stop-at", type=int, default=0, help="stop after this many steps (schedule still uses --total-steps)")
    ap.add_argument("--fail-steps", default="", help="comma-separated failure steps")
    ap.add_argument("--ref-every", type=int, default=0, help="pretrain: refresh the reference every N steps")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-after", default="10,25,50,100,200", help="extra evaluations after each failure")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--keep-states", default="", help="also keep a copy of the state at these steps")
    ap.add_argument("--init-state", type=Path, default=None, help="start from this saved state (pilot runs)")
    ap.add_argument("--ref-state", type=Path, default=None, help="pretrain: reference weights from this saved state")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--min-lr-ratio", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, required=True)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, required=True, help="sequences per micro-batch")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--eval-batches", type=int, default=0, help="pretrain: number of validation batches")
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--amp", choices=("bf16", "none"), default="bf16")
    ap.add_argument("--deterministic", action="store_true",
                    help="deterministic kernels, so trajectories agree bit for bit until their first restoration")
    ap.add_argument("--attn", choices=("sdpa", "eager"), default="sdpa")
    # data and model locations
    ap.add_argument("--model", default="EleutherAI/pythia-410m", help="finetune: pretrained model")
    ap.add_argument("--tokenizer", default="EleutherAI/pythia-410m")
    ap.add_argument("--alpaca-parquet", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--token-dir", type=Path, default=None, help="pretrain: directory with train.bin and val.bin")
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--ffn", type=int, default=3072)
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------- data


class AlpacaData:
    """Full Alpaca, formatted and labelled as experiments/lib/data.py does it.

    The train/validation split replicates ``random_split(..., generator=manual_seed(0))``
    with min(1000, n/10) validation samples.  Batches are trimmed to their longest
    sequence; with right padding and masked labels this leaves the loss unchanged."""

    def __init__(self, args, tokenizer):
        cache = args.cache_dir / f"alpaca_full_pythia_seq{args.seq}.pt"
        if cache.exists():
            blob = torch.load(cache)
        else:
            import pyarrow.parquet as pq

            table = pq.read_table(args.alpaca_parquet).to_pydict()
            texts = []
            for ins, inp, out in zip(table["instruction"], table["input"], table["output"]):
                prompt = f"### Instruction:\n{ins}\n"
                if inp:
                    prompt += f"### Input:\n{inp}\n"
                prompt += f"### Response:\n{out}"
                texts.append(prompt)
            ids, masks = [], []
            for i in range(0, len(texts), 4096):
                enc = tokenizer(texts[i:i + 4096], truncation=True, padding="max_length",
                                max_length=args.seq, return_tensors="pt")
                ids.append(enc["input_ids"].to(torch.int32))
                masks.append(enc["attention_mask"].to(torch.bool))
            blob = {"input_ids": torch.cat(ids), "attention_mask": torch.cat(masks)}
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".tmp{os.getpid()}")
            torch.save(blob, tmp)
            os.replace(tmp, cache)
        self.ids, self.mask = blob["input_ids"], blob["attention_mask"]
        n = self.ids.shape[0]
        n_val = min(1000, max(1, n // 10))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
        self.train_idx, self.val_idx = perm[: n - n_val], perm[n - n_val:]
        self.bs, self.seed = args.batch_size, args.seed
        self.accum = args.accum
        self._perms = {}
        self.n_samples = n

    def _epoch_perm(self, e):
        if e not in self._perms:
            g = torch.Generator().manual_seed(self.seed * 1000 + e)
            self._perms[e] = torch.randperm(len(self.train_idx), generator=g).tolist()
        return self._perms[e]

    def make(self, rows):
        ids, mask = self.ids[rows], self.mask[rows]
        length = max(2, int(mask.sum(1).max()))
        ids, mask = ids[:, :length].long(), mask[:, :length]
        return {"input_ids": ids, "attention_mask": mask.long(), "labels": ids.masked_fill(~mask, -100)}

    def train_micro(self, step, micro):
        n = len(self.train_idx)
        start = (step * self.accum + micro) * self.bs
        rows = []
        for j in range(start, start + self.bs):
            e, r = divmod(j, n)
            rows.append(self.train_idx[self._epoch_perm(e)[r]])
        return self.make(rows)

    def eval_batches(self, size):
        order = sorted(self.val_idx, key=lambda i: int(self.mask[i].sum()))
        return [self.make(order[i:i + size]) for i in range(0, len(order), size)]

    def score_batches(self, count, size):
        g = torch.Generator().manual_seed(7770 + self.seed)
        perm = torch.randperm(len(self.train_idx), generator=g).tolist()
        rows = [self.train_idx[i] for i in perm[: count * size]]
        return [self.make(rows[i:i + size]) for i in range(0, len(rows), size)]


class TokenData:
    """Pre-tokenized uint16 streams; training reads non-overlapping 2048-token chunks
    in a seeded order (no chunk is used twice), scoring uses chunks from the end of
    that order, which training never reaches, and validation reads val.bin."""

    def __init__(self, args):
        self.train = np.memmap(args.token_dir / "train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(args.token_dir / "val.bin", dtype=np.uint16, mode="r")
        self.seq, self.bs, self.accum = args.seq, args.batch_size, args.accum
        n_chunks = (len(self.train) - 1) // self.seq
        self.order = np.random.default_rng(args.seed).permutation(n_chunks)
        need = args.total_steps * self.accum * self.bs + args.score_batches * args.score_batch_size
        if need > n_chunks:
            raise SystemExit(f"train.bin holds {n_chunks} chunks, the run needs {need}")

    def _rows(self, arr, starts):
        x = np.stack([arr[s:s + self.seq] for s in starts]).astype(np.int64)
        ids = torch.from_numpy(x)
        return {"input_ids": ids, "labels": ids}

    def train_micro(self, step, micro):
        base = (step * self.accum + micro) * self.bs
        return self._rows(self.train, [int(c) * self.seq for c in self.order[base:base + self.bs]])

    def eval_batches(self, count, size):
        starts = [i * self.seq for i in range(count * size)]
        if starts[-1] + self.seq > len(self.val):
            raise SystemExit("val.bin is too short for the requested evaluation batches")
        return [self._rows(self.val, starts[i:i + size]) for i in range(0, len(starts), size)]

    def score_batches(self, count, size):
        chunks = self.order[len(self.order) - count * size:]
        rows = [int(c) * self.seq for c in chunks]
        return [self._rows(self.train, rows[i:i + size]) for i in range(0, len(rows), size)]


# ----------------------------------------------------------------------------- selection


def to_device(batch, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


def nonneg(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=float(torch.finfo(torch.float32).max))
    return torch.where(x > 0, x, torch.zeros_like(x))


@torch.no_grad()
def select_smallest(scores: dict, names, m: int) -> dict:
    """Masks of the m globally smallest scores (non-negative float32; +inf never chosen).

    Non-negative IEEE floats order like their int32 bit patterns, so the m-th smallest
    is found by bisection on the bit pattern, counting entries at or below the probe;
    ties at the threshold are taken in global index order (tensor order, then flat index)."""
    sel = {n: torch.zeros_like(scores[n], dtype=torch.bool) for n in names}
    if m <= 0:
        return sel
    bits = {n: scores[n].view(torch.int32) for n in names}

    def count_le(t):
        return sum(int((bits[n] <= t).sum()) for n in names)

    lo, hi = -1, F32_INF_BITS - 1
    if count_le(hi) < m:
        raise RuntimeError("fewer finite scores than deletions requested")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count_le(mid) >= m:
            hi = mid
        else:
            lo = mid
    need = m
    for n in names:
        sel[n] = bits[n] < hi
        need -= int(sel[n].sum())
    for n in names:
        if need <= 0:
            break
        eq = (bits[n] == hi).flatten()
        take = eq & (torch.cumsum(eq.to(torch.int64), 0) <= need)
        sel[n] |= take.view_as(sel[n])
        need -= int(take.sum())
    if need != 0:
        raise RuntimeError("tie resolution did not reach the requested count")
    return sel


def batch_grads(model, batches, dev, visit):
    """FP32 gradient of each scoring batch at the current weights, passed to visit(j, grads)."""
    model.eval()
    for j, b in enumerate(batches, 1):
        model.zero_grad(set_to_none=True)
        out = model(**to_device(b, dev))
        out.loss.backward()
        visit(j)
    model.zero_grad(set_to_none=True)


@torch.no_grad()
def official_excp_mask(d, params, names, opt, alpha):
    """Deletion mask of the official ExCP code (compress_pythia.py): per tensor, keep the
    residual entries with |d| > median(|d|) * min(2.5, alpha / sqrt(mean(exp_avg_sq)))."""
    out = {}
    for n in names:
        v = opt.state[params[n]].get("exp_avg_sq") if params[n] in opt.state else None
        vm = float(v.float().mean()) if v is not None else 0.0
        scale = min(2.5, alpha / math.sqrt(vm)) if vm > 0 else 2.5
        a = d[n].abs()
        out[n] = a <= a.median() * scale
    return out


def compress(model, params, names, ref, method, sparsity, kappa, stages, score_batches, dev,
             opt=None, alpha=5e-5, match_official=False):
    """Prune the residual to ``ref`` in place; returns the restore record.

    ``excp_official`` deletes what the official ExCP threshold deletes; with
    ``match_official`` the other rules delete exactly that many entries."""
    t0 = time.time()
    with torch.no_grad():
        d = {n: (params[n].detach() - ref[n]).float() for n in names}
    total = sum(d[n].numel() for n in names)
    official = None
    at_rate = method == "excp_official" and 0 < sparsity < 1 and not match_official
    if (method == "excp_official" and not at_rate) or match_official:
        official = official_excp_mask(d, params, names, opt, alpha)
        m = sum(int(official[n].sum()) for n in names)
    else:
        m = math.floor(sparsity * total) if method != "restart" else total
    sparsity = m / total
    deleted = {n: torch.zeros_like(d[n], dtype=torch.bool) for n in names}
    info = {"eligible": total, "target_deleted": m, "deletion_rate": sparsity}
    if method == "excp_official" and at_rate:
        # the official per-tensor threshold c * median(|d|), with c set so that exactly m entries go:
        # rank |d| / median(|d|) of its tensor under one global threshold
        with torch.no_grad():
            norm = {n: nonneg(d[n].abs() / d[n].abs().median().clamp_min(1e-30)) for n in names}
        deleted = select_smallest(norm, names, m)
        del norm
    elif method == "excp_official":
        deleted = official
    elif method == "excp":
        sel = select_smallest({n: nonneg(d[n].abs()) for n in names}, names, m)
        deleted = sel
    elif method == "inshrink":
        acc = {n: torch.zeros_like(d[n]) for n in names}

        def visit(j):
            for n in names:
                acc[n] += params[n].grad.float()

        batch_grads(model, score_batches[:8], dev, visit)
        with torch.no_grad():
            sc = {n: nonneg((acc[n] / min(8, len(score_batches)) * params[n].detach().float()).abs()) for n in names}
        deleted = select_smallest(sc, names, m)
        del acc, sc
    elif method == "dacp":
        b = len(score_batches)
        stage_log = []
        for k in range(1, stages + 1):
            target = math.floor((1 - (1 - sparsity) ** (k / stages)) * total) if k < stages else m
            done = sum(int(deleted[n].sum()) for n in names)
            with torch.no_grad():
                for n in names:
                    params[n].copy_(ref[n] + d[n] * (~deleted[n]))
            mean = {n: torch.zeros_like(d[n]) for n in names}
            m2 = {n: torch.zeros_like(d[n]) for n in names}

            def visit(j):
                with torch.no_grad():
                    for n in names:
                        c = -params[n].grad.float() * d[n]
                        delta = c - mean[n]
                        mean[n] += delta / j
                        m2[n] += delta * (c - mean[n])

            batch_grads(model, score_batches, dev, visit)
            with torch.no_grad():
                sc = {}
                for n in names:
                    s = mean[n].abs()
                    if kappa > 0 and b > 1:
                        s = s + kappa * torch.sqrt(torch.clamp(m2[n] / (b - 1), min=0)) / math.sqrt(b)
                    s = nonneg(s)
                    s[deleted[n]] = float("inf")
                    sc[n] = s
                del mean, m2
                sel = select_smallest(sc, names, target - done)
                for n in names:
                    deleted[n] |= sel[n]
                del sc, sel
            stage_log.append({"stage": k, "cumulative_deleted": target})
        info["stages"] = stages
        info["score_batches"] = b
        info["kappa"] = kappa
    elif method == "random":
        # uniformly random deletions (noise control: with a tiny rate the restoration barely changes the
        # model, so the later divergence from the uncompressed run measures the sensitivity of training)
        seed = int(float(d[names[0]].double().abs().sum()) * 1e6) % (2 ** 31 - 1)  # reproducible per state
        g = torch.Generator(device=d[names[0]].device).manual_seed(seed)
        sc = {n: torch.rand(d[n].shape, generator=g, device=d[n].device) for n in names}
        deleted = select_smallest(sc, names, m)
        del sc
    elif method == "restart":
        deleted = {n: torch.ones_like(d[n], dtype=torch.bool) for n in names}
    else:
        raise ValueError(method)
    with torch.no_grad():
        kept_sq, tot_sq, n_del = 0.0, 0.0, 0
        for n in names:
            params[n].copy_(ref[n] + d[n] * (~deleted[n]))
            kept_sq += float((d[n] * (~deleted[n])).double().square().sum())
            tot_sq += float(d[n].double().square().sum())
            n_del += int(deleted[n].sum())
    if n_del != m:
        raise RuntimeError(f"deleted {n_del} entries, expected {m}")
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    info.update({"deleted": n_del, "residual_l2": math.sqrt(tot_sq),
                 "kept_residual_l2_fraction": math.sqrt(kept_sq / tot_sq) if tot_sq > 0 else None,
                 "compress_s": round(time.time() - t0, 2)})
    return info


# ----------------------------------------------------------------------------- run


def build_model(args, dev):
    from transformers import AutoModelForCausalLM, GPTNeoXConfig, GPTNeoXForCausalLM

    if args.mode == "finetune":
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                     attn_implementation=args.attn)
    else:
        cfg = GPTNeoXConfig(vocab_size=50304, hidden_size=args.hidden, num_hidden_layers=args.layers,
                            num_attention_heads=args.heads, intermediate_size=args.ffn,
                            max_position_embeddings=2048, rotary_pct=0.25, use_parallel_residual=True,
                            tie_word_embeddings=False, bos_token_id=0, eos_token_id=0)
        cfg._attn_implementation = args.attn
        torch.manual_seed(args.seed)
        model = GPTNeoXForCausalLM(cfg)
    return model.to(dev)


def build_optimizer(args, model):
    if args.mode == "finetune":
        groups = [{"params": [p for p in model.parameters()], "weight_decay": args.weight_decay}]
    else:
        decay = [p for p in model.parameters() if p.ndim >= 2]
        no_decay = [p for p in model.parameters() if p.ndim < 2]
        groups = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, args.beta2), eps=1e-8, fused=True)


def lr_at(args, step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = min(1.0, (step - args.warmup) / max(1, args.total_steps - args.warmup))
    floor = args.min_lr_ratio
    return args.lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, batches, dev):
    model.eval()
    loss_sum, tokens = 0.0, 0
    for b in batches:
        b = to_device(b, dev)
        out = model(**b)
        n = int((b["labels"][:, 1:] != -100).sum())
        loss_sum += float(out.loss) * n
        tokens += n
    del out, b
    torch.cuda.empty_cache()
    return loss_sum / tokens, tokens


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def main(argv=None):
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "DONE").exists():
        log(f"{args.out} is already done")
        return
    if args.decision is not None:
        dec = json.loads(args.decision.read_text())
        args.sparsity, args.kappa = float(dec["sparsity"]), float(dec.get("kappa", args.kappa))
    needs_rate = args.method in ("excp", "inshrink", "dacp", "random") and not args.match_official
    if args.method == "excp_official" and args.sparsity:
        needs_rate = True
    if needs_rate and not 0 < args.sparsity < 1:
        raise SystemExit("compressing methods need 0 < sparsity < 1")
    stop = args.stop_at or args.total_steps
    chain = args.ref_mode == "chain"
    fail_steps = sorted(set(parse_int_list(args.fail_steps)))
    ckpt_steps, break_steps = set(), set()
    if chain:
        if not (args.ckpt_every > 0 and args.break_every > 0 and args.break_every % args.ckpt_every == 0):
            raise SystemExit("--ref-mode chain needs --break-every to be a multiple of --ckpt-every")
        ckpt_steps = set(range(args.ckpt_every, stop + 1, args.ckpt_every))
        break_steps = set(range(args.break_every, stop, args.break_every))  # not at the final step
        fail_steps = []
    keep_states = set(parse_int_list(args.keep_states))
    eval_after = parse_int_list(args.eval_after)
    eval_steps = set(range(0, stop + 1, args.eval_every)) | {stop}
    for f in list(fail_steps) + sorted(break_steps):
        eval_steps |= {f + a for a in eval_after if f + a <= stop}

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoTokenizer

    if args.mode == "finetune":
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        data = AlpacaData(args, tok)
        eval_b = data.eval_batches(args.eval_batch_size)
    else:
        data = TokenData(args)
        eval_b = data.eval_batches(args.eval_batches, args.eval_batch_size)
    score_b = data.score_batches(args.score_batches, args.score_batch_size)

    model = build_model(args, dev)
    params = dict(model.named_parameters())
    names = [n for n in params if n.startswith("gpt_neox.layers.") and n.endswith(ELIGIBLE_SUFFIXES)]
    opt = build_optimizer(args, model)

    state_path = args.out / "state.pt"
    metrics_path = args.out / "metrics.jsonl"
    step, ref_step, failures_done, train_acc = 0, 0, [], []
    ref = {n: params[n].detach().clone() for n in names}
    start_source = None
    if state_path.exists():
        start_source = state_path
    elif args.init_state is not None:
        start_source = args.init_state
    if start_source is not None:
        st = torch.load(start_source, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step, ref_step = st["step"], st.get("ref_step", 0)
        if st.get("ref") is not None:
            for n in names:
                ref[n].copy_(st["ref"][n].to(dev))
        if start_source == state_path:
            failures_done = st.get("failures", [])
            lines = st.get("metrics_lines", 0)
            if metrics_path.exists():
                kept = metrics_path.read_text().splitlines()[:lines]
                metrics_path.write_text("".join(x + "\n" for x in kept))
            torch.set_rng_state(st["rng"])
            torch.cuda.set_rng_state(st["cuda_rng"])
            log(f"resumed from {state_path} at step {step}")
        else:
            metrics_path.write_text("")
            log(f"started from {start_source} at step {step}")
        del st
    else:
        metrics_path.write_text("")
    if args.mode == "pretrain" and args.ref_state is not None and start_source != state_path:
        st = torch.load(args.ref_state, map_location="cpu", weights_only=False)
        for n in names:
            ref[n].copy_(st["model"][n].to(dev))
        ref_step = st["step"]
        del st
    if args.mode == "finetune":
        ref_step = 0  # the pretrained model

    if not (args.out / "config.json").exists():
        cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        cfg.update({"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__, "eligible_entries": sum(params[n].numel() for n in names),
                    "eligible_tensors": len(names), "parameters": sum(p.numel() for p in model.parameters()),
                    "eval_tokens": None, "fail_steps_list": fail_steps,
                    "checkpoint_steps": sorted(ckpt_steps), "break_steps": sorted(break_steps)})
        atomic_json(args.out / "config.json", cfg)

    def write_metric(rec):
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def save_state(path):
        n_lines = len(metrics_path.read_text().splitlines()) if metrics_path.exists() else 0
        blob = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "ref_step": ref_step,
                "ref": {n: ref[n].cpu() for n in names} if args.mode == "pretrain" or chain else None,
                "failures": failures_done, "metrics_lines": n_lines,
                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        torch.save(blob, tmp)
        os.replace(tmp, path)

    done_fail = {r["step"] for r in failures_done}
    t_start = time.time()
    # a resumed state has finished the evaluation/failure work of its step
    last_eval_step = step if start_source == state_path else None
    while True:
        # failures and evaluations happen at the current step, after its update
        if chain and step in ckpt_steps and step not in done_fail:
            # ExCP protocol: compress against the previous reconstructed checkpoint; resume from the
            # reconstruction only at break steps, otherwise keep training the uncompressed weights
            pre, ntok = evaluate(model, eval_b, dev)
            is_break = step in break_steps
            rec = {"kind": "checkpoint", "step": step, "break": is_break, "ref_step": ref_step,
                   "pre_loss": pre, "eval_tokens": ntok}
            if args.method != "nocomp":
                with torch.no_grad():
                    backup = {n: params[n].detach().clone() for n in names}
                rec.update(compress(model, params, names, ref, args.method, args.sparsity, args.kappa,
                                    args.stages, score_b, dev, opt=opt, alpha=args.excp_alpha,
                                    match_official=args.match_official))
                rec["post_loss"], _ = evaluate(model, eval_b, dev)
                with torch.no_grad():
                    for n in names:
                        ref[n].copy_(params[n].detach())
                        if not is_break:
                            params[n].copy_(backup[n])
                del backup
                torch.cuda.empty_cache()
            else:
                rec["post_loss"] = pre
            ref_step = step
            rec["immediate_damage"] = rec["post_loss"] - pre
            write_metric(rec)
            failures_done.append(rec)
            done_fail.add(step)
            log(f"checkpoint {step}{' (resume)' if is_break else ''}: pre {pre:.5f} recon {rec['post_loss']:.5f} "
                f"damage {1e3 * rec['immediate_damage']:.1f} mnats, deleted {rec.get('deletion_rate', 0):.3f} "
                f"({rec.get('compress_s', 0)} s)")
            if is_break:
                save_state(state_path)
            last_eval_step = step
        elif step in fail_steps and step not in done_fail:
            pre, ntok = evaluate(model, eval_b, dev)
            rec = {"kind": "failure", "step": step, "ref_step": ref_step, "pre_loss": pre, "eval_tokens": ntok}
            if args.method != "nocomp":
                rec.update(compress(model, params, names, ref, args.method, args.sparsity, args.kappa,
                                    args.stages, score_b, dev))
                rec["post_loss"], _ = evaluate(model, eval_b, dev)
            else:
                rec["post_loss"] = pre
            rec["immediate_damage"] = rec["post_loss"] - pre
            write_metric(rec)
            failures_done.append(rec)
            done_fail.add(step)
            log(f"failure at {step}: pre {pre:.5f} post {rec['post_loss']:.5f} "
                f"damage {1e3 * rec['immediate_damage']:.1f} mnats ({rec.get('compress_s', 0)} s)")
            save_state(state_path)
            last_eval_step = step
        elif step in eval_steps and last_eval_step != step:
            loss, ntok = evaluate(model, eval_b, dev)
            tl = float(np.mean(train_acc)) if train_acc else None
            write_metric({"kind": "eval", "step": step, "loss": loss, "train_loss": tl,
                          "lr": lr_at(args, max(0, step - 1)), "elapsed_s": round(time.time() - t_start, 1)})
            train_acc = []
            last_eval_step = step
            log(f"step {step}: val {loss:.5f} train {tl}")
        if step in keep_states and not (args.out / f"state_{step}.pt").exists():
            save_state(args.out / f"state_{step}.pt")
        if step >= stop:
            break
        # saved states always follow the evaluation/failure work of their step
        if (args.save_every and step > 0 and step % args.save_every == 0 and step not in fail_steps
                and step not in break_steps):
            save_state(state_path)
        # one optimizer update: step -> step + 1
        model.train()
        lr = lr_at(args, step)
        for g in opt.param_groups:
            g["lr"] = lr
        micro = [data.train_micro(step, i) for i in range(args.accum)]
        counts = [int((b["labels"][:, 1:] != -100).sum()) for b in micro]
        total_tok = sum(counts)
        loss_val = 0.0
        for b, c in zip(micro, counts):
            b = to_device(b, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16"):
                out = model(**b)
            loss = out.loss * (c / total_tok)
            loss.backward()
            loss_val += float(loss)
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if not math.isfinite(loss_val):
            raise RuntimeError(f"non-finite training loss at step {step}")
        train_acc.append(loss_val)
        step += 1
        if (not chain and args.mode == "pretrain" and args.ref_every and step % args.ref_every == 0
                and step not in fail_steps):
            with torch.no_grad():
                for n in names:
                    ref[n].copy_(params[n].detach())
            ref_step = step

    final_loss, ntok = evaluate(model, eval_b, dev)
    summary = {"method": args.method, "sparsity": args.sparsity, "kappa": args.kappa, "seed": args.seed,
               "mode": args.mode, "steps": step, "final_loss": final_loss, "final_ppl": math.exp(final_loss),
               "eval_tokens": ntok, "failures": failures_done, "wall_s": round(time.time() - t_start, 1),
               "gpu": torch.cuda.get_device_name(0), "host": socket.gethostname()}
    atomic_json(args.out / "summary.json", summary)
    (args.out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    if state_path.exists():
        state_path.unlink()  # copies requested by --keep-states stay
    log(f"done: final val {final_loss:.5f} ppl {math.exp(final_loss):.3f}")


if __name__ == "__main__":
    main()
