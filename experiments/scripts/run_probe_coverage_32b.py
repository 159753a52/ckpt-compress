"""Coverage-tracked random-probe residual scoring on a 32B causal LM.

Phases (each resumable; outputs are never overwritten):

  prepare   train base -> reference step -> checkpoint step, save both states
  score     cycle-1 scoring: gradient, style baselines, block HVP, probe rounds
  branches  single compression-recovery branches from the cycle-1 masks
  chains    repeated compression-recovery cycles for selected methods

GPU policy: the process refuses to run unless CUDA_VISIBLE_DEVICES equals the
configured value (default "5,6,7") and exactly three devices are visible.  The
model is split by decoder layers over cuda:0..2 inside one process; no
distributed process group is created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.lib.distributed_stats import ScoreMoments  # noqa: E402
from experiments.lib.probe_coverage import (  # noqa: E402
    CoverageTracker,
    coverage_probe_weights,
    cosine,
    first_round_meeting,
    mc_probe_weights,
    pack_mask,
    packed_intersection,
    packed_popcount,
    rademacher,
    residual_payload_bytes,
    spearman,
    unpack_mask,
    weighted_token_ce,
)
from experiments.lib.qwen_pipeline import PipelinedQwen, StochasticRoundingAdamW, train_step  # noqa: E402
from experiments.lib.residual_weibull import fit_weibull_from_moments, weibull_counts  # noqa: E402
from experiments.lib.streaming_selection import StreamingSelector, Target  # noqa: E402

SOURCE_FILES = [
    "experiments/scripts/run_probe_coverage_32b.py",
    "experiments/lib/probe_coverage.py",
    "experiments/lib/qwen_pipeline.py",
    "experiments/lib/streaming_selection.py",
    "experiments/lib/residual_weibull.py",
    "experiments/lib/distributed_stats.py",
    "experiments/lib/residual_budget.py",
]


# ============================================================================
# Utilities
# ============================================================================


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sync_all() -> None:
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def now() -> float:
    sync_all()
    return time.perf_counter()


def reset_peaks() -> None:
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def peaks() -> Dict[str, float]:
    return {
        f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / 2**30, 3)
        for i in range(torch.cuda.device_count())
    }


def write_json_new(path: Path, obj) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
    os.replace(tmp, path)


class Log:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, event: str, **fields) -> None:
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


def guard_gpus(expected: str) -> Dict[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != expected:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be {expected!r}, got {visible!r}")
    if torch.cuda.device_count() != len(expected.split(",")):
        raise SystemExit("unexpected visible device count")
    info = {"CUDA_VISIBLE_DEVICES": visible, "devices": []}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info["devices"].append(
            {
                "logical": f"cuda:{i}",
                "physical_index": int(expected.split(",")[i]),
                "name": props.name,
                "total_gib": round(props.total_memory / 2**30, 2),
                "uuid": str(getattr(props, "uuid", "")),
            }
        )
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception as exc:  # pragma: no cover
        info["nvidia_smi"] = f"unavailable: {exc}"
    return info


def provenance(cfg: dict) -> Dict[str, object]:
    commit_file = REPO / "DACP_CODE_COMMIT"
    return {
        "code_commit": commit_file.read_text().strip() if commit_file.exists() else None,
        "source_sha256": {p: sha256_file(REPO / p) for p in SOURCE_FILES},
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }


# ============================================================================
# Context: data, model, state
# ============================================================================


class Context:
    def __init__(self, cfg: dict, work: Path):
        self.cfg = cfg
        self.work = work
        work.mkdir(parents=True, exist_ok=True)
        self.log = Log(work / "events.jsonl")
        self.gpu = guard_gpus(cfg["cuda_visible_devices"])
        self.prov = provenance(cfg)
        self.log("start", gpu=self.gpu, provenance=self.prov)
        self._load_data()
        self._load_model()

    # ------------------------------------------------------------- data
    def _tokens(self, name: str, path: str) -> torch.Tensor:
        cache = self.work / f"tokens_{name}.pt"
        if cache.exists():
            return torch.load(cache, weights_only=True)
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.cfg["model_path"])
        text = Path(path).read_text(encoding="utf-8")
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        torch.save(ids, cache)
        return ids

    def _load_data(self) -> None:
        cfg = self.cfg
        L = cfg["seq_len"]
        data_meta = {}
        chunks = {}
        for name, key in (("train", "train_text"), ("val", "val_text")):
            ids = self._tokens(name, cfg[key])
            n = ids.numel() // L
            chunks[name] = ids[: n * L].view(n, L)
            data_meta[name] = {
                "path": cfg[key],
                "sha256": sha256_file(Path(cfg[key])),
                "tokens": int(ids.numel()),
                "chunks": n,
            }
        c = cfg["chunks"]
        tr = chunks["train"]
        self.score_ids = tr[c["score_start"] : c["score_start"] + c["score_count"]]
        self.gate_ids = tr[c["gate_start"] : c["gate_start"] + c["gate_count"]]
        self.val_ids = chunks["val"][: c["val_count"]]
        self.train_chunks = tr
        ck = cfg["train"]["ckpt_step"]
        used = ck + cfg["recovery_steps"] * cfg["cycles"]
        if used > c["score_start"] or c["score_start"] + c["score_count"] > c["gate_start"]:
            raise SystemExit("data splits overlap")
        if c["gate_start"] + c["gate_count"] > tr.shape[0] or self.val_ids.shape[0] < c["val_count"]:
            raise SystemExit("not enough chunks")
        data_meta["splits"] = {
            "train_steps_chunks": [0, ck],
            "recovery_chunks": [ck, used],
            "score_chunks": [c["score_start"], c["score_start"] + c["score_count"]],
            "gate_chunks": [c["gate_start"], c["gate_start"] + c["gate_count"]],
            "val_chunks": [0, c["val_count"]],
        }
        self.data_meta = data_meta
        self.log("data", **data_meta)

    # ------------------------------------------------------------ model
    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM

        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg["model_path"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=False,  # accelerate is not installed on the target host
        )
        model.eval()
        devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.pipe = PipelinedQwen(model, devs, self.cfg["layer_split"])
        idx = Path(self.cfg["model_path"]) / "model.safetensors.index.json"
        if not idx.exists():  # single-file checkpoints (smoke-test model)
            idx = Path(self.cfg["model_path"]) / "model.safetensors"
        self.model_meta = {
            "path": self.cfg["model_path"],
            "index_sha256": sha256_file(idx),
            "trainable_params": sum(self.pipe.param(n).numel() for n in self.pipe.trainable_names),
            "eligible_params": sum(self.pipe.param(n).numel() for n in self.pipe.eligible_names),
            "eligible_tensors": len(self.pipe.eligible_names),
            "all_params": sum(p.numel() for p in model.parameters()),
        }
        self.log("model", load_s=round(time.perf_counter() - t0, 1), **self.model_meta)

    # ------------------------------------------------------------ state
    def new_optimizer(self) -> StochasticRoundingAdamW:
        t = self.cfg["train"]
        return StochasticRoundingAdamW(
            self.pipe, t["lr"], tuple(t["betas"]), t["eps"], t["weight_decay"], t["clip"], t["sr_seed"]
        )

    def snapshot(self, names: Sequence[str]) -> Dict[str, torch.Tensor]:
        return {n: self.pipe.param(n).detach().to("cpu", copy=True) for n in names}

    @torch.no_grad()
    def load_params(self, values: Dict[str, torch.Tensor]) -> None:
        for n, v in values.items():
            self.pipe.param(n).copy_(v.to(self.pipe.param(n).device))

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        b = self.cfg["chunks"]["eval_batch"]
        t0 = now()
        out = {"val": self.pipe.evaluate(self.val_ids, b), "gate": self.pipe.evaluate(self.gate_ids, b)}
        out["eval_s"] = round(now() - t0, 2)
        return out


# ============================================================================
# Phase: prepare
# ============================================================================


def ckpt_dir(ctx: Context) -> Path:
    return ctx.work / "ckpt"


def phase_prepare(ctx: Context) -> None:
    from safetensors.torch import save_file

    out = ckpt_dir(ctx)
    if (out / "meta.json").exists():
        ctx.log("prepare_skip", reason="checkpoint exists")
        return
    out.mkdir(exist_ok=True)
    t = ctx.cfg["train"]
    opt = ctx.new_optimizer()
    opt.init_state()
    meta = {"evals": {}, "train_log": [], "gpu": ctx.gpu, "provenance": ctx.prov, "data": ctx.data_meta, "model": ctx.model_meta}
    meta["evals"]["base"] = ctx.evaluate()
    ctx.log("eval", tag="base", **meta["evals"]["base"])
    ref_state = None
    reset_peaks()
    t0 = now()
    for step in range(1, t["ckpt_step"] + 1):
        ids = ctx.train_chunks[step - 1 : step]
        rec = train_step(ctx.pipe, opt, ids)
        meta["train_log"].append(rec)
        if step % 10 == 0:
            ctx.log("train", **rec, pace_s=round((now() - t0) / step, 3))
        if step == t["ref_step"]:
            ref_state = ctx.snapshot(ctx.pipe.trainable_names)
            meta["evals"]["ref"] = ctx.evaluate()
            ctx.log("eval", tag="ref", **meta["evals"]["ref"])
    meta["train_seconds"] = round(now() - t0, 1)
    meta["train_peak_gib"] = peaks()
    meta["evals"]["ckpt"] = ctx.evaluate()
    ctx.log("eval", tag="ckpt", **meta["evals"]["ckpt"])
    cur = ctx.snapshot(ctx.pipe.trainable_names)
    opt.state_to(False)
    files = {
        "theta_ref": ref_state,
        "theta": cur,
        "exp_avg": {n: v.contiguous() for n, v in opt.exp_avg.items()},
        "exp_avg_sq": {n: v.contiguous() for n, v in opt.exp_avg_sq.items()},
    }
    meta["files"] = {}
    for name, tensors in files.items():
        path = out / f"{name}.safetensors"
        t1 = time.perf_counter()
        save_file(tensors, str(path))
        meta["files"][name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "save_s": round(time.perf_counter() - t1, 1)}
        ctx.log("saved", name=name, **meta["files"][name])
    meta["step"] = opt.step_count
    write_json_new(out / "meta.json", meta)


def load_ckpt(ctx: Context, verify_hash: bool) -> Tuple[Dict, Dict, Dict, Dict, int, dict]:
    from safetensors.torch import load_file

    meta = json.loads((ckpt_dir(ctx) / "meta.json").read_text())
    loaded = {}
    for name in ("theta_ref", "theta", "exp_avg", "exp_avg_sq"):
        info = meta["files"][name]
        if verify_hash:
            digest = sha256_file(Path(info["path"]))
            if digest != info["sha256"]:
                raise SystemExit(f"checkpoint hash mismatch for {name}")
        loaded[name] = load_file(info["path"])
    ctx.log("ckpt_loaded", step=meta["step"], verified=verify_hash)
    return loaded["theta_ref"], loaded["theta"], loaded["exp_avg"], loaded["exp_avg_sq"], meta["step"], meta


# ============================================================================
# Scoring
# ============================================================================


class Scorer:
    """Holds residual d (GPU bf16), gradient g (CPU bf16) and shared bookkeeping."""

    def __init__(self, ctx: Context, ref: Dict[str, torch.Tensor], out_dir: Path):
        self.ctx = ctx
        self.pipe = ctx.pipe
        self.names = list(self.pipe.eligible_names)
        self.ref = ref
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "masks").mkdir(exist_ok=True)
        self.numels = [self.pipe.param(n).numel() for n in self.names]
        self.total = sum(self.numels)
        self.layer_members: List[List[int]] = [[] for _ in range(self.pipe.num_layers)]
        for i, n in enumerate(self.names):
            self.layer_members[self.pipe.layer_of[n]].append(i)
        stride = ctx.cfg["subsample_stride"]
        self.sub_idx = [torch.arange(0, k, stride) for k in self.numels]
        self.d: List[torch.Tensor] = []
        self.g: List[torch.Tensor] = []
        self.hd: Optional[List[torch.Tensor]] = None
        self.records: Dict[str, object] = {}
        self.targets_cache: Dict[str, List[Target]] = {}

    # ---------------------------------------------------------- residual
    @torch.no_grad()
    def build_residual(self) -> Dict[str, object]:
        zeros = 0
        inexact = 0
        sq = 0.0
        self.fix = []  # per tensor: (flat idx, ref values, theta values) where bf16 d is inexact
        for n in self.names:
            p = self.pipe.param(n)
            b = self.ref[n].to(p.device)
            d = (p.float() - b.float()).to(torch.bfloat16)
            bad = ((p - d) != b) | ((b + d) != p)
            idx = bad.reshape(-1).nonzero().reshape(-1)
            self.fix.append((idx, b.reshape(-1)[idx].clone(), p.detach().reshape(-1)[idx].clone()))
            inexact += int(idx.numel())
            zeros += int((d == 0).sum())
            sq += float(d.float().square().sum().double())
            self.d.append(d)
            del b, bad
        rec = {
            "eligible_coords": self.total,
            "zero_residual_coords": zeros,
            "zero_residual_fraction": zeros / self.total,
            "residual_l2": math.sqrt(sq),
            "bf16_inexact_residual_coords": inexact,
        }
        # Masks are applied and undone arithmetically with d; coordinates where
        # bf16 subtraction/addition is inexact are overwritten from stored values,
        # so both directions are bit-exact.
        self.records["residual"] = rec
        self.ctx.log("residual", **rec)
        return rec

    # ---------------------------------------------------------- batches
    def _micro_batches(self):
        mb = self.ctx.cfg["score_micro_batch"]
        ids = self.ctx.score_ids
        for s in range(0, ids.shape[0], mb):
            yield s, ids[s : s + mb]

    @property
    def positions(self) -> int:
        ids = self.ctx.score_ids
        return ids.shape[0] * (ids.shape[1] - 1)

    # ---------------------------------------------------------- gradient
    def compute_gradient(self) -> Dict[str, object]:
        pipe = self.pipe
        pipe.set_requires_grad(self.names)
        pipe.zero_grad()
        reset_peaks()
        t0 = now()
        total_loss = 0.0
        for _, ids in self._micro_batches():
            logits = pipe.forward(ids)
            loss, _ = pipe.nll_sum(logits, ids)
            total_loss += float(loss.detach().double())
            (loss / self.positions).backward()
            del logits, loss
        t1 = now()
        # Keep g on its GPU until the probe phase needs the memory.
        self.g = [pipe.param(n).grad.detach() for n in self.names]
        pipe.zero_grad()
        pipe.set_requires_grad([])
        rec = {
            "score_loss": total_loss / self.positions,
            "fwd_bwd_s": round(t1 - t0, 3),
            "offload_s": round(now() - t1, 3),
            "peak_gib": peaks(),
        }
        self.records["gradient"] = rec
        self.ctx.log("gradient", **rec)
        return rec

    def g_dev(self, i: int) -> torch.Tensor:
        return self.g[i].to(self.d[i].device)

    def offload_gradient(self) -> float:
        t0 = now()
        if self.g and self.g[0].is_cuda:
            self.g = [g.to("cpu") for g in self.g]
            torch.cuda.empty_cache()
        return round(now() - t0, 3)

    # ---------------------------------------------------------- selection
    def targets_for(self, prefix: str, moments: Optional[List[ScoreMoments]]) -> Tuple[List[Target], Dict[str, object]]:
        cfg = self.ctx.cfg
        targets = []
        meta = {}
        for sp in cfg["sparsities"]:
            k = int(round(sp * self.total))
            targets.append(Target(f"{prefix}_global_{sp}", list(range(len(self.names))), k))
            if moments is not None:
                fits = [fit_weibull_from_moments(m) for m in moments]
                sizes = [sum(self.numels[i] for i in mem) for mem in self.layer_members]
                info = {"cap": cfg["weibull_cap"], "valid_fits": sum(bool(f["valid"]) for f in fits)}
                try:
                    weibull_counts(fits, sizes, k, sp, cfg["weibull_cap_original"])
                    info["original_cap_feasible"] = True
                except ValueError as exc:
                    info["original_cap_feasible"] = False
                    info["original_cap_error"] = str(exc)
                counts, alloc = weibull_counts(fits, sizes, k, sp, cfg["weibull_cap"])
                info["layer_rates"] = [c / s for c, s in zip(counts, sizes)]
                info["threshold"] = alloc.get("threshold")
                info["fallback"] = alloc.get("fallback")
                meta[f"{prefix}_weibull_{sp}"] = info
                for layer, (mem, c) in enumerate(zip(self.layer_members, counts)):
                    targets.append(Target(f"{prefix}_weibull_{sp}@{layer}", mem, int(c)))
        return targets, meta

    def select(self, prefix: str, score_fn: Callable[[int], torch.Tensor], weibull: bool) -> Dict[str, object]:
        """Select masks for all sparsities; save packed masks; return summary."""
        t0 = now()
        n_layers = self.pipe.num_layers
        acc = [[0, 0.0, 0.0, 0, 0.0] for _ in range(n_layers)]
        subs: List[torch.Tensor] = [None] * len(self.names)

        def hook(i: int, s: torch.Tensor) -> None:
            layer = self.pipe.layer_of[self.names[i]]
            a = acc[layer]
            sd = s.double()
            a[0] += s.numel()
            a[1] += float(sd.sum())
            a[2] += float(sd.square().sum())
            a[3] += int((s == 0).sum())
            a[4] = max(a[4], float(s.max()))
            subs[i] = s.reshape(-1)[self.sub_idx[i].to(s.device)].cpu()

        sel = StreamingSelector(len(self.names), score_fn)
        sel.first_pass(hook)
        moments = [ScoreMoments(int(a[0]), a[1], a[2], int(a[3]), a[4]) for a in acc] if weibull else None
        targets, meta = self.targets_for(prefix, moments)
        res = sel.select(targets)
        masks: Dict[str, List[torch.Tensor]] = {}
        for sp in self.ctx.cfg["sparsities"]:
            masks[f"{prefix}_global_{sp}"] = [res[f"{prefix}_global_{sp}"]["packed"][i] for i in range(len(self.names))]
            if weibull:
                full = [None] * len(self.names)
                for layer in range(n_layers):
                    part = res[f"{prefix}_weibull_{sp}@{layer}"]["packed"]
                    for i, v in part.items():
                        full[i] = v
                masks[f"{prefix}_weibull_{sp}"] = full
        summary = {
            "select_s": round(now() - t0, 3),
            "weibull": meta,
            "subsample": torch.cat(subs),
            "masks": masks,
        }
        return summary

    def save_masks(self, masks: Dict[str, List[torch.Tensor]], names: Sequence[str]) -> None:
        for name in names:
            path = self.out / "masks" / f"{name}.pt"
            if path.exists():
                raise FileExistsError(path)
            torch.save(masks[name], path)

    # ---------------------------------------------------------- evaluation
    @torch.no_grad()
    def apply_mask(self, packed: List[torch.Tensor], sign: int) -> None:
        """sign=-1: theta -> b on pruned coords; sign=+1 restores (exact when d exact)."""
        for i, n in enumerate(self.names):
            p = self.pipe.param(n)
            m = unpack_mask(packed[i].to(p.device), p.shape)
            if sign < 0:
                p.sub_(torch.where(m, self.d[i], torch.zeros_like(self.d[i])))
            else:
                p.add_(torch.where(m, self.d[i], torch.zeros_like(self.d[i])))
            idx, bvals, tvals = self.fix[i]
            if idx.numel():
                hit = m.reshape(-1)[idx]
                flat = p.view(-1)
                flat[idx[hit]] = (bvals if sign < 0 else tvals)[hit]

    def immediate_eval(self, packed: List[torch.Tensor]) -> Dict[str, object]:
        self.apply_mask(packed, -1)
        ev = self.ctx.evaluate()
        self.apply_mask(packed, +1)
        return ev

    # ---------------------------------------------------------- baselines
    def score_baselines(self) -> Dict[str, object]:
        out = {}
        defs = {
            "excp_style": lambda i: self.d[i].float().abs(),
            "first_order": lambda i: (self.g_dev(i).float() * self.d[i].float()).abs(),
            "inshrinkerator_style": lambda i: (
                self.g_dev(i).float() * self.pipe.param(self.names[i]).float()
            ).abs(),
        }
        for name, fn in defs.items():
            summ = self.select(name, fn, weibull=False)
            out[name] = summ
        return out

    # ---------------------------------------------------------- HVP
    def compute_hvp(self, layer_limit: Optional[int] = None) -> Dict[str, object]:
        pipe = self.pipe
        reset_peaks()
        t0 = now()
        cached = []
        with torch.no_grad():
            for _, ids in self._micro_batches():
                _, inputs = pipe.forward(ids, collect_inputs=True)
                cached.append((ids, inputs))
        t_cache = now() - t0
        hd: List[Optional[torch.Tensor]] = [None] * len(self.names)
        per_block = []
        layers = range(pipe.num_layers if layer_limit is None else layer_limit)
        for layer in layers:
            tb = now()
            names = pipe.layer_eligible[layer]
            vec = {n: self.d[self.names.index(n)] for n in names}
            acc = None
            for ids, inputs in cached:
                h = pipe.block_hvp(layer, inputs[layer], ids, vec, self.positions)
                if acc is None:
                    acc = h
                else:
                    for n in names:
                        acc[n] += h[n]
                del h
            for n in names:
                hd[self.names.index(n)] = acc[n].to("cpu")
            del acc
            per_block.append(round(now() - tb, 3))
            if layer % 8 == 0:
                self.ctx.log("hvp_block", layer=layer, seconds=per_block[-1])
        pipe.set_requires_grad([])
        pipe.zero_grad()
        del cached
        for i, n in enumerate(self.names):
            if hd[i] is None:  # smoke mode only
                hd[i] = torch.zeros(self.pipe.param(n).shape, dtype=torch.float32)
        self.hd = hd
        rec = {
            "cache_forward_s": round(t_cache, 3),
            "hvp_s": round(now() - t0, 3),
            "per_block_s": per_block,
            "layer_limit": layer_limit,
            "peak_gib": peaks(),
            "passes": f"{len(layers)} blocks x {self.ctx.score_ids.shape[0]} sequences",
        }
        self.records["hvp"] = rec
        self.ctx.log("hvp", **{k: v for k, v in rec.items() if k != "per_block_s"})
        return rec

    def hvp_score(self, i: int) -> torch.Tensor:
        d = self.d[i].float()
        hd = self.hd[i].to(d.device)
        return (-self.g_dev(i).float() * d + 0.5 * d * hd).abs()


# ============================================================================
# Probes
# ============================================================================


class ProbeRun:
    def __init__(self, scorer: Scorer, seed: int, max_rounds: int, estimator: str):
        self.s = scorer
        self.pipe = scorer.pipe
        self.seed = seed
        self.max_rounds = max_rounds
        self.estimator = estimator
        self.tracker = CoverageTracker(scorer.positions, max_rounds, device=self.pipe.devices[-1])
        self.gen = torch.Generator(device=self.pipe.devices[-1])
        self.gen.manual_seed(10_000 + seed)
        self.u = [torch.zeros(d.shape, dtype=torch.float32, device=d.device) for d in scorer.d]
        self.top1 = torch.zeros(scorer.positions, dtype=torch.float64, device=self.pipe.devices[-1])
        self.rounds = 0

    def round(self) -> Dict[str, object]:
        pipe = self.pipe
        s = self.s
        pipe.set_requires_grad(s.names)
        pipe.zero_grad()
        reset_peaks()
        timing = {"forward": 0.0, "sample": 0.0, "backward": 0.0}
        new_count = 0
        weight_sq = 0.0
        offset = 0
        t_round = now()
        for _, ids in s._micro_batches():
            t0 = now()
            logits = pipe.forward(ids)
            flat, _ = pipe.shifted(logits, ids)
            t1 = now()
            with torch.no_grad():
                probs = torch.softmax(flat.detach().float(), dim=-1)
                if self.rounds == 0:
                    self.top1[offset : offset + probs.shape[0]] = probs.max(dim=1).values.double()
                tokens = torch.multinomial(probs, 1, generator=self.gen).reshape(-1)
                p_y = probs.gather(1, tokens[:, None]).reshape(-1)
                del probs
                npos = tokens.numel()
                if self.estimator == "coverage":
                    is_new = self.tracker_update(offset, tokens, p_y)
                    xi = rademacher(npos, self.gen, tokens.device)
                    w = coverage_probe_weights(p_y, is_new, xi, s.positions)
                    new_count += int(is_new.sum())
                else:
                    self.tracker_update(offset, tokens, p_y)
                    w = mc_probe_weights(npos, s.positions, tokens.device)
                weight_sq += float(w.square().sum())
            t2 = now()
            loss = weighted_token_ce(flat, tokens, w)
            loss.backward()
            del logits, flat, loss
            t3 = now()
            timing["forward"] += t1 - t0
            timing["sample"] += t2 - t1
            timing["backward"] += t3 - t2
            offset += npos
        self.tracker.rounds += 1
        t4 = now()
        with torch.no_grad():
            alpha = 0.0
            vsq = 0.0
            for i, n in enumerate(s.names):
                g = pipe.param(n).grad
                alpha += float((g.float() * s.d[i].float()).sum().double())
                vsq += float(g.float().square().sum().double())
            for i, n in enumerate(s.names):
                self.u[i].add_(pipe.param(n).grad.float(), alpha=alpha)
        pipe.zero_grad()
        pipe.set_requires_grad([])
        self.rounds += 1
        rec = {
            "round": self.rounds,
            "new_tokens": new_count,
            "round_repeat_rate": 1.0 - new_count / s.positions if self.estimator == "coverage" else None,
            "alpha": alpha,
            "probe_norm": math.sqrt(vsq),
            "weight_sq_sum": weight_sq,
            "forward_s": round(timing["forward"], 3),
            "sample_s": round(timing["sample"], 3),
            "backward_s": round(timing["backward"], 3),
            "update_s": round(now() - t4, 3),
            "round_s": round(now() - t_round, 3),
            "peak_gib": peaks(),
        }
        rec.update(self.tracker.stats(tuple(r["tau"] for r in s.ctx.cfg["probe"]["rules"] if r["key"].endswith("mass"))))
        # Coverage stratified by the model's top-1 probability at each position.
        q = self.tracker.mass
        strata = {}
        for lo, hi in ((0.0, 0.5), (0.5, 0.9), (0.9, 1.01)):
            sel = (self.top1 >= lo) & (self.top1 < hi)
            cnt = int(sel.sum())
            strata[f"top1_{lo:g}_{min(hi, 1.0):g}"] = {
                "positions": cnt,
                "mean_mass": float(q[sel].mean()) if cnt else None,
                "frac_ge_0.9": float((q[sel] >= 0.9).double().mean()) if cnt else None,
                "frac_ge_0.95": float((q[sel] >= 0.95).double().mean()) if cnt else None,
            }
        rec["strata"] = strata
        return rec

    def tracker_update(self, offset: int, tokens: torch.Tensor, p_y: torch.Tensor) -> torch.Tensor:
        """Update the slice of positions covered by this micro-batch."""
        tr = self.tracker
        n = tokens.numel()
        r = tr.rounds
        seen = tr.seen[offset : offset + n, :r]
        is_new = ~(seen == tokens[:, None]).any(dim=1)
        tr.seen[offset : offset + n, r] = tokens
        tr.mass[offset : offset + n] += torch.where(is_new, p_y.double(), torch.zeros_like(p_y.double()))
        tr.mass.clamp_(max=1.0)
        tr.draws += n
        tr.repeats += int((~is_new).sum())
        return is_new

    def score(self, i: int) -> torch.Tensor:
        d = self.s.d[i].float()
        u = self.u[i]
        if self.estimator == "mc":
            u = u / self.rounds
        return (-self.s.g_dev(i).float() * d + 0.5 * d * u).abs()

    def u_subsample(self) -> torch.Tensor:
        return torch.cat(
            [self.u[i].reshape(-1)[self.s.sub_idx[i].to(self.u[i].device)].cpu() for i in range(len(self.u))]
        )

    def free(self) -> None:
        self.u = []
        torch.cuda.empty_cache()


# ============================================================================
# Phase: score (cycle 1)
# ============================================================================


def mask_overlap(a: List[torch.Tensor], b: List[torch.Tensor], k: int) -> float:
    inter = sum(packed_intersection(x, y) for x, y in zip(a, b))
    return inter / k if k else 1.0


def phase_score(ctx: Context, args) -> None:
    out = ctx.work / "cycle1"
    final = out / "score_records.json"
    if final.exists():
        ctx.log("score_skip", reason="exists")
        return
    ref, theta, _, _, step, meta = load_ckpt(ctx, verify_hash=not args.skip_hash)
    ctx.load_params(theta)
    s = Scorer(ctx, {n: ref[n] for n in ctx.pipe.eligible_names}, out)
    del ref
    cfg = ctx.cfg
    sparsities = cfg["sparsities"]
    rec: Dict[str, object] = {"step": step, "gpu": ctx.gpu, "provenance": ctx.prov, "data": ctx.data_meta, "model": ctx.model_meta}
    s.build_residual()
    rec["uncompressed_eval"] = ctx.evaluate()
    ctx.log("eval", tag="uncompressed", **rec["uncompressed_eval"])
    s.compute_gradient()
    ks = {sp: int(round(sp * s.total)) for sp in sparsities}

    methods: Dict[str, Dict[str, object]] = {}
    for name, summ in s.score_baselines().items():
        methods[name] = summ
    s.compute_hvp(layer_limit=args.hvp_layer_limit)
    hvp = s.select("hvp", s.hvp_score, weibull=True)
    methods["hvp"] = hvp
    hd_sub = torch.cat([s.hd[i].reshape(-1)[s.sub_idx[i]] for i in range(len(s.names))])

    def summarize(name: str, summ: dict) -> dict:
        s.save_masks(summ["masks"], list(summ["masks"]))
        info = {"select_s": summ["select_s"], "weibull": summ["weibull"], "evals": {}, "overlap_hvp": {}}
        for mname, packed in summ["masks"].items():
            info["evals"][mname] = s.immediate_eval(packed)
            sp = float(mname.rsplit("_", 1)[1])
            ref_mask = hvp["masks"][f"hvp_global_{sp}"]
            info["overlap_hvp"][mname] = mask_overlap(packed, ref_mask, ks[sp])
            ctx.log("immediate", mask=mname, val_loss=info["evals"][mname]["val"]["loss"], gate_loss=info["evals"][mname]["gate"]["loss"])
        info["spearman_hvp"] = spearman(summ["subsample"], hvp["subsample"])
        return info

    rec["methods"] = {name: summarize(name, summ) for name, summ in methods.items()}
    # After ten apply/undo cycles the parameters must equal the checkpoint bitwise.
    rec["restore_mismatch_after_evals"] = sum(
        int((ctx.pipe.param(n).detach().cpu() != theta[n]).sum()) for n in ctx.pipe.eligible_names
    )
    ctx.log("restore_check", mismatch=rec["restore_mismatch_after_evals"])
    if rec["restore_mismatch_after_evals"]:
        raise RuntimeError("mask apply/undo is not bit-exact")
    del theta
    rec["residual"] = s.records["residual"]
    rec["gradient"] = s.records["gradient"]
    rec["hvp"] = s.records["hvp"]

    # ------------------------------------------------ probes
    s.hd = None
    rec["gradient_offload_s"] = s.offload_gradient()
    pcfg = cfg["probe"]
    rec["probes"] = {}
    for seed in pcfg["seeds"][: args.max_seeds]:
        run = ProbeRun(s, seed, pcfg["max_rounds"], pcfg["estimator"])
        prev = None
        prev_sub = None
        rounds = []
        for _ in range(min(pcfg["max_rounds"], args.max_rounds)):
            r = run.round()
            summ = s.select(f"probe_s{seed}_m{run.rounds}", run.score, weibull=True)
            r["select_s"] = summ["select_s"]
            r["weibull"] = summ["weibull"]
            r["evals"] = {}
            for mname, packed in summ["masks"].items():
                key = mname.split("_", 3)[-1]  # e.g. global_0.95
                sp = float(key.rsplit("_", 1)[1])
                r["evals"][key] = s.immediate_eval(packed)
                r[f"overlap_hvp_{key}"] = mask_overlap(packed, hvp["masks"][f"hvp_{key}"], ks[sp])
                if prev is not None:
                    r[f"overlap_prev_{key}"] = mask_overlap(packed, prev[mname.replace(f"_m{run.rounds}_", f"_m{run.rounds - 1}_")], ks[sp])
            r["spearman_hvp"] = spearman(summ["subsample"], hvp["subsample"])
            r["spearman_prev"] = spearman(summ["subsample"], prev_sub) if prev_sub is not None else None
            usub = run.u_subsample()
            r["cosine_u_hd_subsample"] = cosine(usub / (run.rounds if run.estimator == "mc" else 1), hd_sub)
            r["norm_ratio_u_hd_subsample"] = float(usub.norm() / hd_sub.norm()) if hd_sub.norm() > 0 else None
            # Persist only masks that a branch can use: fixed rounds, the first
            # round meeting each rule, and the last round (unmet-rule fallback).
            rounds.append(r)
            last = run.rounds >= min(pcfg["max_rounds"], args.max_rounds)
            now_picks = {k: v["round"] for k, v in rule_picks(cfg, rounds).items() if v["met"]}
            first_hit = {k for k, m in now_picks.items() if m == run.rounds}
            keep = []
            if run.rounds in pcfg["fixed_rounds"] or first_hit or last:
                keep += [n for n in summ["masks"] if "_global_" in n]
            if "mean95" in first_hit or (last and "mean95" not in now_picks):
                keep += [n for n in summ["masks"] if "_weibull_" in n]
            s.save_masks(summ["masks"], keep)
            r["saved_masks"] = keep
            prev = summ["masks"]
            prev_sub = summ["subsample"]
            ctx.log(
                "probe_round",
                seed=seed,
                round=run.rounds,
                mean_mass=r["mean_mass"],
                p05=r["p05_mass"],
                round_s=r["round_s"],
                spearman_hvp=r["spearman_hvp"],
                val_095=r["evals"]["global_0.95"]["val"]["loss"] if "global_0.95" in r["evals"] else None,
            )
        run.free()
        rec["probes"][str(seed)] = rounds
    s.hd = None
    write_json_new(final, rec)


# ============================================================================
# Branches
# ============================================================================


def rule_picks(cfg: dict, rounds: List[dict]) -> Dict[str, Dict[str, object]]:
    picks = {}
    for rule in cfg["probe"]["rules"]:
        if not all(rule["key"] in r or r["round"] < rule.get("min_round", 1) for r in rounds):
            continue
        filled = [dict(r) for r in rounds]
        for r in filled:
            r.setdefault(rule["key"], -1.0)
        m, met = first_round_meeting(filled, rule["key"], rule["tau"], rule.get("min_round", 1))
        picks[rule["name"]] = {"round": m, "met": met}
    return picks


def branch_specs(ctx: Context, rec: dict) -> List[dict]:
    cfg = ctx.cfg
    specs = [{"id": "no_compression_r0", "mask": None}, {"id": "no_compression_r1", "mask": None}]
    for sp in cfg["sparsities"]:
        for name in ("excp_style", "inshrinkerator_style", "first_order"):
            specs.append({"id": f"{name}_{sp}", "mask": f"{name}_global_{sp}", "sparsity": sp})
        specs.append({"id": f"hvp_global_{sp}", "mask": f"hvp_global_{sp}", "sparsity": sp})
        specs.append({"id": f"dacp_{sp}", "mask": f"hvp_weibull_{sp}", "sparsity": sp})
        for seed, rounds in rec["probes"].items():
            picks = rule_picks(cfg, rounds)
            ms = set(m for m in cfg["probe"]["fixed_rounds"] if m <= len(rounds))
            ms |= {p["round"] for p in picks.values()}
            for m in sorted(ms):
                specs.append({"id": f"probe_s{seed}_m{m}_global_{sp}", "mask": f"probe_s{seed}_m{m}_global_{sp}", "sparsity": sp, "seed": int(seed), "round": m})
            mw = picks.get("mean95", {}).get("round")
            if mw is not None:
                specs.append({"id": f"probe_s{seed}_m{mw}_weibull_{sp}", "mask": f"probe_s{seed}_m{mw}_weibull_{sp}", "sparsity": sp, "seed": int(seed), "round": mw})
    return specs


@torch.no_grad()
def build_restored(ctx: Context, ref: dict, theta: dict, packed: Optional[List[torch.Tensor]]) -> Dict[str, object]:
    """Load theta into the model, then reset pruned residual coordinates to ref."""
    pipe = ctx.pipe
    ctx.load_params(theta)
    stats = {"pruned": 0, "pruned_nonzero": 0, "kept": 0, "kept_nonzero": 0, "payload": {"bitmap": 0, "coo": 0, "chosen": 0}}
    if packed is None:
        return stats
    for i, n in enumerate(pipe.eligible_names):
        p = pipe.param(n)
        m = unpack_mask(packed[i].to(p.device), p.shape)
        b = ref[n].to(p.device)
        nz = p != b
        pruned = int(m.sum())
        kept = p.numel() - pruned
        stats["pruned"] += pruned
        stats["pruned_nonzero"] += int((m & nz).sum())
        stats["kept"] += kept
        stats["kept_nonzero"] += int((~m & nz).sum())
        pb = residual_payload_bytes(p.numel(), kept)
        for key in pb:
            stats["payload"][key] += pb[key]
        p.copy_(torch.where(m, b, p))
        del m, b, nz
    return stats


def run_recovery(ctx: Context, opt: StochasticRoundingAdamW, start_chunk: int, steps: int) -> List[dict]:
    log = []
    for j in range(steps):
        ids = ctx.train_chunks[start_chunk + j : start_chunk + j + 1]
        log.append(train_step(ctx.pipe, opt, ids))
    return log


def verify_archive(ctx: Context, ref: dict, theta: dict, packed: List[torch.Tensor], path: Path) -> Dict[str, object]:
    """Serialize (keep bitmap, kept bf16 values), decode against the reference,
    and compare bitwise with the restored model.  Must run before recovery.

    Kept coordinates are stored as their bf16 values: the same 2 bytes as a bf16
    residual, but exact even where theta - ref is not representable in bf16.
    """
    pipe = ctx.pipe
    archive = {}
    for i, n in enumerate(pipe.eligible_names):
        keep = ~unpack_mask(packed[i], theta[n].shape)
        archive[n] = {"keep": pack_mask(keep), "values": theta[n][keep].contiguous()}
    torch.save(archive, path)
    size = path.stat().st_size
    del archive
    loaded = torch.load(path, weights_only=True)
    mismatch = 0
    for n in pipe.eligible_names:
        keep = unpack_mask(loaded[n]["keep"], theta[n].shape)
        rec = ref[n].clone()
        rec[keep] = loaded[n]["values"]
        mismatch += int((rec != pipe.param(n).detach().cpu()).sum())
    path.unlink()
    return {"archive_file_bytes": size, "decode_mismatch_coords": mismatch, "value_encoding": "bf16 kept values"}


def phase_branches(ctx: Context, args) -> None:
    rec = json.loads((ctx.work / "cycle1" / "score_records.json").read_text())
    specs = branch_specs(ctx, rec)
    if args.only:
        specs = [s for s in specs if any(tok in s["id"] for tok in args.only.split(","))]
    out = ctx.work / "cycle1" / "branches"
    out.mkdir(exist_ok=True)
    ref, theta, m1, m2, step, meta = load_ckpt(ctx, verify_hash=not args.skip_hash)
    opt = ctx.new_optimizer()
    R = min(ctx.cfg["recovery_steps"], args.max_recovery)
    verified = False
    for spec in specs:
        path = out / f"{spec['id']}.json"
        if path.exists():
            continue
        t0 = now()
        packed = torch.load(ctx.work / "cycle1" / "masks" / f"{spec['mask']}.pt", weights_only=True) if spec["mask"] else None
        stats = build_restored(ctx, ref, theta, packed)
        archive = None
        if packed is not None and not verified and args.verify_archive:
            archive = verify_archive(ctx, ref, theta, packed, out / "archive_tmp.pt")
            verified = True
        opt.load_state(m1, m2, step)
        immediate = ctx.evaluate()
        reset_peaks()
        tr0 = now()
        train_log = run_recovery(ctx, opt, ctx.cfg["train"]["ckpt_step"], R)
        train_s = now() - tr0
        final = ctx.evaluate()
        result = {
            "spec": spec,
            "restore": stats,
            "immediate": immediate,
            "final": final,
            "recovery_steps": R,
            "recovery_train_s": round(train_s, 2),
            "recovery_peak_gib": peaks(),
            "train_log": train_log,
            "branch_s": round(now() - t0, 2),
        }
        if archive is not None:
            result["archive_check"] = archive
        write_json_new(path, result)
        ctx.log("branch", id=spec["id"], imm_val=immediate["val"]["loss"], final_val=final["val"]["loss"], pruned=stats["pruned"])
        opt.state_to(False)
        torch.cuda.empty_cache()


# ============================================================================
# Chains (repeated compression and recovery)
# ============================================================================


def chain_mask(ctx: Context, method: str, ref: dict, out: Path, sp: float, cycle: int, args) -> Tuple[List[torch.Tensor], dict]:
    s = Scorer(ctx, ref, out)
    s.build_residual()
    s.compute_gradient()
    info: Dict[str, object] = {"residual": s.records["residual"], "gradient": s.records["gradient"]}
    cfg = ctx.cfg
    old = cfg["sparsities"]
    cfg["sparsities"] = [sp]
    try:
        if method == "excp_style":
            summ = s.select(f"c{cycle}_{method}", lambda i: s.d[i].float().abs(), weibull=False)
            key = f"c{cycle}_{method}_global_{sp}"
        elif method == "inshrinkerator_style":
            summ = s.select(
                f"c{cycle}_{method}",
                lambda i: (s.g_dev(i).float() * s.pipe.param(s.names[i]).float()).abs(),
                weibull=False,
            )
            key = f"c{cycle}_{method}_global_{sp}"
        elif method in ("dacp", "hvp_global"):
            s.compute_hvp(layer_limit=args.hvp_layer_limit)
            info["hvp"] = s.records["hvp"]
            summ = s.select(f"c{cycle}_hvp", s.hvp_score, weibull=(method == "dacp"))
            key = f"c{cycle}_hvp_{'weibull' if method == 'dacp' else 'global'}_{sp}"
            s.hd = None
        elif method.startswith("probe_"):
            rule = next(r for r in cfg["probe"]["rules"] if r["name"] == method[len("probe_"):])
            info["gradient_offload_s"] = s.offload_gradient()
            run = ProbeRun(s, cfg["chain_probe_seed"] + 100 * cycle, cfg["probe"]["max_rounds"], cfg["probe"]["estimator"])
            rounds = []
            while True:
                r = run.round()
                rounds.append(r)
                if (r["round"] >= rule.get("min_round", 1) and r.get(rule["key"], -1) >= rule["tau"]) or r["round"] >= min(cfg["probe"]["max_rounds"], args.max_rounds):
                    break
            info["probe_rounds"] = rounds
            info["rule_met"] = rounds[-1].get(rule["key"], -1) >= rule["tau"]
            summ = s.select(f"c{cycle}_{method}", run.score, weibull=False)
            key = f"c{cycle}_{method}_global_{sp}"
            run.free()
        else:
            raise ValueError(method)
    finally:
        cfg["sparsities"] = old
    info["select_s"] = summ["select_s"]
    packed = summ["masks"][key]
    s.d = []
    s.g = []
    torch.cuda.empty_cache()
    return packed, info


def phase_chains(ctx: Context, args) -> None:
    cfg = ctx.cfg
    out = ctx.work / "chains"
    out.mkdir(exist_ok=True)
    ref0, theta0, m1_0, m2_0, step0, _ = load_ckpt(ctx, verify_hash=not args.skip_hash)
    R = min(cfg["recovery_steps"], args.max_recovery)
    C = min(cfg["cycles"], args.max_cycles)
    methods = cfg["chain_methods"] if not args.only else [m for m in cfg["chain_methods"] if m in args.only.split(",")]
    for sp in cfg["sparsities"]:
        for method in methods:
            if method == "no_compression" and sp != cfg["sparsities"][0]:
                continue
            cid = f"{method}" if method == "no_compression" else f"{method}_{sp}"
            path = out / f"{cid}.json"
            if path.exists():
                continue
            t0 = now()
            ctx.load_params(theta0)
            opt = ctx.new_optimizer()
            opt.load_state(m1_0, m2_0, step0)
            ref = {n: ref0[n] for n in ctx.pipe.eligible_names}
            cycles = []
            for c in range(1, C + 1):
                crec: Dict[str, object] = {"cycle": c, "step": opt.step_count}
                if method != "no_compression":
                    opt.state_to(False)
                    torch.cuda.empty_cache()
                    tc = now()
                    packed, info = chain_mask(ctx, method, ref, out / f"{cid}_work", sp, c, args)
                    crec["score"] = info
                    crec["score_total_s"] = round(now() - tc, 2)
                    cur = ctx.snapshot(ctx.pipe.eligible_names)
                    stats = build_restored(ctx, ref, cur, packed)
                    del cur
                    crec["restore"] = stats
                    opt.state_to(True)
                crec["immediate"] = ctx.evaluate()
                # The restored checkpoint is the next cycle's reference.
                ref = ctx.snapshot(ctx.pipe.eligible_names)
                start = cfg["train"]["ckpt_step"] + (c - 1) * R
                crec["train_log"] = run_recovery(ctx, opt, start, R)
                crec["final"] = ctx.evaluate()
                cycles.append(crec)
                ctx.log("chain_cycle", id=cid, cycle=c, imm_val=crec["immediate"]["val"]["loss"], final_val=crec["final"]["val"]["loss"])
            write_json_new(path, {"id": cid, "method": method, "sparsity": sp, "recovery_steps": R, "cycles": cycles, "chain_s": round(now() - t0, 1), "provenance": ctx.prov, "gpu": ctx.gpu})
            opt.state_to(False)
            torch.cuda.empty_cache()


# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["prepare", "score", "branches", "chains"])
    ap.add_argument("--config", default=str(REPO / "experiments/configs/probe_coverage_32b.json"))
    ap.add_argument("--work-dir")
    ap.add_argument("--override", default="{}", help="JSON merged into the config (smoke tests only)")
    ap.add_argument("--skip-hash", action="store_true")
    ap.add_argument("--hvp-layer-limit", type=int, default=None, help="smoke tests only")
    ap.add_argument("--max-seeds", type=int, default=99)
    ap.add_argument("--max-rounds", type=int, default=99)
    ap.add_argument("--max-recovery", type=int, default=10**9)
    ap.add_argument("--max-cycles", type=int, default=99)
    ap.add_argument("--only", default="")
    ap.add_argument("--verify-archive", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v

    merge(cfg, json.loads(args.override))
    work = Path(args.work_dir or cfg["work_dir"])
    torch.manual_seed(0)
    ctx = Context(cfg, work)
    ctx.log("phase", phase=args.phase, args=vars(args))
    {"prepare": lambda: phase_prepare(ctx), "score": lambda: phase_score(ctx, args), "branches": lambda: phase_branches(ctx, args), "chains": lambda: phase_chains(ctx, args)}[args.phase]()
    ctx.log("done", phase=args.phase)


if __name__ == "__main__":
    main()
