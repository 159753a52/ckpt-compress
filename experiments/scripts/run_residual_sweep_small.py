"""Immediate-restoration residual-pruning sweep on a small causal LM (Pythia-410M).

Small-model counterpart of ``run_probe_sweep_32b.py``: for one residual regime
``d = theta_ckpt - theta_ref`` over the paper runner's eligible matrices, every
scorer ranks coordinates, the ``floor(p * N)`` lowest-scored coordinates are
restored to the reference (exact global k-smallest unless noted) and the model is
evaluated immediately on the runner's 100 held-out evaluation batches (no
recovery training).  Output JSON follows the 32B sweep schema
(``uncompressed``, ``evals[tag][str(p)]``, ``sparsities``, ``residual``).

Regimes (``--regime``):
  R1  checkpoint step 1000, reference = pretrained base weights
  R2  checkpoint step 3000, reference = pretrained base weights
  Q05_R1, P160_R1, G2_R1, P1B_R1  other models, step-1000 Alpaca checkpoint vs base
  R1W, G2W  Pythia-410M / GPT-2 fine-tuned on WikiText-2 chunks, checkpoint vs base
      (scoring pool from a held-out training-file tail, evaluation on the validation file)
  R3  checkpoint step 3000, reference = step-1000 checkpoint weights.  The two
      checkpoints come from two independent fine-tuning runs of the same base
      with the same hyper-parameters, so R3 is an inter-checkpoint residual,
      not the tail of one trajectory.

Scorers (library code of the paper runner unless noted):
  excp_style            |d|
  inshrinkerator_style  |g * theta|             compute_block_first_order_scores(theta)
  first_order           |g * d|                 compute_block_first_order_scores(d)
  hvp_global            |-g d + 0.5 d (H_bb d)| compute_block_taylor_scores (abs_mean)
  dacp                  same score, Weibull moment allocation (max_layer_ratio 0.999)
  probe_*               compute_probe_gn_scores (sample m1, topk m1, headtail m2, topk m4)
  first_order_ucb       |mean_b c_b| + std_b(c_b) / sqrt(B), c_b = -g_b d (this file)
  iterative_first_order stages over the sparsity list; g recomputed at the
                        partially restored point, pruned set monotone (this file,
                        mirrors run_iterative_rescoring_32b.py)

Batches mirror run_paper_experiments.py for the probe_pythia410m_campaign
manifest: ``set_seed(0)``, Alpaca loaders at batch 2 x seq 128, a cached pool of
140 training batches and 100 evaluation batches; the 8 scoring batches of a seed
are ``partition_repeated_seed_batches(pool, 100, 1, 8, seed).scoring_batches[0]``
(disjoint from evaluation).  Evaluation loss is token weighted.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Sequence, Tuple

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TensorDict = Dict[str, torch.Tensor]

SPARSITIES = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
MAX_LAYER_RATIO = 0.999
PROBES = (("probe_sample_m1", "sample", 1), ("probe_topk_m1", "topk", 1),
          ("probe_headtail_m2", "headtail", 2), ("probe_topk_m4", "topk", 4))
ALL_METHODS = ["excp_style", "inshrinkerator_style", "first_order", "hvp_global", "dacp",
               *[p[0] for p in PROBES], "first_order_ucb", "iterative_first_order"]
TINY = 1e-38
REGIMES = {
    "R1": {"checkpoint": "pythia_410m_alpaca/checkpoint_step_1000.pt", "reference": "base",
           "description": "step-1000 checkpoint minus pretrained base weights"},
    "R2": {"checkpoint": "pythia_410m_alpaca_step3000/checkpoint_step_3000.pt", "reference": "base",
           "description": "step-3000 checkpoint minus pretrained base weights"},
    "R3": {"checkpoint": "pythia_410m_alpaca_step3000/checkpoint_step_3000.pt",
           "reference": "pythia_410m_alpaca/checkpoint_step_1000.pt",
           "description": (
               "inter-checkpoint residual: step-3000 checkpoint minus step-1000 checkpoint; "
               "the two checkpoints are two independent fine-tuning runs from the same base "
               "with the same hyper-parameters (the 3000-step run is not a continuation of "
               "the 1000-step run)")},
    # breadth regimes on other models ("model" defaults to pythia-410m for the entries above)
    "P160_R1": {"model": "pythia-160m", "checkpoint": "pythia_160m_alpaca/checkpoint_step_1000.pt",
                "reference": "base",
                "description": "Pythia-160M step-1000 checkpoint minus pretrained base weights"},
    "G2_R1": {"model": "gpt2", "checkpoint": "gpt2_alpaca/checkpoint_step_1000.pt", "reference": "base",
              "description": "GPT-2 (124M) step-1000 checkpoint minus pretrained base weights"},
    "Q05_R1": {"model": "qwen2.5-0.5b", "checkpoint": "qwen2.5_0.5b_alpaca/checkpoint_step_1000.pt",
               "reference": "base",
               "description": "Qwen2.5-0.5B step-1000 Alpaca checkpoint minus pretrained base weights"},
    "P1B_R1": {"model": "pythia-1b", "checkpoint": "pythia_1b_alpaca/checkpoint_step_1000.pt",
               "reference": "base", "offload_states": True,
               "description": "Pythia-1B step-1000 Alpaca checkpoint minus pretrained base weights"},
    # second dataset: fine-tuned on WikiText-2 chunks (see experiments.lib.data.wikitext_chunk_splits)
    "R1W": {"model": "pythia-410m", "dataset": "wikitext2",
            "checkpoint": "pythia_410m_wikitext2/checkpoint_step_1000.pt", "reference": "base",
            "description": "Pythia-410M step-1000 WikiText-2 checkpoint minus pretrained base weights"},
    "G2W": {"model": "gpt2", "dataset": "wikitext2", "checkpoint": "gpt2_wikitext2/checkpoint_step_1000.pt",
            "reference": "base",
            "description": "GPT-2 (124M) step-1000 WikiText-2 checkpoint minus pretrained base weights"},
}
BASE_SOURCES = {"pythia-410m": "EleutherAI/pythia-410m", "pythia-160m": "EleutherAI/pythia-160m",
                "gpt2": "openai-community/gpt2", "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
                "pythia-1b": "EleutherAI/pythia-1b"}
def regime_model(regime: str) -> str:
    return REGIMES[regime].get("model", "pythia-410m")


def regime_dataset(regime: str) -> str:
    return REGIMES[regime].get("dataset", "alpaca")


def now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


# ---------------------------------------------------------------- mask selection

def _keys(scores: Mapping[str, torch.Tensor], names: Sequence[str], device: str) -> torch.Tensor:
    """Monotone int32 keys of non-negative float32 scores in names/flat order."""
    flat = torch.cat([scores[n].reshape(-1).to(device, torch.float32) for n in names])
    if not bool(torch.isfinite(flat).all()) or bool((flat < 0).any()):
        raise ValueError("scores must be finite and non-negative")
    flat = flat + 0.0  # map -0.0 to +0.0
    return flat.view(torch.int32)


def _kth_key(keys: torch.Tensor, k: int) -> int:
    """Value of the k-th smallest key (1-indexed) via two 16-bit histograms."""
    hi = keys >> 16
    cum = torch.bincount(hi, minlength=1 << 16).cumsum(0)
    b = int(torch.searchsorted(cum, torch.tensor([k], device=cum.device)).item())
    before = int(cum[b - 1].item()) if b > 0 else 0
    low = keys[hi == b] & 0xFFFF
    lc = torch.bincount(low, minlength=1 << 16).cumsum(0)
    lb = int(torch.searchsorted(lc, torch.tensor([k - before], device=lc.device)).item())
    return (b << 16) | lb


def global_keep_masks(
    scores: Mapping[str, torch.Tensor],
    names: Sequence[str],
    sparsities: Sequence[float],
    device: str,
) -> Iterator[Tuple[float, TensorDict, float]]:
    """Exact global k-smallest keep masks (same semantics as residual_masks.global_mask:
    ``floor(p N)`` coordinates removed, threshold ties broken in names/flat order)."""
    keys = _keys(scores, names, device)
    total = keys.numel()
    shapes = [scores[n].shape for n in names]
    sizes = [scores[n].numel() for n in names]
    for sp in sparsities:
        t0 = now()
        k = math.floor(sp * total)
        if k <= 0:
            remove = torch.zeros(total, dtype=torch.bool, device=keys.device)
        else:
            t = _kth_key(keys, k)
            remove = keys < t
            need = k - int(torch.count_nonzero(remove).item())
            if need:
                tied = torch.nonzero(keys == t, as_tuple=False).flatten()[:need]
                remove[tied] = True
            if int(torch.count_nonzero(remove).item()) != k:
                raise RuntimeError("global mask missed its prune budget")
        keep = {n: (~part).view(shape) for n, part, shape in zip(names, remove.split(sizes), shapes)}
        yield sp, keep, now() - t0
        del remove, keep
    del keys


# ---------------------------------------------------------------- experiment state

class Sweep:
    def __init__(self, model, family, layers, blocks, cur: TensorDict, ref: TensorDict,
                 eval_batches, device: str, offload: bool = False):
        from experiments.lib.residual_runtime import evaluate_lm

        self.evaluate_lm = evaluate_lm
        self.model, self.family, self.layers, self.blocks = model, family, layers, blocks
        self.names = [n for layer in layers for n in layer]
        self.params = dict(model.named_parameters())
        self.device = device
        self.eval_batches = eval_batches
        # offload: keep checkpoint/reference copies in pinned CPU memory and stream them per
        # tensor in apply() (identical values; lower device memory for larger models)
        self.offload = offload
        place = (lambda t: t.float().pin_memory()) if offload else (lambda t: t.to(device))
        self.cur_dev = {n: place(cur[n]) for n in self.names}
        self.ref_dev = {n: place(ref[n]) for n in self.names}
        self.delta = {n: (cur[n].float() - ref[n].float()) for n in self.names}  # CPU, as in runner
        if offload:
            self.delta = {n: d.pin_memory() for n, d in self.delta.items()}
        self.cur_cpu = {n: cur[n].float() for n in self.names}

    def delta_for_scoring(self) -> List[torch.Tensor]:
        """Residual tensors for the gradient-moment scorers: on the device, or pinned CPU
        tensors streamed per use (``dev``) when states are offloaded."""
        return [self.delta[n] if self.offload else self.delta[n].to(self.device) for n in self.names]

    def dev(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device, non_blocking=True)

    # model state
    @torch.no_grad()
    def apply(self, keep: Mapping[str, torch.Tensor] | None) -> None:
        for n in self.names:
            if keep is None:
                self.params[n].copy_(self.cur_dev[n])
            elif self.offload:
                torch.where(keep[n].to(self.device), self.cur_dev[n].to(self.device, non_blocking=True),
                            self.ref_dev[n].to(self.device, non_blocking=True), out=self.params[n].data)
            else:
                torch.where(keep[n].to(self.device), self.cur_dev[n], self.ref_dev[n],
                            out=self.params[n].data)

    def evaluate(self) -> Dict[str, float]:
        m = self.evaluate_lm(self.model, self.eval_batches, self.device)
        return {"val": m["loss"], "ppl": m["perplexity"], "tokens": m["tokens"], "eval_s": round(m["seconds"], 2)}

    def residual_stats(self) -> Dict[str, object]:
        n = sum(d.numel() for d in self.delta.values())
        z = sum(int((d == 0).sum()) for d in self.delta.values())
        l2 = math.sqrt(sum(float(d.double().square().sum()) for d in self.delta.values()))
        return {"eligible_coords": n, "zero_residual_fraction": z / n, "residual_l2": l2,
                "eligible_tensors": len(self.names), "layers": len(self.layers)}

    # evaluation of a score family over all sparsities
    def sweep_global(self, scores: Mapping[str, torch.Tensor], sparsities, extra=None) -> dict:
        out = {}
        for sp, keep, sel_s in global_keep_masks(scores, self.names, sparsities, self.device):
            self.apply(keep)
            ev = self.evaluate()
            self.apply(None)
            ev["select_s"] = round(sel_s, 3)
            if extra:
                ev.update(extra)
            out[str(sp)] = ev
        return out

    # per-batch full gradients over the eligible tensors at the current point
    def batch_gradients(self, batches, visit: Callable[[int, List[torch.Tensor]], None]) -> List[float]:
        from experiments.lib.residual_runtime import lm_loss

        model = self.model
        saved = {n: p.requires_grad for n, p in self.params.items()}
        eligible = set(self.names)
        for n, p in self.params.items():
            p.requires_grad_(n in eligible)
        params = [self.params[n] for n in self.names]
        losses = []
        was_training = model.training
        model.eval()
        try:
            for b, batch in enumerate(batches):
                with sdpa_kernel(SDPBackend.MATH):
                    loss = lm_loss(model, batch, self.device)
                    grads = torch.autograd.grad(loss, params)
                losses.append(float(loss.detach()))
                visit(b, [g.detach().float() for g in grads])
                del loss, grads
        finally:
            for n, p in self.params.items():
                p.requires_grad_(saved[n])
            model.train(was_training)
        return losses


def ucb_scores(s: Sweep, batches) -> Tuple[TensorDict, dict]:
    """|mean_b c_b| + std_b(c_b)/sqrt(B) with c_b = -g_b * d (unbiased std, ddof=1)."""
    B = len(batches)
    if B < 2:
        raise ValueError("first_order_ucb needs at least two scoring batches")
    d = s.delta_for_scoring()
    mean = [torch.zeros_like(x, device=s.device) for x in d]
    m2 = [torch.zeros_like(x, device=s.device) for x in d]

    def visit(b: int, grads: List[torch.Tensor]) -> None:
        for i, g in enumerate(grads):
            c = -g * s.dev(d[i])
            diff = c - mean[i]
            mean[i].add_(diff, alpha=1.0 / (b + 1))
            m2[i].add_(diff * (c - mean[i]))

    t0 = now()
    losses = s.batch_gradients(batches, visit)
    scores, mean_abs, std_term = {}, 0.0, 0.0
    for i, n in enumerate(s.names):
        std = (m2[i] / (B - 1)).clamp_min(0).sqrt() / math.sqrt(B)
        scores[n] = (mean[i].abs() + std).cpu()
        mean_abs += float(mean[i].abs().double().sum())
        std_term += float(std.double().sum())
    first_order_mean = {n: mean[i].abs().cpu() for i, n in enumerate(s.names)}
    del d, mean, m2
    torch.cuda.empty_cache()
    return scores, {"score_s": round(now() - t0, 2), "batches": B, "score_losses": losses,
                    "sum_abs_mean": mean_abs, "sum_std_over_sqrtB": std_term,
                    "_first_order_mean": first_order_mean}


def iterative_first_order(s: Sweep, batches, schedule: Sequence[float]) -> Tuple[dict, list]:
    out, stages = {}, []
    d = s.delta_for_scoring()
    keep_prev: TensorDict | None = None
    for sp in schedule:
        t0 = now()
        s.apply(keep_prev)
        g = [torch.zeros_like(x, device=s.device) for x in d]

        def visit(b: int, grads: List[torch.Tensor]) -> None:
            for i, gr in enumerate(grads):
                g[i].add_(gr, alpha=1.0 / len(batches))

        losses = s.batch_gradients(batches, visit)
        s.apply(None)
        grad_s = now() - t0
        scores = {}
        for i, n in enumerate(s.names):
            sc = (g[i] * s.dev(d[i])).abs() + TINY
            if keep_prev is not None:
                sc = torch.where(keep_prev[n], sc, torch.zeros_like(sc))
            scores[n] = sc
        del g
        (_, keep, sel_s), = list(global_keep_masks(scores, s.names, [sp], s.device))
        del scores
        if keep_prev is not None:
            for n in s.names:
                if bool((~keep_prev[n] & keep[n]).any()):
                    raise RuntimeError("iterative pruned set is not monotone")
        s.apply(keep)
        ev = s.evaluate()
        s.apply(None)
        ev.update({"select_s": round(sel_s, 3), "gradient_s": round(grad_s, 2),
                   "score_loss_at_partial_point": sum(losses) / len(losses)})
        out[str(sp)] = ev
        stages.append({"sparsity": sp, **ev})
        keep_prev = keep
        print(f"  iterative {sp}: dval {(ev['val']) :.5f}", flush=True)
    del d, keep_prev
    torch.cuda.empty_cache()
    return out, stages


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if torch.is_tensor(x):
        return _jsonable(x.tolist())
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if isinstance(x, (int, str, bool)) or x is None:
        return x
    if hasattr(x, "__dict__"):
        return _jsonable(vars(x))
    return str(x)


def run_seed(s: Sweep, seed: int, pool, sparsities, methods, uncompressed, check_mask: bool) -> dict:
    from experiments.lib.paper_runner import _build_global_mask
    from experiments.lib.residual_methods import build_weibull_mask
    from experiments.lib.residual_probe_scoring import compute_probe_gn_scores
    from experiments.lib.residual_runtime import batch_hash, set_seed
    from experiments.lib.residual_scoring import (
        compute_block_first_order_scores,
        compute_block_taylor_scores,
    )
    from experiments.lib.residual_training import partition_repeated_seed_batches

    set_seed(seed)
    plan = partition_repeated_seed_batches(pool, 100, 1, 8, seed)
    batches = plan.scoring_batches[0]
    s.apply(None)
    res: Dict[str, object] = {
        "seed": seed,
        "scoring_batches": {"count": len(batches), "sha256": batch_hash(batches),
                            "selected_pool_indices": list(plan.selected_pool_indices)},
        "uncompressed": uncompressed, "evals": {}, "timing": {}, "scoring_meta": {},
    }
    common = dict(task_type="lm", model_family=s.family, block_parameter_names=s.blocks)

    def record(tag: str, scores: TensorDict, score_s: float, meta: dict | None = None) -> None:
        t0 = now()
        res["evals"][tag] = s.sweep_global(scores, sparsities)
        res["timing"][tag] = {"score_s": round(score_s, 2), "select_eval_s": round(now() - t0, 2)}
        if meta is not None:
            res["scoring_meta"][tag] = _jsonable({k: v for k, v in meta.items() if not k.startswith("_")
                                                   and k not in ("checksum_before", "checksum_after", "layer_seconds")})
        dv = {k: round((v["val"] - uncompressed["val"]) * 1e3, 2) for k, v in res["evals"][tag].items()}
        print(f"[seed {seed}] {tag} score {score_s:.1f}s dval(1e-3) {dv}", flush=True)

    fo_scores = None
    if "excp_style" in methods:
        t0 = now()
        mag = {n: v.abs() for n, v in s.delta.items()}
        sc_s = now() - t0
        if check_mask:
            # verify the GPU selector against the runner's exact global mask builder
            sp = sparsities[len(sparsities) // 2]
            ref_masks, _ = _build_global_mask(s.layers, mag, sp)
            (_, keep, _), = list(global_keep_masks(mag, s.names, [sp], s.device))
            mism = sum(int((ref_masks[n] != keep[n].cpu()).sum()) for n in s.names)
            res["mask_check"] = {"sparsity": sp, "mismatched_coords": mism}
            print(f"mask check at {sp}: {mism} mismatches", flush=True)
            if mism:
                raise RuntimeError("GPU global mask differs from the runner's global mask")
        record("excp_style", mag, sc_s)
        del mag
    if "inshrinkerator_style" in methods:
        t0 = now()
        sc, meta = compute_block_first_order_scores(s.model, batches, s.layers, s.cur_cpu, s.device, **common)
        record("inshrinkerator_style", sc, now() - t0, meta)
        del sc
    if "first_order" in methods or "first_order_ucb" in methods:
        t0 = now()
        fo_scores, meta = compute_block_first_order_scores(s.model, batches, s.layers, s.delta, s.device, **common)
        if "first_order" in methods:
            record("first_order", fo_scores, now() - t0, meta)
    if "hvp_global" in methods or "dacp" in methods:
        t0 = now()
        sc, meta = compute_block_taylor_scores(s.model, batches, s.layers, s.delta, s.device,
                                               aggregation="abs_mean", **common)
        hvp_s = now() - t0
        if "hvp_global" in methods:
            record("hvp_global", sc, hvp_s, meta)
        if "dacp" in methods:
            t_all = now()
            res["evals"]["dacp"], res["weibull"] = {}, {}
            for sp in sparsities:
                t1 = now()
                masks, alloc = build_weibull_mask(s.layers, sc, sp, max_layer_ratio=MAX_LAYER_RATIO)
                sel = now() - t1
                s.apply(masks)
                ev = s.evaluate()
                s.apply(None)
                pruned = sum(int((~m).sum()) for m in masks.values())
                ev.update({"select_s": round(sel, 3), "pruned": pruned, "target_pruned": alloc["target_pruned"]})
                res["evals"]["dacp"][str(sp)] = ev
                res["weibull"][str(sp)] = _jsonable({"layer_counts": alloc["weibull_layer_counts"],
                                                     "meta": alloc["weibull"]})
                del masks
            res["timing"]["dacp"] = {"score_s": round(hvp_s, 2), "select_eval_s": round(now() - t_all, 2),
                                     "max_layer_ratio": MAX_LAYER_RATIO}
            dv = {k: round((v["val"] - uncompressed["val"]) * 1e3, 2) for k, v in res["evals"]["dacp"].items()}
            print(f"[seed {seed}] dacp dval(1e-3) {dv}", flush=True)
        del sc
    for tag, est, rounds in PROBES:
        if tag not in methods:
            continue
        t0 = now()
        sc, meta = compute_probe_gn_scores(s.model, batches, s.layers, s.delta, s.device,
                                           estimator=est, rounds=rounds, seed=seed, **common)
        record(tag, sc, now() - t0, meta)
        del sc
        torch.cuda.empty_cache()
    if "first_order_ucb" in methods:
        sc, meta = ucb_scores(s, batches)
        fom = meta.pop("_first_order_mean")
        if fo_scores is not None:
            num = sum(float((fom[n] - fo_scores[n]).abs().double().sum()) for n in s.names)
            den = sum(float(fo_scores[n].double().sum()) for n in s.names)
            meta["rel_l1_mean_vs_block_first_order"] = num / max(den, 1e-300)
        del fom
        record("first_order_ucb", sc, meta["score_s"], meta)
        del sc
    del fo_scores
    if "iterative_first_order" in methods:
        t0 = now()
        res["evals"]["iterative_first_order"], res["iterative_stages"] = iterative_first_order(s, batches, sparsities)
        res["timing"]["iterative_first_order"] = {"total_s": round(now() - t0, 2)}
    s.apply(None)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=sorted(REGIMES), required=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--sparsities", default=",".join(str(x) for x in SPARSITIES))
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--prefix", default="", help="output file name prefix, e.g. tol- -> tol-R1_seed42.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--check-mask", action="store_true")
    args = ap.parse_args()

    from experiments.lib.data import cache_batches, get_data_loaders
    from experiments.lib.models import load_model
    from experiments.lib.residual_runtime import (
        batch_hash,
        configure_hf_offline,
        load_training_checkpoint,
        set_seed,
        sha256_file,
        write_json,
    )
    from experiments.lib.residual_scoring import eligible_layers, transformer_layers

    methods = [m for m in args.methods.split(",") if m]
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"unknown methods {sorted(unknown)}")
    sparsities = [float(x) for x in args.sparsities.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    spec = REGIMES[args.regime]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_regime(args.regime, args.checkpoint_root, args.data_root, args.device)
    s, pool, eval_batches, family = ctx["sweep"], ctx["pool"], ctx["eval_batches"], ctx["family"]
    unc, full_restore, residual, prov = ctx["uncompressed"], ctx["full_restore"], ctx["residual"], ctx["provenance"]
    setup_s = ctx["setup_seconds"]

    for seed in seeds:
        path = args.out_dir / f"{args.prefix}{args.regime}_seed{seed}{args.tag}.json"
        if path.exists():
            print(f"skip existing {path}", flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        res = run_seed(s, seed, pool, sparsities, methods, unc, args.check_mask)
        res.update({
            "regime": args.regime,
            "regime_description": spec["description"],
            "model": regime_model(args.regime),
            "model_family": family,
            "dataset": regime_dataset(args.regime),
            "batch_size": 2,
            "seq_length": 128,
            "eval_batches": {"count": len(eval_batches), "sha256": batch_hash(eval_batches)},
            "training_pool": {"count": len(pool), "sha256": batch_hash(pool)},
            "sparsities": sparsities,
            "methods": methods,
            "residual": residual,
            "full_restore": full_restore,
            "provenance": prov,
            "protocol": "immediate restoration (no recovery training); exact global k-smallest "
                        "keep masks unless noted; dacp uses Weibull moment allocation",
            "seconds": round(time.time() - t0, 1),
            "setup_seconds": round(setup_s, 1),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        })
        write_json(path, _jsonable(res))
        print(f"wrote {path} ({res['seconds']} s, peak {res['peak_gib']} GiB)", flush=True)


def load_regime(regime: str, checkpoint_root: Path, data_root: Path, device: str) -> dict:
    """Load data batches, model, checkpoint and reference for one regime; build the Sweep."""
    from experiments.lib.data import cache_batches, get_data_loaders
    from experiments.lib.models import load_model
    from experiments.lib.residual_runtime import (
        configure_hf_offline,
        load_training_checkpoint,
        set_seed,
        sha256_file,
    )
    from experiments.lib.residual_scoring import eligible_layers, transformer_layers

    spec = REGIMES[regime]
    t_start = time.time()
    configure_hf_offline()

    set_seed(0)
    model_name = regime_model(regime)
    data_meta = None
    if regime_dataset(regime) == "wikitext2":
        # scoring pool from the held-out tail of the training file (never seen in fine-tuning),
        # evaluation from the validation file in file order
        from experiments.lib.data import wikitext_chunk_loaders

        train_loader, val_loader, data_meta = wikitext_chunk_loaders(model_name, 2, 128, str(data_root),
                                                                     train_split="scoring")
        task_type = "lm"
    else:
        train_loader, val_loader, task_type = get_data_loaders(model_name, "alpaca", 2, 128, data_dir=str(data_root))
    pool = cache_batches(train_loader, 140, task_type)
    eval_batches = cache_batches(val_loader, 100, task_type)
    del train_loader, val_loader
    if len(pool) != 140 or len(eval_batches) != 100:
        raise RuntimeError("data loaders returned too few batches")

    model, family = load_model(model_name, pretrained=False, device="cpu", dataset_name="alpaca")
    layers = eligible_layers(model, family)
    blocks = transformer_layers(model, family)
    names = [n for layer in layers for n in layer]

    ckpt_path = checkpoint_root / spec["checkpoint"]
    ckpt = load_training_checkpoint(ckpt_path)
    model.load_state_dict(ckpt.model_state, strict=True)
    cur = {n: ckpt.model_state[n].detach().float().clone() for n in names}
    prov = {"checkpoint": str(ckpt_path), "checkpoint_step": ckpt.step, "checkpoint_sha256": sha256_file(ckpt_path)}
    if data_meta is not None:
        prov["data"] = {"dataset": "wikitext2", "pool_split": "scoring (held-out training-file tail)",
                        "eval_split": "validation file", **data_meta}
    offload = bool(spec.get("offload_states", False))
    if offload:
        prov["offload_states"] = True
    del ckpt
    if spec["reference"] == "base":
        base, _ = load_model(model_name, pretrained=True, device="cpu", dataset_name="alpaca")
        bsd = base.state_dict()
        ref = {n: bsd[n].detach().float().clone() for n in names}
        from dacp.utils.paths import resolve_model_source
        prov["reference"] = {"kind": "pretrained_base", "source": str(resolve_model_source(model_name, BASE_SOURCES[model_name]))}
        del base, bsd
    else:
        rpath = checkpoint_root / spec["reference"]
        rck = load_training_checkpoint(rpath)
        ref = {n: rck.model_state[n].detach().float().clone() for n in names}
        prov["reference"] = {"kind": "checkpoint", "checkpoint": str(rpath), "checkpoint_step": rck.step,
                             "checkpoint_sha256": sha256_file(rpath)}
        del rck
    model.to(device)
    s = Sweep(model, family, layers, blocks, cur, ref, eval_batches, device, offload=offload)
    del cur, ref
    residual = s.residual_stats()
    s.apply(None)
    unc = s.evaluate()
    none_keep = {n: torch.zeros_like(s.delta[n], dtype=torch.bool) for n in names}
    s.apply(none_keep)
    full_restore = s.evaluate()
    s.apply(None)
    del none_keep
    print(f"regime {regime}: residual {residual}; uncompressed {unc}; full restore {full_restore}", flush=True)
    return {"sweep": s, "pool": pool, "eval_batches": eval_batches, "family": family,
            "uncompressed": unc, "full_restore": full_restore, "residual": residual,
            "provenance": prov, "setup_seconds": time.time() - t_start}


if __name__ == "__main__":
    main()
