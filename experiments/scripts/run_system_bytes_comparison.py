"""System-level checkpoint compression comparison on one training checkpoint.

For a {model_state_dict, optimizer_state_dict, step} checkpoint fine-tuned from
a pretrained base, every method encodes the weight residual (or, for the
Inshrinkerator-like method, the full weights) together with both Adam moments
(exp_avg, exp_avg_sq) into serialized bytes, decodes them, and the restored
weights are evaluated on the exact cached evaluation batches used by
run_paper_experiments.py. Each component is serialized with torch.save and
additionally measured after zlib (level 6) and lzma (preset 1) compression.

Methods and fidelity (none of these is a full reproduction):

* excp_pipeline -- reimplementation of the ExCP pipeline following the
  authors' released script (baselines/excp/upstream/compress_pythia.py):
  residual against the base, per-tensor threshold
  median(|dW|) * min(2.5, alpha / sqrt(mean(exp_avg_sq))), joint momentum rule
  (keep both moments where |exp_avg| > beta * mean|exp_avg| and the residual
  survived), 4-bit non-uniform K-means on non-zeros with index 0 reserved for
  zero, two indices per byte and fp16 codebooks. faiss K-means is replaced by
  a seeded 1-D Lloyd K-means on a faiss-sized subsample (256 points/centroid).
* dacp_excpq / probe_topk_m1_excpq -- identical to excp_pipeline except that,
  on the eligible Transformer matrices, the surviving residual entries are
  chosen by the repository's dacp / probe_topk_m1_exact_global scoring at the
  same number of pruned eligible entries as excp_pipeline at that alpha.
  Moments then follow the same ExCP joint rule and the same quantizer.
* iterative_first_order_excpq / iterative_ucb_excpq / first_order_excpq -- same
  as dacp_excpq but the eligible entries are chosen by exact global iterative
  re-linearized pruning (run_residual_ablation_small.run_iterative; stages over
  0.5,0.8,0.9,0.95,0.98,0.99,0.995 below the target, or geometric with a _K<k>
  suffix) or one-shot |g d| (first_order).
* --extra-sparsities: our selections at eligible sparsities beyond ExCP's cap
  (the ExCP factor is capped at 2.5, so ExCP itself cannot reach them); the
  non-eligible tensors and the moment rule follow ExCP at --extra-alpha, and the
  records are labelled operating_point.eligible_sparsity / fidelity
  "ours_beyond_excp_sparsity_cap_...".
* inshrinkerator_like -- per-tensor partition of the full weights and of both
  moments with the repository Inshrinkerator partition() (protect the top
  protect_fraction by magnitude or |w * grad| sensitivity in bf16, prune the
  bottom prune_fraction by magnitude, quantize the rest), non-uniform
  quantization of magnitudes into n_bins centers plus a sign bit. Our
  simplifications: the sketch-based approximate K-means is replaced by the
  same sampled Lloyd K-means, moments use magnitude-only protection, and no
  cross-checkpoint index delta / RLE stage exists for a single checkpoint
  (zlib/lzma stand in for the final entropy stage). Codes are stored as uint8.

Usage (from the repository root):
  python experiments/scripts/run_system_bytes_comparison.py \
    --manifest experiments/configs/probe_pythia410m_campaign.yaml \
    --workload pythia_410m_alpaca --checkpoint-root C --data-root D \
    --seeds 42 --alphas 5e-5,2e-4,1e-3,1e-2 --output out.json
"""

from __future__ import annotations

import argparse
import io
import json
import lzma
import math
import re
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

import run_paper_experiments as rpe  # noqa: E402
from baselines.inshrinkerator.partition import PartitionConfig, partition  # noqa: E402
from experiments.lib.losses import causal_lm_loss, move_batch_to_device  # noqa: E402
from experiments.lib.paper_runner import _score_and_mask  # noqa: E402
from experiments.lib.residual_runtime import evaluate_task  # noqa: E402
from experiments.lib.residual_scoring import eligible_layers, transformer_layers  # noqa: E402
from experiments.lib.residual_training import partition_repeated_seed_batches  # noqa: E402

SCHEMA_VERSION = 1
EXCP_BITS = 4


# ----------------------------------------------------------------- quantizer
def kmeans_1d(values: torch.Tensor, k: int, seed: int, iters: int = 25) -> torch.Tensor:
    """Seeded Lloyd K-means on a 1-D sample (faiss-like 256 points/centroid)."""
    generator = torch.Generator().manual_seed(seed)
    count = values.numel()
    limit = 256 * k
    if count > limit:
        index = torch.randperm(count, generator=generator)[:limit].to(values.device)
        train = values[index].float()
    else:
        train = values.float()
    init = torch.randperm(train.numel(), generator=generator)[:k].to(train.device)
    centers = train[init].clone()
    for _ in range(iters):
        centers, _ = torch.sort(centers)
        labels = assign(train, centers)
        sums = torch.zeros_like(centers).index_add_(0, labels, train)
        counts = torch.zeros_like(centers).index_add_(0, labels, torch.ones_like(train))
        nonempty = counts > 0
        centers = torch.where(nonempty, sums / counts.clamp_min(1), centers)
    return torch.sort(centers).values


def assign(values: torch.Tensor, sorted_centers: torch.Tensor) -> torch.Tensor:
    """Nearest-center index for 1-D sorted centers."""
    if sorted_centers.numel() == 1:
        return torch.zeros(values.numel(), dtype=torch.long, device=values.device)
    midpoints = (sorted_centers[1:] + sorted_centers[:-1]) / 2
    return torch.bucketize(values.float().contiguous(), midpoints.contiguous())


def excp_quant(tensor: torch.Tensor, seed: int, bits: int = EXCP_BITS, codebook_dtype=torch.float16) -> dict:
    """Nonzero K-means codes with zero reserved; 4-bit packing or 8-bit indices."""
    if bits not in (4, 8):
        raise ValueError("ExCP quantization supports 4 or 8 bits")
    flat = tensor.flatten().float()
    nonzero = flat != 0
    nnz = int(nonzero.sum().item())
    if nnz <= 2**bits - 1:
        return {"raw": tensor.detach().float().cpu()}
    centers = kmeans_1d(flat[nonzero], 2**bits - 1, seed)
    if bits == 8:
        codes = torch.zeros(flat.numel(), dtype=torch.uint8, device=flat.device)
        codes[nonzero] = (assign(flat[nonzero], centers) + 1).to(torch.uint8)
        return {"codes": codes.cpu(), "codebook": centers.to(codebook_dtype).cpu(), "numel": flat.numel()}
    codes = torch.zeros(flat.numel() + flat.numel() % 2, dtype=torch.uint8, device=flat.device)
    codes[: flat.numel()][nonzero] = (assign(flat[nonzero], centers) + 1).to(torch.uint8)
    packed = codes[0::2] * 16 + codes[1::2]
    return {"packed": packed.cpu(), "codebook": centers.to(codebook_dtype).cpu(), "numel": flat.numel()}


def excp_dequant(payload: dict, shape: torch.Size, device: str) -> torch.Tensor:
    if "raw" in payload:
        return payload["raw"].to(device).view(shape)
    if "codes" in payload:
        codes = payload["codes"].to(device)
    else:
        packed = payload["packed"].to(device)
        codes = torch.stack([packed // 16, packed % 16], dim=1).flatten()[: payload["numel"]]
    table = torch.cat([torch.zeros(1, device=device), payload["codebook"].to(device).float()])
    return table[codes.long()].view(shape)


def insh_encode(values: torch.Tensor, grad: torch.Tensor, config: dict, seed: int) -> dict:
    """Inshrinkerator-like partition + sign/magnitude non-uniform quantization."""
    n_bins = config["n_bins"]
    part = partition(
        values.float(),
        grad.float(),
        PartitionConfig(
            protect_fraction=config["protect_fraction"],
            prune_fraction=config["prune_fraction"],
            n_bins=n_bins,
            use_sensitivity=bool(grad.abs().max().item() > 0),
        ),
    )
    flat = values.flatten().float()
    protect = part.protect_mask.flatten().bool()
    quantize = part.quantize_mask.flatten().bool()
    codes = torch.zeros(flat.numel(), dtype=torch.uint8, device=flat.device)
    centers = torch.zeros(n_bins, device=flat.device)
    if quantize.any():
        magnitude = flat[quantize].abs()
        centers = kmeans_1d(magnitude, min(n_bins, magnitude.numel()), seed)
        codes[quantize] = (assign(magnitude, centers) + 1).to(torch.uint8)
    codes[protect] = n_bins + 1
    signs = torch.from_numpy(np.packbits((flat[quantize] < 0).cpu().numpy()))
    return {
        "codes": codes.cpu(),
        "signs": signs,
        "centers": centers.half().cpu(),
        "protected": flat[protect].to(torch.bfloat16).cpu(),
        "n_bins": n_bins,
    }


def insh_decode(payload: dict, shape: torch.Size, device: str) -> torch.Tensor:
    codes = payload["codes"].to(device).long()
    n_bins = payload["n_bins"]
    out = torch.zeros(codes.numel(), device=device)
    quantize = (codes >= 1) & (codes <= n_bins)
    count = int(quantize.sum().item())
    if count:
        centers = payload["centers"].to(device).float()
        signs = np.unpackbits(payload["signs"].numpy())[:count].astype(bool)
        sign = 1.0 - 2.0 * torch.from_numpy(signs).to(device).float()
        out[quantize] = centers[codes[quantize] - 1] * sign
    protect = codes == n_bins + 1
    out[protect] = payload["protected"].to(device).float()
    return out.view(shape)


# ------------------------------------------------------------ serialization
def measure_bytes(payload: object, lzma_enabled: bool) -> dict:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    result = {"torch_save": len(raw), "zlib6": len(zlib.compress(raw, 6))}
    if lzma_enabled:
        result["lzma1"] = len(lzma.compress(raw, preset=1))
    return result


def rel_error(restored: dict, reference: dict) -> float:
    num = sum(float((restored[k] - reference[k].float()).pow(2).sum()) for k in reference)
    den = sum(float(reference[k].float().pow(2).sum()) for k in reference)
    return math.sqrt(num / den) if den > 0 else 0.0


# ------------------------------------------------------------------ helpers
def named_moments(model: torch.nn.Module, optimizer_state: dict) -> tuple[dict, dict]:
    names = [name for name, _ in model.named_parameters()]
    state = optimizer_state["state"]
    if sorted(state) != list(range(len(names))):
        raise ValueError("optimizer state indices do not match model parameters")
    exp_avg, exp_avg_sq = {}, {}
    params = dict(model.named_parameters())
    for index, name in enumerate(names):
        entry = state[index]
        if entry["exp_avg"].shape != params[name].shape:
            raise ValueError(f"moment shape mismatch for {name}")
        exp_avg[name] = entry["exp_avg"].float()
        exp_avg_sq[name] = entry["exp_avg_sq"].float()
    return exp_avg, exp_avg_sq


def gradients(model, batches, device) -> dict:
    model.zero_grad(set_to_none=True)
    model.train()
    for batch in batches:
        loss, _ = causal_lm_loss(model, move_batch_to_device(batch, device))
        (loss / len(batches)).backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    model.eval()
    return grads


def excp_weight_masks(residual, exp_avg_sq, alpha, device) -> dict:
    masks = {}
    for name, value in residual.items():
        r = value.to(device)
        factor = min(2.5, float(alpha / exp_avg_sq[name].to(device).mean().sqrt()))
        masks[name] = (r.abs() > r.abs().median() * factor).cpu()
    return masks


def encode_residual_method(residual, exp_avg, exp_avg_sq, masks, beta, seed, device, lzma_on):
    """Shared ExCP encode/decode given the weight keep masks."""
    enc_w, enc_m, enc_v = {}, {}, {}
    for index, name in enumerate(residual):
        keep = masks[name].to(device)
        r = residual[name].to(device) * keep
        m = exp_avg[name].to(device)
        v = exp_avg_sq[name].to(device)
        keep_m = (m.abs() > beta * m.abs().mean()) & (r != 0)
        enc_w[name] = excp_quant(r, seed * 100003 + 3 * index)
        enc_m[name] = excp_quant(m * keep_m, seed * 100003 + 3 * index + 1)
        enc_v[name] = excp_quant(v * keep_m, seed * 100003 + 3 * index + 2)
    sizes = {
        "weights": measure_bytes(enc_w, lzma_on),
        "exp_avg": measure_bytes(enc_m, lzma_on),
        "exp_avg_sq": measure_bytes(enc_v, lzma_on),
    }
    dec_w = {n: excp_dequant(enc_w[n], residual[n].shape, device).cpu() for n in residual}
    dec_m = {n: excp_dequant(enc_m[n], residual[n].shape, device).cpu() for n in residual}
    dec_v = {n: excp_dequant(enc_v[n], residual[n].shape, device).cpu() for n in residual}
    kept = sum(int((dec_w[n] != 0).sum()) for n in residual)
    kept_m = sum(int((dec_m[n] != 0).sum()) for n in residual)
    return sizes, dec_w, dec_m, dec_v, {"kept_residual_nonzeros": kept, "kept_moment_nonzeros": kept_m}


def total_sizes(sizes: dict) -> dict:
    keys = sizes["weights"].keys()
    return {k: sum(part[k] for part in sizes.values()) for k in keys}


LOCAL_OURS = re.compile(r"(iterative_first_order|iterative_ucb|first_order)(?:_K(\d+))?")


def local_selection(name: str, sweep, batches, target: float, seed: int) -> tuple[dict, dict]:
    """Keep masks on the eligible matrices from run_residual_ablation_small.run_iterative.

    iterative_first_order / iterative_ucb: stages over the sweep sparsity list below the
    target, then the target (``_K<k>``: geometric schedule P_j = 1 - (1-P)^(j/k)); gradients
    recomputed at the partially restored point; first_order: one stage (one-shot |g d| with
    the averaged full gradient of the scoring batches).
    """
    import run_residual_ablation_small as abl

    kind, k = LOCAL_OURS.fullmatch(name).groups()
    scorer = abl.score_ucb if kind == "iterative_ucb" else abl.score_first_order
    if kind == "first_order":
        schedule = [target]
    elif k:
        schedule = abl.geometric_schedule(target, int(k))
    else:
        schedule = abl.list_schedule(target)
    _, stages, keep = abl.run_iterative(sweep, batches, schedule, scorer, seed, eval_at=[], return_keep=True)
    return {n: keep[n].cpu() for n in sweep.names}, {"schedule": schedule, "stages": stages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--alphas", default="5e-5,2e-4,1e-3,1e-2")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--insh-bins", default="4,16,64")
    parser.add_argument("--insh-protect", type=float, default=0.005)
    parser.add_argument("--insh-prune", type=float, default=0.2)
    parser.add_argument("--ours", default="dacp,probe_topk_m1_exact_global")
    parser.add_argument("--no-lzma", action="store_true")
    parser.add_argument("--skip-excp-record", action="store_true",
                        help="compute ExCP masks (for the eligible sparsity) but do not encode/emit excp_pipeline")
    parser.add_argument("--extra-sparsities", default="",
                        help="eligible sparsities beyond ExCP's cap for --extra-ours (non-eligible tensors "
                             "and the moment rule follow ExCP at --extra-alpha)")
    parser.add_argument("--extra-alpha", type=float, default=1e-2)
    parser.add_argument("--extra-ours", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lzma_on = not args.no_lzma
    device = args.device

    rpe.configure_hf_offline()
    manifest = rpe.load_paper_manifest(args.manifest)
    provenance = rpe._source_provenance(manifest.path)
    workload = next(w for w in manifest.workloads if w.name == args.workload)
    contracts = {c.name: c for c in manifest.methods}
    context = rpe._load_workload_context(
        workload,
        checkpoint_root=args.checkpoint_root,
        data_root=args.data_root,
        declared_checkpoint_root=rpe.ROOT / "checkpoints",
        declared_data_root=rpe.ROOT / "data",
        expected_data_identity=None,
    )
    from transformers import GPTNeoXForCausalLM

    from dacp.utils.paths import resolve_model_source

    base_source = resolve_model_source("pythia-410m", "EleutherAI/pythia-410m")
    base_model = GPTNeoXForCausalLM.from_pretrained(base_source, torch_dtype=torch.float32)
    base_params = {n: p.detach().clone() for n, p in base_model.named_parameters()}
    del base_model

    from experiments.lib.models import load_model

    model = load_model(workload.model, pretrained=False, device="cpu", dataset_name=workload.dataset)[0]
    model.load_state_dict(context.checkpoint.model_state, strict=True)
    model.to(device)
    model.eval()
    param_names = [n for n, _ in model.named_parameters()]
    if set(param_names) != set(base_params):
        raise ValueError("base and checkpoint parameter names differ")
    current = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()}
    residual = {n: current[n] - base_params[n].float() for n in param_names}
    exp_avg, exp_avg_sq = named_moments(model, context.checkpoint.optimizer_state)
    layers = eligible_layers(model, workload.model_family)
    blocks = transformer_layers(model, workload.model_family)
    eligible = [name for layer in layers for name in layer]
    eligible_count = sum(residual[n].numel() for n in eligible)
    total_params = sum(residual[n].numel() for n in param_names)

    def evaluate_weights(weights: dict) -> dict:
        with torch.no_grad():
            params = dict(model.named_parameters())
            for name, value in weights.items():
                params[name].copy_(value.to(device))
        metrics = evaluate_task(model, context.eval_batches, workload.task_type, device)
        with torch.no_grad():
            for name, value in current.items():
                params[name].copy_(value.to(device))
        return metrics

    started = time.perf_counter()
    reference = {
        "checkpoint": evaluate_task(model, context.eval_batches, workload.task_type, device),
        "base": evaluate_weights({n: base_params[n].float() for n in param_names}),
        "fp32_bytes": {
            "weights": 4 * total_params,
            "exp_avg": 4 * total_params,
            "exp_avg_sq": 4 * total_params,
        },
        "fp32_serialized_residual_and_moments": total_sizes(
            {
                "weights": {"torch_save": 4 * total_params},
                "exp_avg": {"torch_save": 4 * total_params},
                "exp_avg_sq": {"torch_save": 4 * total_params},
            }
        ),
        "parameters": total_params,
        "eligible_parameters": eligible_count,
    }
    print(json.dumps(reference, indent=1), flush=True)
    records = []

    def emit(record: dict) -> None:
        records.append(record)
        print(
            f"{record['method']:22s} seed={record['seed']} op={record['operating_point']} "
            f"ppl={record['restored']['perplexity']:.4f} bytes_zlib={record['total_bytes']['zlib6']}",
            flush=True,
        )
        write()

    def write() -> None:
        args.output.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "script": "experiments/scripts/run_system_bytes_comparison.py",
                    "argv": sys.argv[1:],
                    "provenance": {
                        **provenance,
                        "checkpoint": str(context.checkpoint_path),
                        "checkpoint_sha256": context.checkpoint_sha256,
                        "checkpoint_step": context.checkpoint.step,
                        "base_model": str(base_source),
                    },
                    "data_identity": context.data_identity,
                    "reference": reference,
                    "records": records,
                    "elapsed_seconds": time.perf_counter() - started,
                    "written_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=1,
                default=str,
            ),
            encoding="utf-8",
        )

    seeds = [int(s) for s in args.seeds.split(",")]
    alphas = [float(a) for a in args.alphas.split(",") if a]
    ours = [o for o in args.ours.split(",") if o]
    extra_ours = [o for o in args.extra_ours.split(",") if o]
    extra_sparsities = [float(x) for x in args.extra_sparsities.split(",") if x]
    sweep = None
    if any(LOCAL_OURS.fullmatch(o) for o in ours + extra_ours):
        from run_residual_sweep_small import Sweep

        sweep = Sweep(model, workload.model_family, layers, blocks,
                      {n: current[n] for n in eligible}, {n: base_params[n].float() for n in eligible},
                      context.eval_batches, device)

    def select(our: str, scoring_batches, target: float, seed: int) -> tuple[dict, dict]:
        t_score = time.perf_counter()
        if LOCAL_OURS.fullmatch(our):
            masks, info = local_selection(our, sweep, scoring_batches, target, seed)
            meta = {"internal_method": our, "allocation_kind": "exact_global_iterative"
                    if our.startswith("iterative") else "exact_global", **info}
        else:
            delta = {n: residual[n].to(device) for n in eligible}
            masks, compression = _score_and_mask(
                SimpleNamespace(internal_method=contracts[our].internal_method),
                model,
                scoring_batches,
                layers,
                blocks,
                delta,
                target,
                workload.max_layer_ratio,
                workload.task_type,
                workload.model_family,
                device,
                False,
                seed=seed,
            )
            del delta
            meta = {"internal_method": contracts[our].internal_method,
                    "allocation_kind": compression.get("allocation_kind")}
        meta["scoring_seconds"] = time.perf_counter() - t_score
        return {n: masks[n].cpu() for n in eligible}, meta

    points = [(a, None) for a in alphas] + [(args.extra_alpha, sp) for sp in extra_sparsities]
    for seed in seeds:
        plan = partition_repeated_seed_batches(
            context.training_pool, workload.total_steps, 1, workload.hvp_batches, seed
        )
        scoring_batches = plan.scoring_batches[0]
        for alpha, override in points:
            t0 = time.perf_counter()
            excp_masks = excp_weight_masks(residual, exp_avg_sq, alpha, device)
            pruned_eligible = sum(int((~excp_masks[n]).sum()) for n in eligible)
            eligible_sparsity = pruned_eligible / eligible_count
            variants = []
            if override is None and not args.skip_excp_record:
                variants.append(("excp_pipeline", excp_masks, None))
            target = eligible_sparsity if override is None else override
            for our in (ours if override is None else extra_ours):
                eligible_masks, meta = select(our, scoring_batches, target, seed)
                merged = dict(excp_masks)
                merged.update(eligible_masks)
                if override is not None:
                    meta["beyond_excp_cap"] = True
                variants.append((f"{our}_excpq", merged, meta))
            for name, masks, meta in variants:
                t1 = time.perf_counter()
                sizes, dec_w, dec_m, dec_v, counts = encode_residual_method(
                    residual, exp_avg, exp_avg_sq, masks, args.beta, seed, device, lzma_on
                )
                restored = {n: base_params[n].float() + dec_w[n] for n in param_names}
                emit(
                    {
                        "method": name,
                        "fidelity": (
                            "reimplementation_of_upstream_excp_script"
                            if name == "excp_pipeline"
                            else "ours_with_excp_moment_rule_and_quantizer"
                            if override is None
                            else "ours_beyond_excp_sparsity_cap_with_excp_moment_rule_and_quantizer"
                        ),
                        "seed": seed,
                        "operating_point": {"alpha": alpha, "beta": args.beta, "bits": EXCP_BITS,
                                            **({} if override is None else {"eligible_sparsity": override})},
                        "excp_eligible_sparsity": eligible_sparsity,
                        "eligible_sparsity": target if name != "excp_pipeline" else eligible_sparsity,
                        "eligible_kept": sum(int(masks[n].sum()) for n in eligible),
                        "all_kept_mask": sum(int(masks[n].sum()) for n in param_names),
                        **counts,
                        "mask_method": meta,
                        "bytes": sizes,
                        "total_bytes": total_sizes(sizes),
                        "restored": evaluate_weights(restored),
                        "relative_l2_error": {
                            "residual": rel_error(dec_w, residual),
                            "exp_avg": rel_error(dec_m, exp_avg),
                            "exp_avg_sq": rel_error(dec_v, exp_avg_sq),
                        },
                        "seconds": time.perf_counter() - t1,
                    }
                )
                del dec_w, dec_m, dec_v, restored
            print(f"alpha {alpha} done in {time.perf_counter() - t0:.0f}s", flush=True)

        grads = gradients(model, scoring_batches, device)
        for n_bins in [int(b) for b in args.insh_bins.split(",") if b]:
            t1 = time.perf_counter()
            config = {
                "n_bins": n_bins,
                "protect_fraction": args.insh_protect,
                "prune_fraction": args.insh_prune,
            }
            enc = {"weights": {}, "exp_avg": {}, "exp_avg_sq": {}}
            for index, name in enumerate(param_names):
                w = current[name].to(device)
                enc["weights"][name] = insh_encode(w, grads[name], config, seed * 100003 + 3 * index)
                for offset, (part, source) in enumerate(
                    (("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)), start=1
                ):
                    x = source[name].to(device)
                    enc[part][name] = insh_encode(
                        x, torch.zeros_like(x), config, seed * 100003 + 3 * index + offset
                    )
            sizes = {part: measure_bytes(payload, lzma_on) for part, payload in enc.items()}
            dec = {
                part: {n: insh_decode(enc[part][n], current[n].shape, device).cpu() for n in param_names}
                for part in enc
            }
            emit(
                {
                    "method": "inshrinkerator_like",
                    "fidelity": "reimplementation_of_inshrinkerator_partition_and_quantization_simplified",
                    "seed": seed,
                    "operating_point": config,
                    "bytes": sizes,
                    "total_bytes": total_sizes(sizes),
                    "restored": evaluate_weights(dec["weights"]),
                    "relative_l2_error": {
                        "weights": rel_error(dec["weights"], current),
                        "residual": rel_error(
                            {n: dec["weights"][n] - base_params[n].float() for n in param_names},
                            residual,
                        ),
                        "exp_avg": rel_error(dec["exp_avg"], exp_avg),
                        "exp_avg_sq": rel_error(dec["exp_avg_sq"], exp_avg_sq),
                    },
                    "seconds": time.perf_counter() - t1,
                }
            )
            del enc, dec
        del grads
    write()


if __name__ == "__main__":
    main()
