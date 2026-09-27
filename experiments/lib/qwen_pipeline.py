"""Single-process layer-pipelined Qwen2 runner for residual checkpoint experiments.

The model is split by decoder layers over the visible CUDA devices (no data or
tensor parallelism).  All helpers operate on ordinary ``nn.Parameter`` objects,
so gradients, block-wise Hessian--vector products and optimizer states stay on
the device that owns each parameter.
"""
from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

from experiments.lib.probe_coverage import stochastic_round_bf16

ELIGIBLE_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


@contextmanager
def math_sdpa():
    """Force the math SDPA kernel (the fused kernels lack double backward)."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            yield
    except ImportError:  # pragma: no cover - older torch
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            yield


class PipelinedQwen:
    """Place a ``Qwen2ForCausalLM`` over devices by contiguous decoder layers."""

    def __init__(self, model: nn.Module, devices: Sequence[torch.device | str], split: Sequence[int]):
        layers = model.model.layers
        if sum(split) != len(layers) or len(split) != len(devices):
            raise ValueError("split must assign every decoder layer to one device")
        self.model = model
        self.devices = [torch.device(d) for d in devices]
        self.layer_device: List[torch.device] = []
        for dev, count in zip(self.devices, split):
            self.layer_device.extend([dev] * count)
        model.config.use_cache = False
        model.model.embed_tokens.to(self.devices[0])
        for layer, dev in zip(layers, self.layer_device):
            layer.to(dev)
        model.model.norm.to(self.devices[-1])
        model.lm_head.to(self.devices[-1])
        self.rotary = {}
        for dev in self.devices:
            self.rotary[dev] = copy.deepcopy(model.model.rotary_emb).to(dev)
        self._rope_cache: Dict[Tuple[torch.device, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        # Frozen embedding and output head; decoder layers and the final norm train.
        model.model.embed_tokens.weight.requires_grad_(False)
        model.lm_head.weight.requires_grad_(False)
        self.named = dict(model.named_parameters())
        self.trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.eligible_names = [n for n in self.trainable_names if n.endswith(ELIGIBLE_SUFFIXES)]
        self.layer_of = {}
        for name in self.eligible_names:
            self.layer_of[name] = int(name.split(".")[2])
        self.layer_eligible: List[List[str]] = [[] for _ in layers]
        for name in self.eligible_names:
            self.layer_eligible[self.layer_of[name]].append(name)

    # ------------------------------------------------------------------ utils
    @property
    def num_layers(self) -> int:
        return len(self.model.model.layers)

    def param(self, name: str) -> nn.Parameter:
        return self.named[name]

    def set_requires_grad(self, names: Iterable[str]) -> None:
        wanted = set(names)
        for name, p in self.named.items():
            p.requires_grad_(name in wanted)

    def zero_grad(self) -> None:
        for p in self.named.values():
            p.grad = None

    def _rope(self, dev: torch.device, batch: int, length: int, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (dev, batch, length)
        if key not in self._rope_cache:
            pos = torch.arange(length, device=dev)[None, :].expand(batch, -1)
            dummy = torch.zeros(1, device=dev, dtype=dtype)
            cos, sin = self.rotary[dev](dummy, pos)
            self._rope_cache[key] = (cos, sin, pos)
        return self._rope_cache[key]

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        start_layer: int = 0,
        hidden: Optional[torch.Tensor] = None,
        collect_inputs: bool = False,
    ):
        """Return logits [B, L, V]; optionally the detached input of every layer."""
        batch, length = input_ids.shape
        inputs: List[torch.Tensor] = []
        if start_layer == 0:
            h = self.model.model.embed_tokens(input_ids.to(self.devices[0]))
        else:
            if hidden is None:
                raise ValueError("hidden input is required when start_layer > 0")
            h = hidden
        dtype = self.model.model.embed_tokens.weight.dtype
        for index in range(start_layer, self.num_layers):
            dev = self.layer_device[index]
            h = h.to(dev)
            if collect_inputs:
                inputs.append(h.detach())
            cos, sin, pos = self._rope(dev, batch, length, dtype)
            out = self.model.model.layers[index](
                h, attention_mask=None, position_ids=pos, position_embeddings=(cos, sin)
            )
            h = out[0] if isinstance(out, tuple) else out
        h = self.model.model.norm(h.to(self.devices[-1]))
        logits = self.model.lm_head(h)
        return (logits, inputs) if collect_inputs else logits

    @staticmethod
    def shifted(logits: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flat = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        target = input_ids[:, 1:].reshape(-1).to(logits.device)
        return flat, target

    def nll_sum(self, logits: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
        flat, target = self.shifted(logits, input_ids)
        loss = torch.nn.functional.cross_entropy(flat.float(), target, reduction="sum")
        return loss, int(target.numel())

    @torch.no_grad()
    def evaluate(self, chunks: torch.Tensor, batch_size: int) -> Dict[str, float]:
        """Token-weighted mean NLL and perplexity over all shifted positions."""
        total = 0.0
        count = 0
        for start in range(0, chunks.shape[0], batch_size):
            ids = chunks[start : start + batch_size]
            logits = self.forward(ids)
            loss, n = self.nll_sum(logits, ids)
            total += float(loss.double())
            count += n
            del logits
        mean = total / count
        return {"loss": mean, "ppl": math.exp(mean), "tokens": count, "nll_sum": total}

    # -------------------------------------------------- block-wise exact HVP
    def block_hvp(
        self,
        layer_index: int,
        hidden_in: torch.Tensor,
        input_ids: torch.Tensor,
        vector: Dict[str, torch.Tensor],
        total_positions: int,
    ) -> Dict[str, torch.Tensor]:
        """H_l d_l for one decoder layer (other parameters held constant).

        Matches ``dacp.tools.importance.compute_hvp_blockwise``: loss is the
        token-sum NLL divided by the global position count, first-order grads
        use ``create_graph=True`` for this block only, then one more backward of
        ``grad . d``.  Inputs to the block do not depend on its parameters, so the
        forward starts from the cached block input.
        """
        names = self.layer_eligible[layer_index]
        params = [self.named[n] for n in names]
        self.set_requires_grad(names)
        with math_sdpa():
            logits = self.forward(input_ids, start_layer=layer_index, hidden=hidden_in)
            loss, _ = self.nll_sum(logits, input_ids)
            loss = loss / total_positions
            grads = torch.autograd.grad(loss, params, create_graph=True)
            gv = sum((g.float() * vector[n].float()).sum() for g, n in zip(grads, names))
            hv = torch.autograd.grad(gv, params)
        out = {n: h.detach().float() for n, h in zip(names, hv)}
        del logits, loss, grads, gv, hv
        return out


class StochasticRoundingAdamW:
    """AdamW on bf16 parameters with fp32 arithmetic and stochastic rounding.

    Moments are stored in bf16 (round-to-nearest).  The random bits of every
    rounding depend only on (seed, global step, parameter index), so branches
    that start from the same step see the same rounding noise.
    """

    def __init__(
        self,
        pipe: PipelinedQwen,
        lr: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        clip: float,
        seed: int,
    ):
        self.pipe = pipe
        self.names = list(pipe.trainable_names)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip = clip
        self.seed = seed
        self.step_count = 0
        self.exp_avg: Dict[str, torch.Tensor] = {}
        self.exp_avg_sq: Dict[str, torch.Tensor] = {}

    def init_state(self) -> None:
        for n in self.names:
            p = self.pipe.param(n)
            self.exp_avg[n] = torch.zeros_like(p, dtype=torch.bfloat16)
            self.exp_avg_sq[n] = torch.zeros_like(p, dtype=torch.bfloat16)

    def state_to(self, device_of_param: bool) -> None:
        """Move moments to their parameter device (True) or to CPU (False)."""
        for n in self.names:
            dev = self.pipe.param(n).device if device_of_param else torch.device("cpu")
            self.exp_avg[n] = self.exp_avg[n].to(dev)
            self.exp_avg_sq[n] = self.exp_avg_sq[n].to(dev)

    def load_state(self, exp_avg: Dict[str, torch.Tensor], exp_avg_sq: Dict[str, torch.Tensor], step: int) -> None:
        for n in self.names:
            dev = self.pipe.param(n).device
            self.exp_avg[n] = exp_avg[n].to(dev, copy=True)
            self.exp_avg_sq[n] = exp_avg_sq[n].to(dev, copy=True)
        self.step_count = int(step)

    @torch.no_grad()
    def grad_norm(self) -> float:
        total = 0.0
        for n in self.names:
            g = self.pipe.param(n).grad
            if g is not None:
                total += float(g.float().square().sum().double())
        return math.sqrt(total)

    @torch.no_grad()
    def step(self) -> float:
        self.step_count += 1
        t = self.step_count
        b1, b2 = self.betas
        bc1 = 1.0 - b1**t
        bc2 = 1.0 - b2**t
        norm = self.grad_norm()
        scale = min(1.0, self.clip / (norm + 1e-6)) if self.clip > 0 else 1.0
        for index, n in enumerate(self.names):
            p = self.pipe.param(n)
            if p.grad is None:
                continue
            g = p.grad.float() * scale
            m = self.exp_avg[n].float().mul_(b1).add_(g, alpha=1.0 - b1)
            v = self.exp_avg_sq[n].float().mul_(b2).addcmul_(g, g, value=1.0 - b2)
            update = (m / bc1) / ((v / bc2).sqrt_().add_(self.eps))
            decay = self.weight_decay if p.dim() >= 2 else 0.0
            new = p.float().mul_(1.0 - self.lr * decay).add_(update, alpha=-self.lr)
            gen = torch.Generator(device=p.device)
            gen.manual_seed((self.seed * 1_000_003 + t * 10_007 + index) % (2**63 - 1))
            p.copy_(stochastic_round_bf16(new, gen))
            self.exp_avg[n].copy_(m.to(torch.bfloat16))
            self.exp_avg_sq[n].copy_(v.to(torch.bfloat16))
            del g, m, v, update, new
        return norm


def train_step(pipe: PipelinedQwen, opt: StochasticRoundingAdamW, ids: torch.Tensor) -> Dict[str, float]:
    pipe.set_requires_grad(pipe.trainable_names)
    pipe.zero_grad()
    logits = pipe.forward(ids)
    loss, n = pipe.nll_sum(logits, ids)
    mean = loss / n
    mean.backward()
    del logits
    norm = opt.step()
    pipe.zero_grad()
    return {"loss": float(mean.detach()), "grad_norm": norm, "step": opt.step_count}


__all__ = [
    "ELIGIBLE_SUFFIXES",
    "PipelinedQwen",
    "StochasticRoundingAdamW",
    "math_sdpa",
    "train_step",
]
