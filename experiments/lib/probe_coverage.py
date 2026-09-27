"""Coverage-tracked sampled Gauss--Newton probes for residual damage scoring.

Mathematical contract (mean token cross-entropy over N valid positions):

    F   = (1/N) sum_t sum_y p_ty v_ty v_ty^T,   v_ty = J_t^T (e_y - p_t)
    F_S = (1/N) sum_t sum_{y in S_t} p_ty v_ty v_ty^T

``F`` is the Gauss--Newton (model-distribution Fisher) matrix of the mean loss;
it is *not* the exact Hessian and not the true-label empirical Fisher.  At probe
round k, every position t draws ``y_t ~ p_t`` (with replacement across rounds).
Only a token not yet in ``S_t`` contributes; it receives the backward weight

    c_t = xi_t * sqrt(p_ty) * 1[new] / sqrt(N),   xi_t in {-1, +1}

so one ordinary backward of ``sum_t c_t CE_t`` yields ``w_k = J^T sum_t c_t (p_t - e_y)``.
Conditional on the sampled sets, ``E_xi[sum_k w_k w_k^T] = F_S`` exactly, and the
residual curvature product is estimated by ``u = sum_k w_k (w_k . d)``.  A
repeated token has zero weight and adds no coverage mass.

Truncation bound (conditions: ``||v_ty|| <= G`` for all t, y):
``0 <= F - F_S`` (PSD order) and ``||F - F_S||_2 <= (1 - mean_t q_t) G^2``.
The bound concerns the probability-weighted truncated operator only; it does not
bound the finite Rademacher estimate, the Hessian/Gauss--Newton gap, the mask
ranking, or the restored validation loss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

# --------------------------------------------------------------------------
# Coverage tracking and probe weights
# --------------------------------------------------------------------------


@dataclass
class CoverageTracker:
    """Per-position de-duplicated candidate sets S_t and their mass q_t."""

    num_positions: int
    max_rounds: int
    device: torch.device | str = "cpu"
    seen: torch.Tensor = field(init=False)
    mass: torch.Tensor = field(init=False)
    rounds: int = field(init=False, default=0)
    draws: int = field(init=False, default=0)
    repeats: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.num_positions <= 0 or self.max_rounds <= 0:
            raise ValueError("num_positions and max_rounds must be positive")
        self.seen = torch.full(
            (self.num_positions, self.max_rounds), -1, dtype=torch.long, device=self.device
        )
        self.mass = torch.zeros(self.num_positions, dtype=torch.float64, device=self.device)

    def update(self, tokens: torch.Tensor, token_probs: torch.Tensor) -> torch.Tensor:
        """Register one draw per position; return the boolean 'new token' mask."""
        if self.rounds >= self.max_rounds:
            raise RuntimeError("CoverageTracker exceeded max_rounds")
        tokens = tokens.to(self.seen.device).long().reshape(-1)
        token_probs = token_probs.to(self.seen.device).double().reshape(-1)
        if tokens.numel() != self.num_positions or token_probs.numel() != self.num_positions:
            raise ValueError("one draw per position is required")
        is_new = ~(self.seen[:, : self.rounds] == tokens[:, None]).any(dim=1)
        self.seen[:, self.rounds] = tokens
        self.mass += torch.where(is_new, token_probs, torch.zeros_like(token_probs))
        # Guard against round-off above one.
        self.mass.clamp_(max=1.0)
        self.rounds += 1
        self.draws += int(tokens.numel())
        self.repeats += int((~is_new).sum().item())
        return is_new

    def stats(self, taus: Sequence[float] = (0.9, 0.95)) -> Dict[str, object]:
        q = self.mass.detach().cpu().double()
        quant = torch.quantile(q.float(), torch.tensor([0.01, 0.05, 0.10, 0.50])).tolist()
        out: Dict[str, object] = {
            "rounds": self.rounds,
            "mean_mass": float(q.mean()),
            "min_mass": float(q.min()),
            "p01_mass": float(quant[0]),
            "p05_mass": float(quant[1]),
            "p10_mass": float(quant[2]),
            "median_mass": float(quant[3]),
            "cumulative_draws": self.draws,
            "cumulative_repeats": self.repeats,
            "cumulative_repeat_rate": self.repeats / max(1, self.draws),
            "truncation_factor": float(1.0 - q.mean()),
        }
        for tau in taus:
            out[f"frac_positions_ge_{tau:g}"] = float((q >= tau).double().mean())
        return out


def coverage_probe_weights(
    token_probs: torch.Tensor,
    is_new: torch.Tensor,
    signs: torch.Tensor,
    total_positions: int,
) -> torch.Tensor:
    """Per-position backward weights c_t = xi_t sqrt(p_ty) 1[new] / sqrt(N)."""
    if total_positions <= 0:
        raise ValueError("total_positions must be positive")
    p = token_probs.double().clamp_min(0.0)
    w = signs.double() * p.sqrt() * is_new.double() / math.sqrt(total_positions)
    return w


def mc_probe_weights(num: int, total_positions: int, device=None) -> torch.Tensor:
    """Standard Monte-Carlo Fisher weights 1/sqrt(N) (average over rounds later)."""
    return torch.full((num,), 1.0 / math.sqrt(total_positions), dtype=torch.float64, device=device)


def rademacher(num: int, generator: torch.Generator, device) -> torch.Tensor:
    bits = torch.randint(0, 2, (num,), generator=generator, device=device)
    return bits.double().mul_(2.0).sub_(1.0)


def weighted_token_ce(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """sum_t w_t CE(logits_t, targets_t) computed in float32."""
    logits = logits.float().reshape(-1, logits.shape[-1])
    ce = torch.nn.functional.cross_entropy(logits, targets.reshape(-1), reduction="none")
    return (ce * weights.to(ce.dtype).reshape(-1)).sum()


# --------------------------------------------------------------------------
# Truncation bound (used by tests and by the theory note)
# --------------------------------------------------------------------------


def truncation_operator_gap(
    probs: torch.Tensor, vectors: torch.Tensor, covered: torch.Tensor
) -> Tuple[float, float]:
    """Return (||F - F_S||_2, (1 - mean q) * G^2) for explicit vectors.

    probs: [N, V]; vectors: [N, V, P] with vectors[t, y] = v_ty; covered: [N, V] bool.
    """
    n = probs.shape[0]
    p = probs.double()
    v = vectors.double()
    missing = (~covered).double() * p
    gap = torch.einsum("ty,typ,tyq->pq", missing, v, v) / n
    spectral = float(torch.linalg.eigvalsh(gap).abs().max())
    g_sq = float((v.norm(dim=-1) ** 2).max())
    q_mean = float((covered.double() * p).sum(dim=1).mean())
    return spectral, (1.0 - q_mean) * g_sq


# --------------------------------------------------------------------------
# Stopping rules (evaluated on recorded per-round statistics)
# --------------------------------------------------------------------------


def first_round_meeting(
    per_round: Sequence[Mapping[str, float]],
    key: str,
    tau: float,
    min_round: int = 1,
) -> Tuple[int, bool]:
    """First round (1-based) whose ``key`` >= tau; else (last round, False)."""
    if not per_round:
        raise ValueError("no rounds recorded")
    for index, record in enumerate(per_round, start=1):
        if index >= min_round and float(record[key]) >= tau:
            return index, True
    return len(per_round), False


# --------------------------------------------------------------------------
# Exact global selection over many tensors (nonnegative float32 scores)
# --------------------------------------------------------------------------

_POS_INF_KEY = 0x7F800000


def _keys(score: torch.Tensor) -> torch.Tensor:
    if score.dtype != torch.float32:
        raise TypeError("scores must be float32")
    return score.view(torch.int32)


def validate_scores(scores: Iterable[torch.Tensor]) -> None:
    for s in scores:
        if not torch.isfinite(s).all():
            raise ValueError("non-finite score")
        if (s < 0).any():
            raise ValueError("scores must be nonnegative")


def kth_smallest_key(scores: Sequence[torch.Tensor], k: int) -> int:
    """Integer key of the k-th smallest value (1-based) across all tensors."""
    total = sum(int(s.numel()) for s in scores)
    if not 1 <= k <= total:
        raise ValueError("k out of range")
    lo, hi = 0, _POS_INF_KEY
    while lo < hi:
        mid = (lo + hi) // 2
        count = 0
        for s in scores:
            count += int((_keys(s) <= mid).sum().item())
        if count >= k:
            hi = mid
        else:
            lo = mid + 1
    return lo


def select_smallest(scores: Sequence[torch.Tensor], k: int) -> List[torch.Tensor]:
    """Boolean masks marking exactly k smallest entries; ties by (tensor, flat index)."""
    if k <= 0:
        return [torch.zeros_like(s, dtype=torch.bool) for s in scores]
    total = sum(int(s.numel()) for s in scores)
    if k >= total:
        return [torch.ones_like(s, dtype=torch.bool) for s in scores]
    key = kth_smallest_key(scores, k)
    masks = [(_keys(s) < key) for s in scores]
    remaining = k - sum(int(m.sum().item()) for m in masks)
    for s, m in zip(scores, masks):
        if remaining <= 0:
            break
        ties = (_keys(s) == key).reshape(-1)
        n_ties = int(ties.sum().item())
        if n_ties == 0:
            continue
        take = min(remaining, n_ties)
        if take == n_ties:
            m.reshape(-1).logical_or_(ties)
        else:
            order = torch.cumsum(ties.to(torch.int64), dim=0)
            m.reshape(-1).logical_or_(ties & (order <= take))
        remaining -= take
    if remaining != 0:
        raise RuntimeError("tie resolution failed")
    return masks


# --------------------------------------------------------------------------
# Bit packing and mask comparison
# --------------------------------------------------------------------------

_BIT_WEIGHTS: Dict[torch.device, torch.Tensor] = {}
_POPCOUNT: Dict[torch.device, torch.Tensor] = {}


def _bit_weights(device) -> torch.Tensor:
    device = torch.device(device)
    if device not in _BIT_WEIGHTS:
        _BIT_WEIGHTS[device] = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=device)
    return _BIT_WEIGHTS[device]


def _popcount_table(device) -> torch.Tensor:
    device = torch.device(device)
    if device not in _POPCOUNT:
        table = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.int64, device=device)
        _POPCOUNT[device] = table
    return _POPCOUNT[device]


def pack_mask(mask: torch.Tensor) -> torch.Tensor:
    flat = mask.reshape(-1).to(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8, device=flat.device)])
    return (flat.view(-1, 8) * _bit_weights(flat.device)).sum(dim=1, dtype=torch.uint8)


def unpack_mask(packed: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    numel = int(math.prod(shape))
    bits = (packed[:, None] & _bit_weights(packed.device)) != 0
    return bits.reshape(-1)[:numel].reshape(shape)


def packed_popcount(packed: torch.Tensor) -> int:
    return int(_popcount_table(packed.device)[packed.long()].sum().item())


def packed_intersection(a: torch.Tensor, b: torch.Tensor) -> int:
    return packed_popcount(a & b.to(a.device))


# --------------------------------------------------------------------------
# Stochastic rounding float32 -> bfloat16
# --------------------------------------------------------------------------


def stochastic_round_bf16(x: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    """Unbiased stochastic rounding of finite float32 values to bfloat16."""
    x = x.float().contiguous()
    bits = x.view(torch.int32)
    noise = torch.randint(0, 1 << 16, x.shape, generator=generator, device=x.device, dtype=torch.int32)
    rounded = (bits + noise) & torch.tensor(-65536, dtype=torch.int32, device=x.device)
    return rounded.view(torch.float32).to(torch.bfloat16)


# --------------------------------------------------------------------------
# Archive byte accounting (bitmap or COO, whichever is smaller per tensor)
# --------------------------------------------------------------------------


def residual_payload_bytes(numel: int, kept: int, value_bytes: int = 2) -> Dict[str, int]:
    bitmap = (numel + 7) // 8 + value_bytes * kept
    coo = (4 if numel < 2**32 else 8) * kept + value_bytes * kept
    return {"bitmap": bitmap, "coo": coo, "chosen": min(bitmap, coo)}


# --------------------------------------------------------------------------
# Rank statistics on a fixed coordinate subsample
# --------------------------------------------------------------------------


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    ra = torch.empty_like(a)
    rb = torch.empty_like(b)
    ra[a.argsort()] = torch.arange(a.numel(), dtype=torch.float64)
    rb[b.argsort()] = torch.arange(b.numel(), dtype=torch.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(ra.norm() * rb.norm())
    return float((ra * rb).sum()) / denom if denom > 0 else float("nan")


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    denom = float(a.norm() * b.norm())
    return float((a * b).sum()) / denom if denom > 0 else float("nan")


__all__ = [
    "CoverageTracker",
    "coverage_probe_weights",
    "cosine",
    "first_round_meeting",
    "kth_smallest_key",
    "mc_probe_weights",
    "pack_mask",
    "packed_intersection",
    "packed_popcount",
    "rademacher",
    "residual_payload_bytes",
    "select_smallest",
    "spearman",
    "stochastic_round_bf16",
    "truncation_operator_gap",
    "unpack_mask",
    "validate_scores",
    "weighted_token_ce",
]
