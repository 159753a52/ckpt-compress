"""Exact k-smallest selection over score tensors that are recomputed on demand.

Scores are never all resident at once.  ``score_fn(i)`` returns tensor i as
nonnegative finite float32 on its device.  Selection uses the monotone integer
view of nonnegative floats: pass 1 builds a 65536-bin histogram of ``key >> 15``,
pass 2 a 32768-bin histogram of the low bits inside the pivot bin, pass 3 writes
masks.  Ties at the exact k-th value are resolved by (tensor order, flat index).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch

from experiments.lib.probe_coverage import pack_mask

COARSE_BITS = 16
FINE_BITS = 15
COARSE_BINS = 1 << COARSE_BITS
FINE_BINS = 1 << FINE_BITS


@dataclass(frozen=True)
class Target:
    name: str
    members: Sequence[int]  # tensor indices, in the fixed tie-breaking order
    k: int


def _keys(score: torch.Tensor) -> torch.Tensor:
    if score.dtype != torch.float32:
        raise TypeError("scores must be float32")
    return score.reshape(-1).view(torch.int32)


class StreamingSelector:
    def __init__(self, num_tensors: int, score_fn: Callable[[int], torch.Tensor]):
        self.num_tensors = num_tensors
        self.score_fn = score_fn
        self.hists: Optional[torch.Tensor] = None
        self.numels: List[int] = [0] * num_tensors

    def _checked(self, index: int) -> torch.Tensor:
        s = self.score_fn(index)
        if not bool(torch.isfinite(s).all()):
            raise ValueError(f"non-finite score in tensor {index}")
        if bool((s < 0).any()):
            raise ValueError(f"negative score in tensor {index}")
        return s

    def first_pass(self, hook: Optional[Callable[[int, torch.Tensor], None]] = None) -> None:
        hists = torch.zeros(self.num_tensors, COARSE_BINS, dtype=torch.int64)
        for i in range(self.num_tensors):
            s = self._checked(i)
            self.numels[i] = int(s.numel())
            hists[i] = torch.bincount(_keys(s) >> FINE_BITS, minlength=COARSE_BINS).cpu()
            if hook is not None:
                hook(i, s)
            del s
        self.hists = hists

    def select(self, targets: Sequence[Target]) -> Dict[str, Dict[str, object]]:
        if self.hists is None:
            raise RuntimeError("first_pass must run before select")
        plans = {}
        membership: Dict[int, List[str]] = {i: [] for i in range(self.num_tensors)}
        for t in targets:
            size = sum(self.numels[i] for i in t.members)
            if not 0 <= t.k <= size:
                raise ValueError(f"target {t.name}: k={t.k} outside [0, {size}]")
            plan = {"k": t.k, "size": size, "mode": "partial"}
            if t.k == 0:
                plan["mode"] = "none"
            elif t.k == size:
                plan["mode"] = "all"
            else:
                cum = self.hists[list(t.members)].sum(dim=0).cumsum(0)
                coarse = int(torch.searchsorted(cum, torch.tensor(t.k), right=False))
                plan["coarse"] = coarse
                plan["below"] = int(cum[coarse - 1]) if coarse > 0 else 0
                plan["fine"] = torch.zeros(FINE_BINS, dtype=torch.int64)
            plans[t.name] = plan
            for i in t.members:
                membership[i].append(t.name)
        # Pass 2: fine histograms inside each pivot bin.
        need_fine = {n for n, p in plans.items() if p["mode"] == "partial"}
        if need_fine:
            for i in range(self.num_tensors):
                names = [n for n in membership[i] if n in need_fine]
                if not names:
                    continue
                keys = _keys(self._checked(i))
                high = keys >> FINE_BITS
                for n in names:
                    sel = keys[high == plans[n]["coarse"]] & (FINE_BINS - 1)
                    plans[n]["fine"] += torch.bincount(sel, minlength=FINE_BINS).cpu()
                del keys, high
            for n in need_fine:
                p = plans[n]
                need = p["k"] - p["below"]
                fcum = p["fine"].cumsum(0)
                fine = int(torch.searchsorted(fcum, torch.tensor(need), right=False))
                p["key"] = (p["coarse"] << FINE_BITS) | fine
                p["count_less"] = p["below"] + (int(fcum[fine - 1]) if fine > 0 else 0)
                p["ties_to_take"] = p["k"] - p["count_less"]
                del p["fine"]
        # Pass 3: masks, in fixed tensor order for tie breaking.
        results = {t.name: {"packed": {}, "selected": 0, "threshold_key": plans[t.name].get("key")} for t in targets}
        remaining = {n: p.get("ties_to_take", 0) for n, p in plans.items()}
        for i in range(self.num_tensors):
            names = membership[i]
            if not names:
                continue
            s = None
            keys = None
            for n in names:
                mode = plans[n]["mode"]
                if mode == "none":
                    mask = torch.zeros(self.numels[i], dtype=torch.bool)
                elif mode == "all":
                    mask = torch.ones(self.numels[i], dtype=torch.bool)
                else:
                    if keys is None:
                        s = self._checked(i)
                        keys = _keys(s)
                    key = plans[n]["key"]
                    mask = keys < key
                    if remaining[n] > 0:
                        ties = keys == key
                        n_ties = int(ties.sum())
                        take = min(remaining[n], n_ties)
                        if take == n_ties:
                            mask |= ties
                        elif take > 0:
                            mask |= ties & (torch.cumsum(ties.to(torch.int64), 0) <= take)
                        remaining[n] -= take
                results[n]["packed"][i] = pack_mask(mask).cpu()
                results[n]["selected"] += int(mask.sum())
                del mask
            del s, keys
        for t in targets:
            if results[t.name]["selected"] != t.k:
                raise RuntimeError(f"target {t.name}: selected {results[t.name]['selected']} != {t.k}")
        return results


__all__ = ["StreamingSelector", "Target", "COARSE_BINS", "FINE_BINS"]
