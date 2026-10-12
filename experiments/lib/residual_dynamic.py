"""Empirical-CDF budget selection and the manuscript's scalar feedback rule."""
import math
import torch


def budget_keep(scores, names, budget, pmax, allowed=None, device="cuda:0"):
    if not math.isfinite(budget) or budget < 0 or not 0 <= pmax < 1:
        raise ValueError("invalid damage or count budget")
    flat = torch.cat([scores[n].reshape(-1).to(device, torch.float32) for n in names])
    if not bool(torch.isfinite(flat).all()) or bool((flat < 0).any()):
        raise ValueError("scores must be finite and nonnegative")
    flat.add_(0.0)
    keys = flat.view(torch.int32)
    eligible = torch.ones_like(keys, dtype=torch.bool) if allowed is None else torch.cat([allowed[n].reshape(-1).to(device) for n in names])
    limit = math.floor(pmax * keys.numel())
    low, high = -1, int(keys.max()) + 1
    while high - low > 1:
        mid = (low + high) // 2
        omitted = (keys <= mid) & eligible
        count = int(torch.count_nonzero(omitted))
        cost = float(torch.where(omitted, flat, 0).sum(dtype=torch.float64))
        if count <= limit and cost <= budget:
            low = mid
        else:
            high = mid
    omitted = (keys <= low) & eligible
    count = int(torch.count_nonzero(omitted))
    cost = float(torch.where(omitted, flat, 0).sum(dtype=torch.float64))
    candidates = (keys > low) & eligible
    if bool(candidates.any()) and count < limit:
        next_key = int(keys[candidates].min())
        next_value = float(flat[(keys == next_key) & eligible][0])
        remaining = limit - count if next_value == 0 else min(limit - count, math.floor(max(0, budget - cost) / next_value))
        # Resolve equal-score coordinates by their fixed global index.
        for start in range(0, keys.numel(), 1 << 20):
            indices = torch.nonzero((keys[start:start + (1 << 20)] == next_key) & eligible[start:start + (1 << 20)]).flatten()[:remaining]
            omitted[start + indices] = True
            remaining -= indices.numel()
            if remaining == 0:
                break
    count = int(torch.count_nonzero(omitted))
    cost = float(torch.where(omitted, flat, 0).sum(dtype=torch.float64))
    if count > limit or cost > budget + max(1e-12, budget * 1e-10):
        raise RuntimeError("empirical budget correction failed")
    keep, offset = {}, 0
    for name in names:
        size = scores[name].numel()
        keep[name] = (~omitted[offset:offset + size]).reshape(scores[name].shape).cpu()
        offset += size
    return keep, {"proxy_cost": cost, "pruned": count, "eligible": keys.numel(), "sparsity": count / keys.numel(), "raw_proxy_budget": budget}


def update_scale(scale, proxy, delta, rho, floor, eps_max):
    if not all(math.isfinite(x) for x in (scale, proxy, delta)) or proxy <= 1e-20:
        return None
    ratio = max(delta, 0) / proxy
    if delta <= eps_max:
        return max(floor, (1 - rho) * scale + rho * ratio)
    return max(floor, scale, ratio)


def self_check():
    scores = {"a": torch.tensor([0., 0., 1., 1.]), "b": torch.tensor([2., 3.])}
    keep, meta = budget_keep(scores, ["a", "b"], 2., 0.9, device="cpu")
    assert meta["pruned"] == 4 and meta["proxy_cost"] == 2 and keep["b"].all()
    keep, meta = budget_keep(scores, ["a", "b"], 0., 0.2, device="cpu")
    assert meta["pruned"] == 1 and keep["a"].tolist() == [False, True, True, True]
    assert update_scale(1., 1., 2., .5, .01, .1) == 2.
    assert update_scale(1., 1., .05, .5, .01, .1) == .525
    assert update_scale(1., 0., .05, .5, .01, .1) is None
    print("dynamic selector and feedback self-check passed")


if __name__ == "__main__":
    self_check()
