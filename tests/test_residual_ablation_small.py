"""CPU checks for the iterative residual-pruning ablation helpers on a tiny GPT-NeoX."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

transformers = pytest.importorskip("transformers")


def _tiny_sweep():
    from run_residual_sweep_small import Sweep

    from experiments.lib.residual_scoring import eligible_layers, transformer_layers

    torch.manual_seed(0)
    cfg = transformers.GPTNeoXConfig(vocab_size=64, hidden_size=16, num_hidden_layers=2,
                                     num_attention_heads=2, intermediate_size=32, max_position_embeddings=32)
    model = transformers.GPTNeoXForCausalLM(cfg).eval()
    layers = eligible_layers(model, "pythia")
    blocks = transformer_layers(model, "pythia")
    names = [n for layer in layers for n in layer]
    params = dict(model.named_parameters())
    cur = {n: params[n].detach().clone() for n in names}
    ref = {n: cur[n] + 0.05 * torch.randn_like(cur[n]) for n in names}
    ids = torch.randint(0, 64, (2, 12))
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    return Sweep(model, "pythia", layers, blocks, cur, ref, [batch], "cpu"), [batch, batch]


def test_geometric_schedule_ends_at_target_and_is_increasing():
    import run_residual_ablation_small as abl

    for k in (1, 2, 4, 8, 16):
        sched = abl.geometric_schedule(0.995, k)
        assert len(sched) == k and sched[-1] == 0.995
        assert all(a < b for a, b in zip(sched, sched[1:]))
        assert math.isclose(1 - sched[0], 0.005 ** (1 / k))
    assert abl.list_schedule(0.9) == [0.5, 0.8, 0.9]
    assert abl.list_schedule(0.93) == [0.5, 0.8, 0.9, 0.93]


@pytest.mark.parametrize("scorer", ["first_order", "ucb"])
def test_run_iterative_is_monotone_hits_budget_and_restores_model(scorer):
    import run_residual_ablation_small as abl

    s, batches = _tiny_sweep()
    before = {n: p.detach().clone() for n, p in s.model.named_parameters()}
    sched = abl.geometric_schedule(0.9, 4)
    _, stages, keep = abl.run_iterative(s, batches, sched, abl.SCORERS[scorer], 0, eval_at=[], return_keep=True)
    total = sum(s.delta[n].numel() for n in s.names)
    assert sum(int((~keep[n]).sum()) for n in s.names) == math.floor(0.9 * total)
    assert len(stages) == 4
    for n, p in s.model.named_parameters():
        assert torch.equal(p.detach(), before[n])


def test_one_stage_first_order_matches_direct_gradient_score():
    import run_residual_ablation_small as abl

    s, batches = _tiny_sweep()
    scores, _ = abl.score_first_order(s, batches, None, 0)
    from experiments.lib.residual_runtime import lm_loss

    params = [s.params[n] for n in s.names]
    g = [torch.zeros_like(p) for p in params]
    for b in batches:
        for acc, gr in zip(g, torch.autograd.grad(lm_loss(s.model, b, "cpu"), params)):
            acc.add_(gr / len(batches))
    for i, n in enumerate(s.names):
        torch.testing.assert_close(scores[n], (g[i] * s.delta[n]).abs(), rtol=1e-5, atol=1e-9)
