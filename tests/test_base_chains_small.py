"""CPU checks for the base-reference compression-recovery chains on a tiny GPT-NeoX."""
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


def _setup():
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

    def batch():
        ids = torch.randint(0, 64, (2, 12))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}

    batches = [batch() for _ in range(4)]
    s = Sweep(model, "pythia", layers, blocks, cur, ref, [batch()], "cpu")
    ref_cpu = {n: s.ref_dev[n].clone() for n in names}
    return s, batches, ref_cpu


@pytest.mark.parametrize("method", ["no_compression", "excp_style", "inshrinkerator_style", "dacp",
                                    "first_order_ucb", "iterative_ucb"])
def test_chain_restores_to_base_keeps_moments_and_trains(method):
    import run_base_chains_small as bc

    from experiments.lib.residual_runtime import lm_loss

    s, batches, ref_cpu = _setup()
    opt = torch.optim.AdamW(s.model.parameters(), lr=1e-3, weight_decay=0.01)
    last = {}
    compressed_params = []

    def train_fn(model, o, batch):
        # between the last step of a cycle and the first step of the next, only compression runs:
        # the moments must be untouched and the pruned coordinates must sit at the base values
        if last:
            for p in model.parameters():
                if p in o.state:
                    assert torch.equal(o.state[p]["exp_avg"], last[p])
        compressed_params.append({n: s.params[n].detach().clone() for n in s.names})
        model.train()
        o.zero_grad()
        loss = lm_loss(model, batch, "cpu")
        loss.backward()
        o.step()
        for p in model.parameters():
            last[p] = o.state[p]["exp_avg"].clone()
        return float(loss)

    P = 0.9
    cycles = bc.run_chain(s, method, P, 2, [[batches[0]], [batches[1]]], opt, train_fn, batches[:2], batches,
                          0, 4, 4, ref_cpu, 1000, lambda _m: None)
    assert [c["cycle"] for c in cycles] == [1, 2] and [c["step"] for c in cycles] == [1000, 1001]
    total = sum(v.numel() for v in ref_cpu.values())
    for c, snap in zip(cycles, compressed_params):
        assert math.isfinite(c["immediate"]["val"]["loss"]) and math.isfinite(c["final"]["val"]["loss"])
        assert len(c["train_log"]["train_losses"]) == 1
        at_ref = sum(int((snap[n] == ref_cpu[n]).sum()) for n in s.names)
        if method == "no_compression":
            assert "restore" not in c
            continue
        kept = c["restore"]["kept"]
        assert kept == sum(c["restore"]["kept_per_layer"])
        if method != "dacp":  # Weibull allocation floors per-layer counts
            assert kept == total - math.floor(P * total) == c["restore"]["target_kept"]
        else:
            assert abs(kept - c["restore"]["target_kept"]) <= len(s.layers)
        assert at_ref >= total - kept
    # the Sweep tracks the trained point after the chain
    for n in s.names:
        torch.testing.assert_close(s.delta[n], s.params[n].detach() - ref_cpu[n])


def test_chain_ids_match_32b_layout():
    import run_base_chains_small as bc

    assert bc.chain_id("no_compression", 0.95, 16, 16) == "no_compression"
    assert bc.chain_id("iterative_ucb", 0.95, 16, 16) == "iterative_ucb_B16_K16_0.95"
    assert bc.chain_id("first_order_ucb", 0.98, 16, 16) == "first_order_ucb_B16_0.98"
    assert bc.chain_id("dacp", 0.95, 16, 16) == "dacp_0.95"
