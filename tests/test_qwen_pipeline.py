import pytest
import torch

transformers = pytest.importorskip("transformers")

from experiments.lib.qwen_pipeline import (
    PipelinedQwen,
    StochasticRoundingAdamW,
    math_sdpa,
    train_step,
)


def _tiny(dtype=torch.float32):
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "sdpa"
    model = transformers.Qwen2ForCausalLM(cfg).to(dtype)
    model.eval()
    return model


def test_forward_matches_huggingface():
    model = _tiny()
    ids = torch.randint(0, 64, (2, 12), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    pipe = PipelinedQwen(model, ["cpu", "cpu"], [2, 1])
    with torch.no_grad():
        out = pipe.forward(ids)
    assert torch.allclose(out, ref, atol=1e-5)
    stats = pipe.evaluate(ids, batch_size=1)
    with torch.no_grad():
        hf_loss = model(input_ids=ids, labels=ids).loss
    assert stats["tokens"] == 22
    assert abs(stats["loss"] - float(hf_loss)) < 1e-5


def test_block_hvp_matches_autograd_reference():
    model = _tiny(torch.float64)
    pipe = PipelinedQwen(model, ["cpu", "cpu"], [2, 1])
    ids = torch.randint(0, 64, (1, 10), generator=torch.Generator().manual_seed(2))
    layer = 1
    names = pipe.layer_eligible[layer]
    gen = torch.Generator().manual_seed(3)
    vec = {n: torch.randn(pipe.param(n).shape, generator=gen, dtype=torch.float64) for n in names}
    with torch.no_grad():
        _, inputs = pipe.forward(ids, collect_inputs=True)
    got = pipe.block_hvp(layer, inputs[layer], ids, vec, total_positions=9)

    params = [pipe.param(n) for n in names]
    # Reference: full forward from the embeddings with all block params tracked.
    pipe.set_requires_grad(names)
    with math_sdpa():
        logits = pipe.forward(ids)
        loss, n = pipe.nll_sum(logits, ids)
        grads = torch.autograd.grad(loss / n, params, create_graph=True)
        gv = sum((g * vec[k]).sum() for g, k in zip(grads, names))
        ref = torch.autograd.grad(gv, params)
    for k, r in zip(names, ref):
        # block_hvp returns float32; compare at float32 resolution.
        assert torch.allclose(got[k].double(), r, rtol=1e-5, atol=1e-6)


def test_eligibility_and_training_step_reduce_loss():
    model = _tiny(torch.bfloat16)
    pipe = PipelinedQwen(model, ["cpu", "cpu"], [1, 2])
    assert not pipe.param("model.embed_tokens.weight").requires_grad
    assert all(n.split(".")[-2].endswith("proj") for n in pipe.eligible_names)
    assert len(pipe.eligible_names) == 3 * 7
    opt = StochasticRoundingAdamW(pipe, lr=3e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, clip=1.0, seed=0)
    opt.init_state()
    ids = torch.randint(0, 64, (1, 16), generator=torch.Generator().manual_seed(4))
    first = train_step(pipe, opt, ids)["loss"]
    for _ in range(15):
        last = train_step(pipe, opt, ids)["loss"]
    assert last < first
    assert opt.step_count == 16
