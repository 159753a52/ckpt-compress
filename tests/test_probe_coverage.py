import math

import pytest
import torch

from experiments.lib.probe_coverage import (
    CoverageTracker,
    coverage_probe_weights,
    first_round_meeting,
    pack_mask,
    packed_intersection,
    packed_popcount,
    rademacher,
    residual_payload_bytes,
    select_smallest,
    spearman,
    stochastic_round_bf16,
    truncation_operator_gap,
    unpack_mask,
    weighted_token_ce,
)


def _toy(n=5, v=4, dim=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g, dtype=torch.float64)
    w = torch.randn(v, dim, generator=g, dtype=torch.float64)
    return x, w


def _explicit_vectors(x, w):
    """v_ty = J_t^T (e_y - p_t) for logits o_t = W x_t, flattened over W."""
    probs = torch.softmax(x @ w.T, dim=-1)
    n, v = probs.shape
    eye = torch.eye(v, dtype=torch.float64)
    resid = eye[None, :, :] - probs[:, None, :]  # [n, y, v]
    vec = torch.einsum("tyv,td->tyvd", resid, x).reshape(n, v, -1)
    return probs, vec


def test_full_coverage_operator_equals_gauss_newton():
    x, w = _toy()
    probs, vec = _explicit_vectors(x, w)
    n = x.shape[0]
    f = torch.einsum("ty,typ,tyq->pq", probs, vec, vec) / n
    # Gauss-Newton J^T C_o J with C_o = blockdiag(diag(p) - p p^T)/N
    gn = torch.zeros_like(f)
    for t in range(n):
        c = torch.diag(probs[t]) - torch.outer(probs[t], probs[t])
        jac = torch.kron(torch.eye(w.shape[0], dtype=torch.float64), x[t][None, :])
        gn += jac.T @ c @ jac / n
    assert torch.allclose(f, gn, atol=1e-12)
    # For a linear-in-parameter logit model the GN matrix is the exact Hessian.
    labels = torch.tensor([0, 1, 2, 3, 0])

    def loss(flat):
        logits = x @ flat.reshape(w.shape).T
        return torch.nn.functional.cross_entropy(logits, labels)

    hess = torch.autograd.functional.hessian(loss, w.reshape(-1))
    assert torch.allclose(hess, gn, atol=1e-10)


def test_weighted_backward_matches_explicit_seed_vector():
    x, w = _toy()
    probs, vec = _explicit_vectors(x, w)
    n = x.shape[0]
    tokens = torch.tensor([1, 0, 3, 2, 2])
    c = torch.tensor([0.3, -0.2, 0.5, 0.0, 0.1], dtype=torch.float64)
    param = w.clone().requires_grad_(True)
    loss = weighted_token_ce(x @ param.T, tokens, c)
    loss.backward()
    grad = param.grad.to(torch.float64).reshape(-1)
    # grad = J^T sum_t c_t (p_t - e_y) = -sum_t c_t v_ty
    expected = -(c[:, None] * vec[torch.arange(n), tokens]).sum(dim=0)
    assert torch.allclose(grad, expected, atol=1e-5)


def test_rademacher_expectation_equals_truncated_operator():
    x, w = _toy()
    probs, vec = _explicit_vectors(x, w)
    n, v = probs.shape
    gen = torch.Generator().manual_seed(1)
    tracker = CoverageTracker(n, 3)
    covered = torch.zeros(n, v, dtype=torch.bool)
    weights = []
    tokens_all = []
    for _ in range(3):
        tokens = torch.multinomial(probs.float(), 1, generator=gen).reshape(-1)
        p_y = probs[torch.arange(n), tokens]
        is_new = tracker.update(tokens, p_y)
        covered[torch.arange(n), tokens] = True
        weights.append(p_y.sqrt() * is_new.double() / math.sqrt(n))
        tokens_all.append(tokens)
    # Exact conditional expectation over xi: sum_k sum_t c_t^2 v v^T
    expect = torch.zeros(vec.shape[-1], vec.shape[-1], dtype=torch.float64)
    for c, tok in zip(weights, tokens_all):
        a = vec[torch.arange(n), tok]
        expect += torch.einsum("t,tp,tq->pq", c**2, a, a)
    f_s = torch.einsum("ty,typ,tyq->pq", covered.double() * probs, vec, vec) / n
    assert torch.allclose(expect, f_s, atol=1e-12)
    # Coverage mass equals the explicit covered probability.
    assert torch.allclose(tracker.mass, (covered.double() * probs).sum(dim=1), atol=1e-12)
    # Monte-Carlo over xi approaches the same operator.
    acc = torch.zeros_like(f_s)
    draws = 4000
    for _ in range(draws):
        for c, tok in zip(weights, tokens_all):
            xi = rademacher(n, gen, "cpu")
            wvec = ((xi * c)[:, None] * vec[torch.arange(n), tok]).sum(0)
            acc += torch.outer(wvec, wvec)
    acc /= draws
    assert torch.allclose(acc, f_s, atol=5e-2 * float(f_s.abs().max()))


def test_repeated_tokens_add_no_mass():
    tracker = CoverageTracker(2, 3)
    probs = torch.tensor([0.7, 0.4], dtype=torch.float64)
    new1 = tracker.update(torch.tensor([5, 1]), probs)
    new2 = tracker.update(torch.tensor([5, 2]), torch.tensor([0.7, 0.3], dtype=torch.float64))
    assert new1.tolist() == [True, True]
    assert new2.tolist() == [False, True]
    assert torch.allclose(tracker.mass, torch.tensor([0.7, 0.7], dtype=torch.float64))
    w = coverage_probe_weights(torch.tensor([0.7, 0.3]), new2, torch.tensor([1.0, -1.0]), 4)
    assert w[0] == 0 and math.isclose(float(w[1]), -math.sqrt(0.3) / 2, rel_tol=1e-6)
    stats = tracker.stats()
    assert stats["cumulative_repeats"] == 1 and stats["cumulative_draws"] == 4


def test_truncation_bound_holds():
    x, w = _toy(n=6, v=5, dim=3, seed=3)
    probs, vec = _explicit_vectors(x, w)
    gen = torch.Generator().manual_seed(2)
    for trial in range(20):
        covered = torch.rand(probs.shape, generator=gen) < 0.5
        gap, bound = truncation_operator_gap(probs, vec, covered)
        assert gap <= bound + 1e-12


def test_first_round_meeting():
    rec = [{"m": 0.5}, {"m": 0.91}, {"m": 0.96}]
    assert first_round_meeting(rec, "m", 0.9) == (2, True)
    assert first_round_meeting(rec, "m", 0.95) == (3, True)
    assert first_round_meeting(rec, "m", 0.99) == (3, False)


def test_select_smallest_exact_with_ties():
    a = torch.tensor([0.0, 0.5, 0.2, 0.2], dtype=torch.float32)
    b = torch.tensor([[0.2, 0.1], [0.0, 0.9]], dtype=torch.float32)
    masks = select_smallest([a, b], 4)
    assert sum(int(m.sum()) for m in masks) == 4
    # zeros and 0.1 first, then the first tie 0.2 in (tensor, index) order
    assert masks[0].tolist() == [True, False, True, False]
    assert masks[1].tolist() == [[False, True], [True, False]]
    gen = torch.Generator().manual_seed(0)
    s = [torch.rand(1000, generator=gen), torch.rand(37, generator=gen)]
    for k in (1, 100, 500, 1036):
        m = select_smallest(s, k)
        flat = torch.cat([t.reshape(-1) for t in s])
        sel = torch.cat([t.reshape(-1) for t in m])
        assert int(sel.sum()) == k
        assert flat[sel].max() <= flat[~sel].min() if k < flat.numel() else True


def test_pack_roundtrip_and_popcount():
    gen = torch.Generator().manual_seed(0)
    m = torch.rand(3, 7, generator=gen) < 0.4
    p = pack_mask(m)
    assert torch.equal(unpack_mask(p, m.shape), m)
    assert packed_popcount(p) == int(m.sum())
    m2 = torch.rand(3, 7, generator=gen) < 0.5
    assert packed_intersection(p, pack_mask(m2)) == int((m & m2).sum())


def test_stochastic_rounding_is_bf16_and_unbiased():
    x = torch.full((200000,), 1.0 + 2**-10, dtype=torch.float32)
    gen = torch.Generator().manual_seed(0)
    y = stochastic_round_bf16(x, gen)
    assert y.dtype == torch.bfloat16
    vals = set(y.float().unique().tolist())
    assert vals <= {1.0, 1.0 + 2**-7}
    assert abs(float(y.float().mean()) - (1.0 + 2**-10)) < 2e-5
    neg = stochastic_round_bf16(-x, gen)
    assert abs(float(neg.float().mean()) + (1.0 + 2**-10)) < 2e-5
    exact = torch.tensor([0.5, -2.0, 0.0], dtype=torch.float32)
    assert torch.equal(stochastic_round_bf16(exact, gen).float(), exact)


def test_payload_bytes_and_spearman():
    b = residual_payload_bytes(80, 4)
    assert b == {"bitmap": 10 + 8, "coo": 16 + 8, "chosen": 18}
    a = torch.arange(10.0)
    assert spearman(a, a * 3 + 1) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
