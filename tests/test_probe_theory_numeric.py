"""Numerical checks for the probe-theory appendix (app:probe-theory).

Model: linear softmax, logits o_t = W x_t, parameters theta = vec(W).
Mean token cross-entropy over N positions; v_{t,y} = J_t^T (e_y - p_t).
All checks run on CPU in float64.
"""

import itertools
import math

import pytest
import torch

DT = torch.float64


# ---------------------------------------------------------------- helpers
def make_model(n_pos=4, vocab=5, dim=3, seed=0, scale=1.5):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(vocab, dim, generator=g, dtype=DT) * scale
    X = torch.randn(n_pos, dim, generator=g, dtype=DT)
    return W, X


def probs(W, X):
    return torch.softmax(X @ W.T, dim=-1)  # (N, V)


def v_vectors(W, X):
    """v[t, y] = -grad_theta CE_t(y) = J_t^T (e_y - p_t), via autograd."""
    n_pos, vocab = X.shape[0], W.shape[0]
    out = torch.zeros(n_pos, vocab, W.numel(), dtype=DT)
    for t in range(n_pos):
        for y in range(vocab):
            Wp = W.clone().requires_grad_(True)
            ce = torch.nn.functional.cross_entropy(
                (X[t] @ Wp.T)[None], torch.tensor([y]))
            (grad,) = torch.autograd.grad(ce, Wp)
            out[t, y] = -grad.reshape(-1)
    return out


def fisher(P, V, sets=None):
    n_pos, vocab = P.shape
    F = torch.zeros(V.shape[-1], V.shape[-1], dtype=DT)
    for t in range(n_pos):
        for y in range(vocab):
            if sets is None or y in sets[t]:
                F += P[t, y] * torch.outer(V[t, y], V[t, y])
    return F / n_pos


def rademacher_exact(a):
    """Exact E_xi[w w^T] and E_xi||w w^T - A||_F^2 by enumerating xi."""
    n = a.shape[0]
    A = a.T @ a
    mean = torch.zeros_like(A)
    sq = torch.zeros((), dtype=DT)
    signs = list(itertools.product([-1.0, 1.0], repeat=n))
    for xi in signs:
        w = torch.tensor(xi, dtype=DT) @ a
        M = torch.outer(w, w)
        mean += M
        sq += ((M - A) ** 2).sum()
    return mean / len(signs), sq / len(signs), A


def expected_coverage(p, m):
    return float((p * (1 - (1 - p) ** m)).sum())


# ------------------------------------------------------------ sanity: F
def test_fisher_equals_gauss_newton():
    W, X = make_model()
    P, V = probs(W, X), v_vectors(W, X)
    N, Vs, D = X.shape[0], W.shape[0], X.shape[1]
    F = fisher(P, V)
    GN = torch.zeros_like(F)
    for t in range(N):
        J = torch.kron(torch.eye(Vs, dtype=DT), X[t][None])  # d o_t / d vec(W)
        C = torch.diag(P[t]) - torch.outer(P[t], P[t])
        GN += J.T @ C @ J
    assert torch.allclose(F, GN / N, atol=1e-12)


# ------------------------------------------------------------ (a) coverage
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_coverage_closed_form_vs_monte_carlo(seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.softmax(torch.randn(8, generator=g, dtype=DT) * 1.5, 0)
    trials = 200_000
    draws = torch.multinomial(p.expand(trials, -1), 8, replacement=True,
                              generator=g)
    prev, prev_inc = 0.0, math.inf
    for m in range(1, 9):
        seen = torch.zeros(trials, 8, dtype=torch.bool)
        seen.scatter_(1, draws[:, :m], True)
        q = (seen.to(DT) * p).sum(1)
        exact = expected_coverage(p, m)
        se = float(q.std()) / math.sqrt(trials)
        assert abs(float(q.mean()) - exact) < 5 * se + 1e-12
        # top-m dominance holds pointwise
        top_m = float(torch.sort(p, descending=True).values[:m].sum())
        assert float(q.max()) <= top_m + 1e-12
        # distinct-count formula
        n_distinct = seen.sum(1).to(DT)
        exact_distinct = float((1 - (1 - p) ** m).sum())
        se_d = float(n_distinct.std()) / math.sqrt(trials)
        assert abs(float(n_distinct.mean()) - exact_distinct) < 5 * se_d + 1e-12
        # monotone and concave
        inc = exact - prev
        assert inc >= -1e-15 and inc <= prev_inc + 1e-15
        assert abs(inc - float((p ** 2 * (1 - p) ** (m - 1)).sum())) < 1e-12
        prev, prev_inc = exact, inc
    h2 = -math.log(float((p ** 2).sum()))
    assert abs(expected_coverage(p, 1) - math.exp(-h2)) < 1e-12


@pytest.mark.parametrize("seed", range(6))
def test_rounds_bounds(seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.softmax(torch.randn(50, generator=g, dtype=DT) * 2.0, 0)
    p2 = float((p ** 2).sum())
    for tau in [0.5, 0.7, 0.9, 0.95]:
        m_tau = next(m for m in range(1, 100_000)
                     if expected_coverage(p, m) >= tau)
        assert m_tau >= math.log(1 / (1 - tau)) / math.log(1 / (1 - p2)) - 1e-9
        for eps in [0.005, 0.01, 0.02, 0.05, 0.1]:
            t_le = float(p[p <= eps].sum())
            t_lt = float(p[p < eps].sum())
            if t_le > 1 - tau:
                lb = math.log(t_le / (1 - tau)) / math.log(1 / (1 - eps))
                assert m_tau >= lb - 1e-9
                assert lb >= (1 - eps) / eps * math.log(t_le / (1 - tau)) - 1e-9
            if t_lt < 1 - tau:
                ub = math.ceil(math.log((1 - t_lt) / (1 - tau - t_lt)) / eps)
                assert m_tau <= ub


def test_topm_bias_bound_and_truncation():
    W, X = make_model(seed=3)
    P, V = probs(W, X), v_vectors(W, X)
    F = fisher(P, V)
    G = float(V.norm(dim=-1).max())
    for m in range(1, W.shape[0] + 1):
        top = [set(torch.argsort(P[t], descending=True)[:m].tolist())
               for t in range(X.shape[0])]
        Fs = fisher(P, V, top)
        Mbar = float(sum(P[t, list(top[t])].sum() for t in range(len(top)))
                     / len(top))
        eig = torch.linalg.eigvalsh(F - Fs)
        assert float(eig.min()) >= -1e-12
        assert float(eig.max()) <= (1 - Mbar) * G ** 2 + 1e-12


# ------------------------------------------------------------ (b) head-tail
def test_rademacher_lemma_exact():
    g = torch.Generator().manual_seed(7)
    a = torch.randn(5, 4, generator=g, dtype=DT)
    mean, sq, A = rademacher_exact(a)
    assert torch.allclose(mean, A, atol=1e-12)
    norms = (a ** 2).sum(1)
    gram = a @ a.T
    off = ~torch.eye(5, dtype=torch.bool)
    formula = float((torch.outer(norms, norms) + gram ** 2)[off].sum())
    assert abs(float(sq) - formula) < 1e-10
    assert float(sq) <= 2 * float(norms.sum()) ** 2 + 1e-12


def head_tail_parts(P, V, m):
    """Head vectors per probe j<m and, per position, tail tokens/probs."""
    N = P.shape[0]
    order = torch.argsort(P, dim=1, descending=True)
    head_a = []
    for j in range(m - 1):
        rows = [math.sqrt(float(P[t, order[t, j]]) / N) * V[t, order[t, j]]
                for t in range(N)]
        head_a.append(torch.stack(rows))
    tails, r = [], []
    for t in range(N):
        R = order[t, m - 1:].tolist()
        rt = float(P[t, R].sum())
        r.append(rt)
        tails.append([(y, float(P[t, y]) / rt) for y in R])
    return head_a, tails, r


@pytest.mark.parametrize("m", [2, 3, 4])
def test_head_tail_unbiased_and_second_moment(m):
    W, X = make_model(n_pos=4, vocab=5, dim=3, seed=11)
    P, V = probs(W, X), v_vectors(W, X)
    N = X.shape[0]
    F = fisher(P, V)
    G = float(V.norm(dim=-1).max())
    head_a, tails, r = head_tail_parts(P, V, m)

    total_mean = torch.zeros_like(F)
    total_err = 0.0
    F_H = torch.zeros_like(F)
    head_bound = 0.0
    for a in head_a:  # head probes: exact over xi
        mean, sq, A = rademacher_exact(a)
        total_mean += mean
        F_H += A
        total_err += float(sq)
        pibar = float((a ** 2).sum()) / G ** 2  # <= mean prob of rank j
        head_bound += 2 * G ** 4 * pibar ** 2

    # tail probe: exact over tau (enumerate all tail tuples) and xi
    F_R = F - F_H
    tail_mean = torch.zeros_like(F)
    tail_sq = 0.0
    for combo in itertools.product(*tails):
        prob = math.prod(pr for _, pr in combo)
        a = torch.stack([math.sqrt(r[t] / N) * V[t, y]
                         for t, (y, _) in enumerate(combo)])
        mean, sq, A = rademacher_exact(a)
        tail_mean += prob * mean
        # E_xi||ww^T - F_R||^2 = E_xi||ww^T - A||^2 + ||A - F_R||^2
        tail_sq += prob * (float(sq) + float(((A - F_R) ** 2).sum()))
    total_mean += tail_mean
    total_err += tail_sq

    assert torch.allclose(tail_mean, F_R, atol=1e-12)
    assert torch.allclose(total_mean, F, atol=1e-12)
    rbar = sum(r) / N
    fine = 2 * G ** 4 * rbar ** 2 + G ** 4 * sum(x * x for x in r) / N ** 2
    assert tail_sq <= fine + 1e-12
    assert fine <= 3 * G ** 4 * rbar ** 2 + 1e-12
    assert total_err <= head_bound + fine + 1e-12

    # Monte Carlo check of the full estimator (mean and error decomposition)
    gen = torch.Generator().manual_seed(m)
    trials = 20_000
    order = torch.argsort(P, dim=1, descending=True)
    acc = torch.zeros_like(F)
    err = 0.0
    for _ in range(trials):
        Fh = torch.zeros_like(F)
        for a in head_a:
            xi = torch.randint(0, 2, (N,), generator=gen).to(DT) * 2 - 1
            w = xi @ a
            Fh += torch.outer(w, w)
        rows = []
        for t in range(N):
            R = order[t, m - 1:]
            k = torch.multinomial(P[t, R] / r[t], 1, generator=gen)
            rows.append(math.sqrt(r[t] / N) * V[t, R[k].item()])
        xi = torch.randint(0, 2, (N,), generator=gen).to(DT) * 2 - 1
        w = xi @ torch.stack(rows)
        Fh += torch.outer(w, w)
        acc += Fh
        err += float(((Fh - F) ** 2).sum())
    assert torch.allclose(acc / trials, F, atol=0.05 * float(F.abs().max()))
    assert abs(err / trials - total_err) < 0.1 * total_err


def test_top_only_is_biased_and_coverage_tracked_targets_FS():
    W, X = make_model(n_pos=3, vocab=5, dim=2, seed=5)
    P, V = probs(W, X), v_vectors(W, X)
    F = fisher(P, V)
    head_a, _, _ = head_tail_parts(P, V, 3)
    F_H = sum(a.T @ a for a in head_a)
    assert float(torch.linalg.eigvalsh(F - F_H).min()) >= -1e-12
    assert float((F - F_H).norm()) > 1e-6  # nonzero bias


# ------------------------------------------------------------ (c) ranking
@pytest.mark.parametrize("seed", range(20))
def test_ranking_lemmas(seed):
    g = torch.Generator().manual_seed(seed)
    n, k = 400, 150
    a = torch.randn(n, generator=g, dtype=DT) * torch.rand(n, generator=g,
                                                            dtype=DT) ** 3
    rho = 0.3
    b = rho * a.abs() * (2 * torch.rand(n, generator=g, dtype=DT) - 1)
    s, s1 = (a + b).abs(), a.abs()
    K = set(torch.topk(s, k).indices.tolist())
    Ka = set(torch.topk(s1, k).indices.tolist())
    eps = float((b - 0).abs().max())
    thr = float(torch.sort(s, descending=True).values[k - 1])
    for i in K ^ Ka:
        assert abs(float(s[i]) - thr) <= 2 * eps + 1e-12
    th_a = float(torch.sort(s1, descending=True).values[k - 1])
    kap = (1 + rho) / (1 - rho)
    for i in Ka - K:
        assert th_a - 1e-12 <= float(s1[i]) <= kap * th_a + 1e-12
    for i in K - Ka:
        assert th_a / kap - 1e-12 <= float(s1[i]) <= th_a + 1e-12
    # gap condition: shrink b below half the gap -> identical masks
    srt = torch.sort(s1, descending=True).values
    gap = float(srt[k - 1] - srt[k])
    b_small = b / (float(b.abs().max()) + 1e-300) * 0.49 * gap
    Ks = set(torch.topk((a + b_small).abs(), k).indices.tolist())
    assert Ks == Ka


# ------------------------------------------------------------ (d) identity
@pytest.mark.parametrize("seed", range(5))
def test_additive_attribution_cross_term(seed):
    W, X = make_model(n_pos=4, vocab=5, dim=3, seed=seed)
    y = torch.randint(0, 5, (4,), generator=torch.Generator().manual_seed(seed))
    theta = W.reshape(-1).clone()

    def loss(th):
        return torch.nn.functional.cross_entropy(X @ th.reshape(5, 3).T, y)

    g = torch.autograd.functional.jacobian(loss, theta)
    H = torch.autograd.functional.hessian(loss, theta)
    gen = torch.Generator().manual_seed(100 + seed)
    d = torch.randn(theta.numel(), generator=gen, dtype=DT)
    for H_use in (H, fisher(probs(W, X), v_vectors(W, X))):
        Hd = H_use @ d
        alpha = -g * d + 0.5 * d * Hd
        alpha_diag = -g * d + 0.5 * torch.diagonal(H_use) * d ** 2
        for _ in range(20):
            z = (torch.rand(theta.numel(), generator=gen) < 0.5).to(DT)
            zb = 1 - z
            delta = -(z * d)
            Q = g @ delta + 0.5 * delta @ H_use @ delta
            A = (z * alpha).sum()
            assert torch.allclose(A - Q, -0.5 * delta @ H_use @ (zb * d),
                                  atol=1e-12)
            assert torch.allclose(A - Q, 0.5 * (z * d) @ H_use @ (zb * d),
                                  atol=1e-12)
            bound = 0.5 * torch.linalg.matrix_norm(H_use, 2) \
                * (z * d).norm() * (zb * d).norm()
            assert abs(float(A - Q)) <= float(bound) + 1e-12
            off = H_use * torch.outer(z * d, z * d)
            off = off.sum() - torch.diagonal(off).sum()
            assert torch.allclose(Q - (z * alpha_diag).sum(), 0.5 * off,
                                  atol=1e-12)
        # z = 1: attribution equals the Taylor change exactly
        assert torch.allclose(alpha.sum(), g @ (-d) + 0.5 * d @ H_use @ d,
                              atol=1e-12)
