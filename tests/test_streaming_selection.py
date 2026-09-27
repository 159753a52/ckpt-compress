import pytest
import torch

from experiments.lib.probe_coverage import select_smallest, unpack_mask
from experiments.lib.streaming_selection import StreamingSelector, Target


def _scores():
    gen = torch.Generator().manual_seed(0)
    a = torch.rand(300, generator=gen)
    a[:40] = 0.0
    b = torch.rand(7, 11, generator=gen) * 1e-3
    b[0, :5] = 0.25
    c = torch.full((50,), 0.25)
    return [a.float(), b.float(), c.float()]


@pytest.mark.parametrize("k", [0, 1, 39, 40, 41, 200, 380, 427])
def test_global_target_matches_reference(k):
    scores = _scores()
    sel = StreamingSelector(3, lambda i: scores[i])
    sel.first_pass()
    out = sel.select([Target("g", [0, 1, 2], k)])["g"]
    ref = select_smallest(scores, k)
    for i, s in enumerate(scores):
        got = unpack_mask(out["packed"][i], s.shape)
        assert torch.equal(got, ref[i])
    assert out["selected"] == k


def test_group_targets_are_independent():
    scores = _scores()
    moments = []
    sel = StreamingSelector(3, lambda i: scores[i])
    sel.first_pass(lambda i, s: moments.append((i, float(s.sum()))))
    assert [m[0] for m in moments] == [0, 1, 2]
    res = sel.select([Target("a", [0], 100), Target("bc", [1, 2], 60), Target("all", [0, 1, 2], 5)])
    ref_a = select_smallest([scores[0]], 100)[0]
    assert torch.equal(unpack_mask(res["a"]["packed"][0], scores[0].shape), ref_a)
    ref_bc = select_smallest([scores[1], scores[2]], 60)
    assert torch.equal(unpack_mask(res["bc"]["packed"][1], scores[1].shape), ref_bc[0])
    assert torch.equal(unpack_mask(res["bc"]["packed"][2], scores[2].shape), ref_bc[1])
    assert res["all"]["selected"] == 5


def test_rejects_negative_scores():
    sel = StreamingSelector(1, lambda i: torch.tensor([-1.0]))
    with pytest.raises(ValueError):
        sel.first_pass()
