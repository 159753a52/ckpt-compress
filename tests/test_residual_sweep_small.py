import math

import torch

from experiments.lib.residual_masks import global_mask
from experiments.scripts.run_residual_sweep_small import global_keep_masks


def test_global_keep_masks_match_library_with_ties():
    gen = torch.Generator().manual_seed(0)
    scores = {
        "a": torch.rand(37, 11, generator=gen),
        "b": torch.rand(5, 64, generator=gen),
        "c": torch.rand(3, 3, generator=gen),
    }
    # exact zeros and repeated values exercise threshold tie handling
    scores["a"][:4] = 0.0
    scores["b"][1, :20] = 0.25
    scores["c"][0, 0] = -0.0
    names = list(scores)
    total = sum(v.numel() for v in scores.values())
    sparsities = [0.0, 0.01, 0.1, 0.5, 0.9, 0.995, 1.0]
    for sp, keep, _ in global_keep_masks(scores, names, sparsities, "cpu"):
        k = math.floor(sp * total)
        ref = global_mask(names, scores, k)
        for n in names:
            assert torch.equal(ref[n], keep[n]), (sp, n)
        assert sum(int((~keep[n]).sum()) for n in names) == k
