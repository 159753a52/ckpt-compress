import pytest
import torch

from experiments.lib.importance_compare.score_comparison import (
    compute_gamma,
    compute_mask_iou,
    compute_rank_correlation,
    compute_relative_l2_error,
)


@pytest.mark.parametrize(
    "comparison",
    [compute_relative_l2_error, compute_rank_correlation],
)
def test_score_comparison_rejects_parameter_key_mismatch(comparison) -> None:
    reference = {"shared": torch.ones(2), "missing": torch.ones(1)}
    approximate = {"shared": torch.ones(2), "unexpected": torch.ones(1)}

    with pytest.raises(ValueError, match="identical parameter keys"):
        comparison(reference, approximate)


def test_mask_comparison_rejects_parameter_key_mismatch() -> None:
    with pytest.raises(ValueError, match="identical parameter keys"):
        compute_mask_iou(
            {"reference": torch.ones(2)},
            {"approximation": torch.ones(2)},
            0.5,
        )


def test_relative_l2_does_not_hide_nonzero_error_against_zero_reference() -> None:
    with pytest.raises(ValueError, match="undefined for zero reference"):
        compute_relative_l2_error(
            {"weight": torch.zeros(2)},
            {"weight": torch.tensor([0.0, 1.0])},
        )


def test_two_point_nonconstant_correlations_are_computed() -> None:
    correlations = compute_rank_correlation(
        {"weight": torch.tensor([1.0, 2.0])},
        {"weight": torch.tensor([2.0, 1.0])},
    )

    assert correlations["weight"] == pytest.approx({"spearman": -1.0, "pearson": -1.0})
    assert correlations["__global__"] == pytest.approx({"spearman": -1.0, "pearson": -1.0})


def test_nonfinite_scipy_correlation_fails_closed(monkeypatch) -> None:
    from experiments.lib.importance_compare import score_comparison

    monkeypatch.setattr(
        score_comparison.scipy_stats,
        "spearmanr",
        lambda *_: type("Result", (), {"statistic": float("nan")})(),
    )
    with pytest.raises(ValueError, match="non-finite"):
        compute_rank_correlation(
            {"weight": torch.tensor([1.0, 2.0, 3.0])},
            {"weight": torch.tensor([3.0, 1.0, 2.0])},
        )


def test_mask_iou_uses_an_exact_global_budget_under_ties() -> None:
    scores = {"layer": torch.ones(5)}

    assert compute_mask_iou(scores, scores, 0.4) == {
        "__global__": 1.0,
        "__agreement__": 1.0,
    }


def test_comparison_validates_shapes_and_prune_ratio() -> None:
    with pytest.raises(ValueError, match="shapes"):
        compute_mask_iou(
            {"layer": torch.ones(1, 2)},
            {"layer": torch.ones(2)},
            0.5,
        )
    with pytest.raises(ValueError, match="prune_ratio"):
        compute_mask_iou({"layer": torch.ones(2)}, {"layer": torch.ones(2)}, 1.1)


def test_constant_rank_correlation_does_not_hide_one_sided_degeneracy() -> None:
    same = compute_rank_correlation(
        {"layer": torch.ones(3)},
        {"layer": torch.ones(3)},
    )
    different = compute_rank_correlation(
        {"layer": torch.ones(3)},
        {"layer": torch.full((3,), 2.0)},
    )
    varying = compute_rank_correlation(
        {"layer": torch.ones(3)},
        {"layer": torch.tensor([1.0, 2.0, 3.0])},
    )

    assert same["layer"] == {"spearman": 1.0, "pearson": 1.0}
    assert different["layer"] == {"spearman": 0.0, "pearson": 0.0}
    assert varying["layer"] == {"spearman": 0.0, "pearson": 0.0}


def test_gamma_matches_vit_constructor_token_counts() -> None:
    assert compute_gamma("vit-l-32", 128)["T"] == 145
    assert compute_gamma("vit-b-16", 128)["T"] == 197

    with pytest.raises(ValueError, match="seq_length"):
        compute_gamma("gpt2-small", 0)


def test_summary_requires_global_and_layer_correlations() -> None:
    from experiments.lib.importance_compare.score_comparison import summarize_comparison

    gamma = {"d": 1, "T": 1, "n_ell": 1, "gamma_pct": 1.0}
    with pytest.raises(KeyError, match="__global__"):
        summarize_comparison({}, {}, {}, gamma)
    with pytest.raises(KeyError, match="layer"):
        summarize_comparison(
            {"__global__": 0.1, "layer": 0.2},
            {"__global__": {"spearman": 1.0, "pearson": 1.0}},
            {},
            gamma,
        )
