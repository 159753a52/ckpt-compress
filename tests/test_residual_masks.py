import unittest
from unittest import mock

import numpy as np
import pytest
import torch

from experiments.lib.residual_masks import (
    _ordered_score_key_chunks,
    apply_layer_mask,
    apply_mask_from_device_states,
    cache_mask_states_on_device,
    exact_keep_mask,
    exact_keep_mask_from_order,
    global_mask,
    layer_mask_at_count,
    layer_masks,
    layer_rates,
    layer_score_orders,
    mask_metrics,
    mask_overlap,
    restore_with_mask,
)


def _valid_mask_pair() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return (
        {"weight": torch.tensor([True, False])},
        {"weight": torch.tensor([False, True])},
    )


def _invalid_overlap_case(case: str) -> tuple[object, object]:
    left, right = _valid_mask_pair()
    if case == "left-not-mapping":
        return [], right
    if case == "right-not-mapping":
        return left, []
    if case == "left-empty":
        return {}, right
    if case == "right-empty":
        return left, {}
    if case == "left-non-string-key":
        return {1: left["weight"]}, right
    if case == "right-non-string-key":
        return left, {1: right["weight"]}
    if case == "key-mismatch":
        return left, {"other": right["weight"]}
    if case == "left-non-tensor":
        return {"weight": [True, False]}, right
    if case == "right-non-tensor":
        return left, {"weight": [False, True]}
    if case == "left-non-bool":
        return {"weight": torch.ones(2)}, right
    if case == "right-non-bool":
        return left, {"weight": torch.ones(2)}
    if case == "left-empty-tensor":
        return (
            {"weight": torch.empty(0, dtype=torch.bool)},
            {"weight": torch.empty(0, dtype=torch.bool)},
        )
    if case == "right-empty-tensor":
        return (
            {"weight": torch.empty(0, dtype=torch.bool)},
            {"weight": torch.empty(0, dtype=torch.bool)},
        )
    if case == "left-sparse":
        sparse = torch.sparse_coo_tensor(
            torch.tensor([[0]]),
            torch.tensor([True]),
            size=(2,),
        )
        return {"weight": sparse}, right
    if case == "right-sparse":
        sparse = torch.sparse_coo_tensor(
            torch.tensor([[0]]),
            torch.tensor([True]),
            size=(2,),
        )
        return left, {"weight": sparse}
    if case == "left-quantized":
        quantized = torch.quantize_per_tensor(
            torch.tensor([1.0, 0.0]), scale=1.0, zero_point=0, dtype=torch.qint8
        )
        return {"weight": quantized}, right
    if case == "right-quantized":
        quantized = torch.quantize_per_tensor(
            torch.tensor([1.0, 0.0]), scale=1.0, zero_point=0, dtype=torch.qint8
        )
        return left, {"weight": quantized}
    if case == "left-nested":
        nested = torch.nested.nested_tensor([torch.tensor([1.0])])
        return {"weight": nested}, right
    if case == "right-nested":
        nested = torch.nested.nested_tensor([torch.tensor([1.0])])
        return left, {"weight": nested}
    if case == "shape-mismatch":
        return left, {"weight": torch.tensor([False, True, False])}
    if case == "device-mismatch":
        return left, {"weight": torch.ones(2, dtype=torch.bool, device="meta")}
    raise AssertionError(f"unknown overlap case: {case}")


@pytest.mark.parametrize(
    ("case", "error_type", "message"),
    [
        ("left-not-mapping", TypeError, "left masks must be a mapping"),
        ("right-not-mapping", TypeError, "right masks must be a mapping"),
        ("left-empty", ValueError, "left masks must contain at least one"),
        ("right-empty", ValueError, "right masks must contain at least one"),
        ("left-non-string-key", TypeError, "left masks keys must be strings"),
        ("right-non-string-key", TypeError, "right masks keys must be strings"),
        ("key-mismatch", ValueError, "keys must match"),
        ("left-non-tensor", TypeError, "torch.Tensor"),
        ("right-non-tensor", TypeError, "torch.Tensor"),
        ("left-non-bool", TypeError, "bool dtype"),
        ("right-non-bool", TypeError, "bool dtype"),
        ("left-empty-tensor", ValueError, "must contain at least one element"),
        ("right-empty-tensor", ValueError, "must contain at least one element"),
        ("left-sparse", TypeError, "dense strided"),
        ("right-sparse", TypeError, "dense strided"),
        ("left-quantized", TypeError, "must not be quantized"),
        ("right-quantized", TypeError, "must not be quantized"),
        ("left-nested", TypeError, "must not be nested"),
        ("right-nested", TypeError, "must not be nested"),
        ("shape-mismatch", ValueError, "shape.*must match"),
        ("device-mismatch", TypeError, "materialized device"),
    ],
)
def test_mask_overlap_rejects_invalid_contract(
    case: str,
    error_type: type[Exception],
    message: str,
) -> None:
    left, right = _invalid_overlap_case(case)
    with pytest.raises(error_type, match=message):
        mask_overlap(left, right)  # type: ignore[arg-type]


def _invalid_metrics_case(case: str) -> tuple[object, object]:
    masks, _ = _valid_mask_pair()
    scores = {"weight": torch.tensor([1.0, 2.0])}
    if case == "masks-not-mapping":
        return [], scores
    if case == "scores-not-mapping":
        return masks, []
    if case == "masks-empty":
        return {}, scores
    if case == "scores-empty":
        return masks, {}
    if case == "masks-non-string-key":
        return {1: masks["weight"]}, scores
    if case == "scores-non-string-key":
        return masks, {1: scores["weight"]}
    if case == "key-mismatch":
        return masks, {"other": scores["weight"]}
    if case == "mask-non-tensor":
        return {"weight": [True, False]}, scores
    if case == "score-non-tensor":
        return masks, {"weight": [1.0, 2.0]}
    if case == "mask-non-bool":
        return {"weight": torch.ones(2)}, scores
    if case == "mask-empty-tensor":
        return {"weight": torch.empty(0, dtype=torch.bool)}, scores
    if case == "score-empty-tensor":
        return masks, {"weight": torch.empty(0)}
    if case == "score-integer":
        return masks, {"weight": torch.tensor([1, 2])}
    if case == "score-complex":
        return masks, {"weight": torch.tensor([1.0 + 0.0j, 2.0])}
    if case == "score-non-finite":
        return masks, {"weight": torch.tensor([1.0, float("inf")])}
    if case == "score-sparse":
        sparse = torch.sparse_coo_tensor(
            torch.tensor([[0]]),
            torch.tensor([1.0]),
            size=(2,),
        )
        return masks, {"weight": sparse}
    if case == "score-quantized":
        quantized = torch.quantize_per_tensor(
            torch.tensor([1.0, 2.0]), scale=1.0, zero_point=0, dtype=torch.qint8
        )
        return masks, {"weight": quantized}
    if case == "score-nested":
        nested = torch.nested.nested_tensor([torch.tensor([1.0])])
        return masks, {"weight": nested}
    if case == "shape-mismatch":
        return masks, {"weight": torch.ones(3)}
    if case == "device-mismatch":
        return {"weight": torch.ones(2, dtype=torch.bool, device="meta")}, scores
    raise AssertionError(f"unknown metrics case: {case}")


@pytest.mark.parametrize(
    ("case", "error_type", "message"),
    [
        ("masks-not-mapping", TypeError, "masks must be a mapping"),
        ("scores-not-mapping", TypeError, "taylor_scores must be a mapping"),
        ("masks-empty", ValueError, "masks must contain at least one"),
        ("scores-empty", ValueError, "taylor_scores must contain at least one"),
        ("masks-non-string-key", TypeError, "masks keys must be strings"),
        ("scores-non-string-key", TypeError, "taylor_scores keys must be strings"),
        ("key-mismatch", ValueError, "keys must match"),
        ("mask-non-tensor", TypeError, "torch.Tensor"),
        ("score-non-tensor", TypeError, "torch.Tensor"),
        ("mask-non-bool", TypeError, "bool dtype"),
        ("mask-empty-tensor", ValueError, "must contain at least one element"),
        ("score-empty-tensor", ValueError, "must contain at least one element"),
        ("score-integer", TypeError, "real floating point"),
        ("score-complex", TypeError, "real floating point"),
        ("score-non-finite", ValueError, "finite"),
        ("score-sparse", TypeError, "dense strided"),
        ("score-quantized", TypeError, "must not be quantized"),
        ("score-nested", TypeError, "must not be nested"),
        ("shape-mismatch", ValueError, "shape.*must match"),
        ("device-mismatch", TypeError, "materialized device"),
    ],
)
def test_mask_metrics_rejects_invalid_contract(
    case: str,
    error_type: type[Exception],
    message: str,
) -> None:
    masks, scores = _invalid_metrics_case(case)
    with pytest.raises(error_type, match=message):
        mask_metrics(masks, scores)  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("operation", ["overlap", "metrics"])
def test_mask_contract_rejects_mismatched_materialized_devices(operation: str) -> None:
    if operation == "overlap":
        left, _ = _valid_mask_pair()
        right = {"weight": torch.tensor([False, True], device="cuda")}
        with pytest.raises(ValueError, match="device.*must match"):
            mask_overlap(left, right)
    else:
        masks, _ = _valid_mask_pair()
        scores = {"weight": torch.tensor([1.0, 2.0], device="cuda")}
        with pytest.raises(ValueError, match="device.*must match"):
            mask_metrics(masks, scores)


class TestResidualMasks(unittest.TestCase):
    def test_mask_metrics_and_overlap_allow_mixed_empty_and_non_empty_entries(self) -> None:
        masks = {
            "empty": torch.empty(0, dtype=torch.bool),
            "weight": torch.tensor([False, True]),
        }
        scores = {
            "empty": torch.empty(0),
            "weight": torch.tensor([3.0, 4.0]),
        }
        other_masks = {
            "empty": torch.empty(0, dtype=torch.bool),
            "weight": torch.tensor([True, True]),
        }

        self.assertEqual(mask_metrics(masks, scores), {"pruned": 1, "proxy_cost": 3.0})
        self.assertEqual(
            mask_overlap(masks, other_masks),
            {"pruned_jaccard": 0.0, "mask_disagreements": 1},
        )

    def test_mask_overlap_keeps_zero_jaccard_for_non_empty_all_keep_masks(self) -> None:
        masks = {
            "left": torch.ones(2, dtype=torch.bool),
            "right": torch.ones(1, dtype=torch.bool),
        }

        overlap = mask_overlap(masks, masks)

        self.assertEqual(overlap, {"pruned_jaccard": 0.0, "mask_disagreements": 0})

    def test_exact_keep_mask_handles_ties_and_boundaries(self) -> None:
        scores = torch.tensor([1.0, 1.0, 2.0])

        self.assertTrue(torch.equal(exact_keep_mask(scores, 0), torch.ones(3).bool()))
        self.assertTrue(
            torch.equal(
                exact_keep_mask(scores, 2),
                torch.tensor([False, False, True]),
            )
        )
        self.assertTrue(torch.equal(exact_keep_mask(scores, 3), torch.zeros(3).bool()))
        with self.assertRaisesRegex(ValueError, "prune_count must be in"):
            exact_keep_mask(scores, 4)

    def test_exact_keep_mask_rejects_invalid_counts_and_scores(self) -> None:
        for invalid_count in (True, 1.5, float("nan")):
            with (
                self.subTest(prune_count=invalid_count),
                self.assertRaisesRegex(ValueError, "prune_count must be an integer"),
            ):
                exact_keep_mask(torch.ones(3), invalid_count)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            exact_keep_mask(torch.tensor([1.0, float("nan")]), 1)
        self.assertTrue(
            torch.equal(
                exact_keep_mask(torch.tensor([1.0, 2.0]), 1.0),
                torch.tensor([False, True]),
            )
        )

    def test_order_mask_uses_the_same_prune_count_boundaries(self) -> None:
        order = torch.tensor([2, 0, 1])

        self.assertTrue(
            torch.equal(
                exact_keep_mask_from_order(order, 2),
                torch.tensor([False, True, False]),
            )
        )
        for invalid_count in (-1, 4):
            with (
                self.subTest(prune_count=invalid_count),
                self.assertRaisesRegex(ValueError, "prune_count must be in"),
            ):
                exact_keep_mask_from_order(order, invalid_count)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            exact_keep_mask_from_order(order.reshape(1, -1), 1)

    def test_cached_score_orders_match_direct_masks_with_ties(self) -> None:
        layers = [["left", "right"], ["last"]]
        scores = {
            "left": torch.tensor([3.0, 1.0, 1.0]),
            "right": torch.tensor([2.0, 1.0]),
            "last": torch.tensor([4.0, 2.0, 2.0, 3.0]),
        }
        counts = [3, 2]

        direct = layer_masks(layers, scores, counts)
        cached = layer_masks(
            layers,
            scores,
            counts,
            layer_score_orders(layers, scores),
        )

        self.assertEqual(direct.keys(), cached.keys())
        for name in direct:
            self.assertTrue(torch.equal(direct[name], cached[name]))
        for names, expected_count in zip(layers, counts):
            actual_count = sum((~direct[name]).sum().item() for name in names)
            self.assertEqual(actual_count, expected_count)
        self.assertEqual(layer_rates(layers, direct), [0.6, 0.5])

    def test_layer_masks_reject_empty_layers_and_mismatched_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            layer_score_orders([[]], {})
        with self.assertRaisesRegex(ValueError, "at least one score"):
            layer_score_orders([["empty"]], {"empty": torch.empty(0)})
        with self.assertRaisesRegex(ValueError, "Prune counts must match"):
            layer_masks([["scores"]], {"scores": torch.ones(2)}, [])
        with self.assertRaisesRegex(ValueError, "Score orders must match"):
            layer_masks([["scores"]], {"scores": torch.ones(2)}, [1], [])
        with self.assertRaisesRegex(ValueError, "wrong number"):
            layer_masks(
                [["scores"]],
                {"scores": torch.ones(2)},
                [1],
                [torch.tensor([0])],
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            layer_mask_at_count([], {}, 0)
        with self.assertRaisesRegex(ValueError, "at least one score"):
            layer_mask_at_count(["empty"], {"empty": torch.empty(0)}, 0)

    def test_global_mask_rejects_invalid_tensor_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            global_mask([], {}, 0)
        with self.assertRaisesRegex(ValueError, "unique"):
            global_mask(["score", "score"], {"score": torch.ones(1)}, 1)
        with self.assertRaisesRegex(ValueError, "at least one score"):
            global_mask(["empty"], {"empty": torch.empty(0)}, 0)

    def test_global_mask_preserves_shapes_and_exact_budget(self) -> None:
        scores = {
            "matrix": torch.tensor([[5.0, 1.0], [4.0, 2.0]]),
            "vector": torch.tensor([3.0, 0.5]),
        }

        masks = global_mask(["matrix", "vector"], scores, prune_count=3)

        self.assertEqual(masks["matrix"].shape, scores["matrix"].shape)
        self.assertEqual(masks["vector"].shape, scores["vector"].shape)
        self.assertEqual(sum((~mask).sum().item() for mask in masks.values()), 3)

    def test_global_mask_matches_flat_reference_across_random_inputs(self) -> None:
        generator = torch.Generator().manual_seed(20260813)
        for count in range(12):
            scores = {
                "first": torch.randn(17, generator=generator),
                "second": torch.randn(3, 5, generator=generator),
                "third": torch.randn(9, generator=generator),
            }
            names = list(scores)
            total = sum(value.numel() for value in scores.values())
            expected = layer_mask_at_count(names, scores, count * (total // 11))
            actual = global_mask(names, scores, count * (total // 11))
            for name in names:
                self.assertTrue(torch.equal(actual[name], expected[name]))

    def test_global_mask_matches_stable_reference_for_cross_tensor_ties(self) -> None:
        scores = {
            "first": torch.tensor([float("-inf"), -1.0]),
            "second": torch.tensor([-0.0, 0.0, 1.0, 1.0]),
            "third": torch.tensor([1.0, 2.0]),
        }
        # Infinities remain invalid under both old and bounded-memory paths.
        with self.assertRaisesRegex(ValueError, "finite"):
            global_mask(list(scores), scores, 3)
        scores["first"][0] = -1.0
        for count in range(9):
            expected = layer_mask_at_count(list(scores), scores, count)
            actual = global_mask(list(scores), scores, count)
            for name in scores:
                self.assertTrue(torch.equal(actual[name], expected[name]))

    def test_global_mask_processes_scores_in_bounded_chunks(self) -> None:
        scores = {"weight": torch.arange(23, dtype=torch.float32)}
        with mock.patch("experiments.lib.residual_masks._GLOBAL_SELECTION_CHUNK_ELEMENTS", 5):
            actual = global_mask(["weight"], scores, 11)
        self.assertTrue(
            torch.equal(
                actual["weight"],
                torch.tensor([False] * 11 + [True] * 12),
            )
        )

    def test_score_chunks_bound_noncontiguous_reshape_copies(self) -> None:
        scores = torch.arange(60, dtype=torch.float64).reshape(3, 4, 5).permute(2, 0, 1)
        self.assertFalse(scores.is_contiguous())

        chunks = list(_ordered_score_key_chunks(scores, chunk_elements=7))
        reference = list(_ordered_score_key_chunks(scores.contiguous(), chunk_elements=7))

        self.assertTrue(chunks)
        self.assertLessEqual(max(chunk.numel() for chunk in chunks), 7)
        self.assertTrue(torch.equal(torch.cat(chunks), torch.cat(reference)))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            list(_ordered_score_key_chunks(scores, chunk_elements=0))

    def test_global_mask_matches_independent_stable_numpy_reference(self) -> None:
        generator = torch.Generator().manual_seed(314159)
        for dtype in (torch.float16, torch.float32, torch.float64):
            for prune_count in (0, 1, 13, 29, 39):
                scores = {
                    "left": torch.randn(4, 5, generator=generator, dtype=dtype).t(),
                    "middle": torch.tensor(
                        [-0.0, 0.0, 1.0, 1.0, -1.0, -1.0],
                        dtype=dtype,
                    ),
                    "right": torch.randn(14, generator=generator, dtype=dtype),
                }
                names = list(scores)
                flattened = np.concatenate(
                    [score.detach().float().reshape(-1).numpy() for score in scores.values()]
                )
                order = np.argsort(flattened, kind="stable")
                expected_keep = np.ones(flattened.size, dtype=bool)
                expected_keep[order[:prune_count]] = False

                actual = global_mask(names, scores, prune_count)
                actual_keep = np.concatenate([actual[name].reshape(-1).numpy() for name in names])
                self.assertTrue(np.array_equal(actual_keep, expected_keep))

    def test_mask_metrics_and_overlap(self) -> None:
        scores = {
            "left": torch.tensor([1.0, 2.0, 3.0]),
            "right": torch.tensor([4.0, 5.0]),
        }
        first = {
            "left": torch.tensor([False, True, True]),
            "right": torch.tensor([True, False]),
        }
        second = {
            "left": torch.tensor([False, False, True]),
            "right": torch.tensor([True, True]),
        }

        self.assertEqual(mask_metrics(first, scores), {"pruned": 2, "proxy_cost": 6.0})
        overlap = mask_overlap(first, second)
        self.assertAlmostEqual(overlap["pruned_jaccard"], 1 / 3)
        self.assertEqual(overlap["mask_disagreements"], 2)

    def test_restore_and_partial_apply_have_distinct_state_scope(self) -> None:
        model = torch.nn.BatchNorm1d(2)
        current = {
            "weight": torch.tensor([1.0, 2.0]),
            "bias": torch.tensor([3.0, 4.0]),
            "running_mean": torch.tensor([5.0, 6.0]),
            "running_var": torch.tensor([7.0, 8.0]),
            "num_batches_tracked": torch.tensor(9, dtype=torch.long),
        }
        reference = {
            "weight": torch.tensor([10.0, 20.0]),
            "bias": torch.tensor([30.0, 40.0]),
            "running_mean": torch.tensor([50.0, 60.0]),
            "running_var": torch.tensor([70.0, 80.0]),
            "num_batches_tracked": torch.tensor(90, dtype=torch.long),
        }

        restore_with_mask(
            model,
            current,
            reference,
            {"weight": torch.tensor([True, False])},
            "cpu",
        )
        self.assertTrue(torch.equal(model.weight, torch.tensor([1.0, 20.0])))
        for name in ("bias", "running_mean", "running_var", "num_batches_tracked"):
            self.assertTrue(torch.equal(getattr(model, name), current[name]))

        with torch.no_grad():
            model.bias.fill_(-3.0)
            model.running_mean.fill_(-5.0)

        apply_layer_mask(
            model,
            current,
            reference,
            layer_mask_at_count(
                ["weight"],
                {"weight": torch.tensor([1.0, 2.0])},
                prune_count=1,
            ),
            "cpu",
        )
        self.assertTrue(torch.equal(model.weight, torch.tensor([10.0, 2.0])))
        self.assertTrue(torch.equal(model.bias, torch.full((2,), -3.0)))
        self.assertTrue(torch.equal(model.running_mean, torch.full((2,), -5.0)))
        self.assertTrue(torch.equal(model.running_var, current["running_var"]))
        self.assertTrue(torch.equal(model.num_batches_tracked, current["num_batches_tracked"]))

    def test_device_state_cache_applies_mask_exactly(self) -> None:
        model = torch.nn.Linear(2, 2, bias=False)
        current = {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        reference = {"weight": torch.tensor([[10.0, 20.0], [30.0, 40.0]])}
        current_device, reference_device = cache_mask_states_on_device(
            current, reference, ["weight"], "cpu"
        )
        self.assertNotEqual(current_device["weight"].data_ptr(), current["weight"].data_ptr())
        self.assertNotEqual(
            reference_device["weight"].data_ptr(),
            reference["weight"].data_ptr(),
        )

        apply_mask_from_device_states(
            model,
            current_device,
            reference_device,
            {"weight": torch.tensor([[True, False], [False, True]])},
            "cpu",
        )

        expected = torch.tensor([[1.0, 20.0], [30.0, 4.0]])
        self.assertTrue(torch.equal(model.weight, expected))


if __name__ == "__main__":
    unittest.main()
