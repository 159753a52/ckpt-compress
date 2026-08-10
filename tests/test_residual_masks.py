import unittest

import torch

from experiments.lib.residual_masks import (
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


class TestResidualMasks(unittest.TestCase):
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
            with self.subTest(prune_count=invalid_count), self.assertRaisesRegex(
                ValueError, "prune_count must be an integer"
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
            with self.subTest(prune_count=invalid_count), self.assertRaisesRegex(
                ValueError, "prune_count must be in"
            ):
                exact_keep_mask_from_order(order, invalid_count)

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

    def test_global_mask_preserves_shapes_and_exact_budget(self) -> None:
        scores = {
            "matrix": torch.tensor([[5.0, 1.0], [4.0, 2.0]]),
            "vector": torch.tensor([3.0, 0.5]),
        }

        masks = global_mask(["matrix", "vector"], scores, prune_count=3)

        self.assertEqual(masks["matrix"].shape, scores["matrix"].shape)
        self.assertEqual(masks["vector"].shape, scores["vector"].shape)
        self.assertEqual(sum((~mask).sum().item() for mask in masks.values()), 3)

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
        self.assertTrue(
            torch.equal(model.num_batches_tracked, current["num_batches_tracked"])
        )

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
