import unittest
from unittest import mock

import torch
import torch.nn as nn

from experiments.lib.distributed_stats import reduce_candidate_scalars
from experiments.lib.residual_tp import (
    TPConfig,
    compute_tp_local_projections,
    evaluate_tp_candidates_streamed,
    infer_tp_split_dim,
    shard_residual_scope_tp,
    shard_tensor_tp,
    unshard_tensor_tp,
)


class TestResidualTPScoring(unittest.TestCase):
    def test_shard_and_unshard_tensor_roundtrip(self) -> None:
        tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        
        # Test split along dim 0 (4 rows -> two 2x6 shards)
        shard_0_d0 = shard_tensor_tp(tensor, dim=0, rank=0, world_size=2)
        shard_1_d0 = shard_tensor_tp(tensor, dim=0, rank=1, world_size=2)
        self.assertEqual(shard_0_d0.shape, torch.Size([2, 6]))
        self.assertEqual(shard_1_d0.shape, torch.Size([2, 6]))
        reconstructed_d0 = unshard_tensor_tp([shard_0_d0, shard_1_d0], dim=0)
        self.assertTrue(torch.equal(tensor, reconstructed_d0))

        # Test split along dim 1 (6 cols -> two 4x3 shards)
        shard_0_d1 = shard_tensor_tp(tensor, dim=1, rank=0, world_size=2)
        shard_1_d1 = shard_tensor_tp(tensor, dim=1, rank=1, world_size=2)
        self.assertEqual(shard_0_d1.shape, torch.Size([4, 3]))
        self.assertEqual(shard_1_d1.shape, torch.Size([4, 3]))
        reconstructed_d1 = unshard_tensor_tp([shard_0_d1, shard_1_d1], dim=1)
        self.assertTrue(torch.equal(tensor, reconstructed_d1))

    def test_shard_tensor_validation(self) -> None:
        tensor = torch.zeros(5, 5)
        with self.assertRaises(ValueError):
            shard_tensor_tp(tensor, dim=0, rank=0, world_size=2)
        with self.assertRaises(ValueError):
            shard_tensor_tp(tensor, dim=0, rank=2, world_size=2)

    def test_infer_tp_split_dim(self) -> None:
        w_2d = torch.zeros(10, 10)
        w_1d = torch.zeros(10)

        # Conv1D patterns
        self.assertEqual(infer_tp_split_dim("transformer.h.0.attn.c_attn.weight", w_2d), 1)
        self.assertEqual(infer_tp_split_dim("transformer.h.0.mlp.c_fc.weight", w_2d), 1)
        self.assertEqual(infer_tp_split_dim("transformer.h.0.attn.c_proj.weight", w_2d), 0)
        self.assertEqual(infer_tp_split_dim("transformer.h.0.mlp.c_proj.weight", w_2d), 0)

        # Standard Linear patterns
        self.assertEqual(infer_tp_split_dim("model.layers.0.mlp.gate_proj.weight", w_2d), 0)
        self.assertEqual(infer_tp_split_dim("model.layers.0.mlp.down_proj.weight", w_2d), 1)

        # Replicated / 1D patterns
        self.assertIsNone(infer_tp_split_dim("transformer.h.0.ln_1.weight", w_1d))
        self.assertIsNone(infer_tp_split_dim("transformer.wte.weight", w_2d))

    def test_reduce_candidate_scalars(self) -> None:
        # Non-distributed fallback
        scores = [1.5, 2.5, 0.5]
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            reduced, meta = reduce_candidate_scalars(scores)
        self.assertTrue(torch.equal(reduced, torch.tensor([1.5, 2.5, 0.5], dtype=torch.float64)))
        self.assertFalse(meta["distributed"])

        # Distributed all-reduce mock
        def fake_all_reduce(tensor, op, group=None):
            tensor.mul_(2.0)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_backend", return_value="gloo"),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch("torch.distributed.all_reduce", side_effect=fake_all_reduce),
        ):
            reduced, meta = reduce_candidate_scalars(scores)
        self.assertTrue(torch.equal(reduced, torch.tensor([3.0, 5.0, 1.0], dtype=torch.float64)))
        self.assertTrue(meta["distributed"])
        self.assertEqual(meta["communicated_scalars_per_rank"], 3)
        self.assertEqual(meta["communicated_bytes_per_rank"], 24)

    def test_local_candidate_projection_decomposition(self) -> None:
        # Verify that sum of rank 0 and rank 1 local projections exactly matches full projection
        torch.manual_seed(42)
        full_grad = torch.randn(4, 6)
        full_delta = torch.randn(4, 6)

        # Candidate masks: mask A keeps top half, mask B keeps all, mask C keeps none
        mask_a = torch.zeros(4, 6, dtype=torch.bool)
        mask_a[:2, :] = True
        mask_b = torch.ones(4, 6, dtype=torch.bool)
        mask_c = torch.zeros(4, 6, dtype=torch.bool)

        candidate_masks_full = {
            "cand_a": {"layer": mask_a},
            "cand_b": {"layer": mask_b},
            "cand_c": {"layer": mask_c},
        }

        # Full score
        full_scores = compute_tp_local_projections(
            {"layer": full_grad},
            {"layer": full_delta},
            candidate_masks_full,
        )

        # Split along dim 0 (e.g. ColParallel Linear)
        grad_r0 = shard_tensor_tp(full_grad, dim=0, rank=0, world_size=2)
        grad_r1 = shard_tensor_tp(full_grad, dim=0, rank=1, world_size=2)
        delta_r0 = shard_tensor_tp(full_delta, dim=0, rank=0, world_size=2)
        delta_r1 = shard_tensor_tp(full_delta, dim=0, rank=1, world_size=2)

        masks_r0 = {
            k: {"layer": shard_tensor_tp(v["layer"], dim=0, rank=0, world_size=2)}
            for k, v in candidate_masks_full.items()
        }
        masks_r1 = {
            k: {"layer": shard_tensor_tp(v["layer"], dim=0, rank=1, world_size=2)}
            for k, v in candidate_masks_full.items()
        }

        scores_r0 = compute_tp_local_projections({"layer": grad_r0}, {"layer": delta_r0}, masks_r0)
        scores_r1 = compute_tp_local_projections({"layer": grad_r1}, {"layer": delta_r1}, masks_r1)

        summed_scores = scores_r0 + scores_r1
        self.assertTrue(torch.allclose(full_scores, summed_scores, atol=1e-7))

    def test_evaluate_tp_candidates_streamed(self) -> None:
        torch.manual_seed(42)
        linear = nn.Linear(4, 4, bias=False)
        delta = {"weight": torch.randn(4, 4)}
        
        # Candidate 0: keep all (0 damage)
        # Candidate 1: keep none (large damage)
        candidate_masks = {
            "keep_all": {"weight": torch.ones(4, 4, dtype=torch.bool)},
            "keep_none": {"weight": torch.zeros(4, 4, dtype=torch.bool)},
        }

        x = torch.randn(2, 4)
        def probe_loss_fn():
            out = linear(x)
            return (out ** 2).sum()

        cfg = TPConfig(rank=0, world_size=1, device="cpu")
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            winning_idx, global_scores, meta = evaluate_tp_candidates_streamed(
                model=linear,
                probe_loss_fn=probe_loss_fn,
                local_delta=delta,
                candidate_masks=candidate_masks,
                tp_config=cfg,
            )

        self.assertEqual(winning_idx, 0)
        self.assertEqual(meta["winning_candidate"], "keep_all")
        self.assertAlmostEqual(global_scores[0].item(), 0.0, places=6)
        self.assertGreater(global_scores[1].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
