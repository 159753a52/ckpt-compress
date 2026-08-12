import unittest
from unittest import mock

import torch

from experiments.lib.distributed_stats import (
    ScoreMoments,
    layer_score_moments,
    merge_score_moments,
    reduce_score_moments,
    score_moments,
)
from experiments.lib.residual_weibull import (
    fit_layer_weibull_mom,
    fit_weibull_from_moments,
    fit_weibull_mom,
)


class TestDistributedStats(unittest.TestCase):
    def test_score_moments_and_layer_assembly(self) -> None:
        moments = score_moments(torch.tensor([0.0, 1.0, 2.0, 3.0]))
        self.assertEqual(moments, ScoreMoments(4, 6.0, 14.0, 1, 3.0))

        layers = layer_score_moments(
            [["left", "right"]],
            {
                "left": torch.tensor([0.0, 1.0]),
                "right": torch.tensor([2.0, 3.0]),
            },
        )
        self.assertEqual(layers, [moments])
        self.assertEqual(
            merge_score_moments(
                [ScoreMoments(2, 1.0, 1.0, 1, 1.0), ScoreMoments(2, 5.0, 13.0, 0, 3.0)]
            ),
            moments,
        )

    def test_score_moments_reject_invalid_values(self) -> None:
        for values in (torch.tensor([-1.0]), torch.tensor([float("nan")])):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    score_moments(values)

    def test_uninitialized_process_group_is_explicit_noop(self) -> None:
        source = [ScoreMoments(2, 3.0, 5.0, 0, 2.0)]
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            reduced, metadata = reduce_score_moments(source)

        self.assertEqual(reduced, source)
        self.assertEqual(metadata["distributed"], False)
        self.assertEqual(metadata["communicated_scalars_per_rank"], 0)

    def test_initialized_process_group_reduces_additive_and_max_stats(self) -> None:
        source = [
            ScoreMoments(2, 3.0, 5.0, 1, 2.0),
            ScoreMoments(1, 4.0, 16.0, 0, 4.0),
        ]
        calls = []

        def fake_all_reduce(tensor, op, group=None):
            calls.append((tensor.shape, op, group))
            if op == torch.distributed.ReduceOp.SUM:
                tensor.mul_(2)
            elif op == torch.distributed.ReduceOp.MAX:
                tensor.add_(1)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_backend", return_value="gloo"),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch("torch.distributed.all_reduce", side_effect=fake_all_reduce),
        ):
            reduced, metadata = reduce_score_moments(source, process_group="paper")

        self.assertEqual(
            reduced,
            [
                ScoreMoments(4, 6.0, 10.0, 2, 3.0),
                ScoreMoments(2, 8.0, 32.0, 0, 5.0),
            ],
        )
        self.assertEqual([call[0] for call in calls], [torch.Size([2, 4]), torch.Size([2])])
        self.assertEqual(metadata["world_size"], 2)
        self.assertEqual(metadata["communicated_scalars_per_rank"], 10)

    def test_weibull_fit_is_identical_for_values_and_moments(self) -> None:
        values = torch.tensor([0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
        direct = fit_weibull_mom(values)
        streamed = fit_weibull_from_moments(score_moments(values))

        self.assertEqual(direct.keys(), streamed.keys())
        for key in direct:
            if isinstance(direct[key], float):
                self.assertAlmostEqual(direct[key], streamed[key], places=12)
            else:
                self.assertEqual(direct[key], streamed[key])

    def test_layer_fit_records_reduction_provenance(self) -> None:
        fits = fit_layer_weibull_mom(
            [["scores"]],
            {"scores": torch.tensor([0.1, 0.2, 0.4, 0.8])},
            distributed=True,
        )
        self.assertEqual(fits[0]["moment_reduction"]["distributed"], False)


if __name__ == "__main__":
    unittest.main()
