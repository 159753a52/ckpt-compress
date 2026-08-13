import unittest

import torch

from experiments.scripts.finetune.train_nlp import _build_lr_scheduler


class TestTrainNlpScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.optimizer = torch.optim.SGD([self.parameter], lr=0.1)

    def test_linear_schedule_uses_optimizer_steps_including_tail_batch(self) -> None:
        scheduler, step_mode = _build_lr_scheduler(
            self.optimizer,
            "linear",
            epochs=2,
            train_batches=3,
            gradient_accumulation_steps=2,
            warmup_steps=1,
        )

        self.assertIsNotNone(scheduler)
        self.assertEqual(step_mode, "optimizer_step")
        assert scheduler is not None
        for _ in range(4):
            self.optimizer.step()
            scheduler.step()
        self.assertEqual(scheduler.last_epoch, 4)
        self.assertEqual(self.optimizer.param_groups[0]["lr"], 0.0)

    def test_cosine_schedule_remains_epoch_based(self) -> None:
        scheduler, step_mode = _build_lr_scheduler(
            self.optimizer,
            "cosine",
            epochs=3,
            train_batches=10,
            gradient_accumulation_steps=4,
            warmup_steps=0,
        )

        self.assertEqual(step_mode, "epoch")
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(scheduler.T_max, 3)

    def test_linear_schedule_rejects_impossible_warmup(self) -> None:
        with self.assertRaisesRegex(ValueError, "total optimizer steps"):
            _build_lr_scheduler(
                self.optimizer,
                "linear",
                epochs=2,
                train_batches=3,
                gradient_accumulation_steps=2,
                warmup_steps=4,
            )

    def test_scheduler_builder_rejects_invalid_step_counts(self) -> None:
        invalid_arguments = (
            ({"epochs": 0}, "epochs"),
            ({"train_batches": 0}, "training data"),
            ({"gradient_accumulation_steps": 0}, "gradient_accumulation_steps"),
            ({"warmup_steps": -1}, "warmup_steps"),
        )
        defaults = {
            "epochs": 2,
            "train_batches": 3,
            "gradient_accumulation_steps": 2,
            "warmup_steps": 1,
        }

        for overrides, message in invalid_arguments:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                _build_lr_scheduler(
                    self.optimizer,
                    "linear",
                    **{**defaults, **overrides},
                )


if __name__ == "__main__":
    unittest.main()
