import unittest
from tempfile import TemporaryDirectory
from unittest import mock

import torch
import torch.nn as nn

from dacp.utils.trainer import BaseTrainer


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.projection = nn.Linear(4, 8)

    def forward(self, inputs):
        return type("Output", (), {"logits": self.projection(self.embedding(inputs))})()


class _MaskRecordingLM(_TinyLM):
    def __init__(self):
        super().__init__()
        self.seen_attention_masks = []

    def forward(self, input_ids, attention_mask=None):
        self.seen_attention_masks.append(attention_mask)
        return super().forward(input_ids)


def _make_trainer(
    model,
    train_loader,
    val_loader,
    optimizer=None,
    accumulation=1,
    scheduler=None,
    scheduler_step_mode="epoch",
    checkpoint_dir="/tmp/ckpt-compress-trainer-tests",
):
    if optimizer is None:
        optimizer = _CountingSGD(model.parameters(), lr=0.1)
    return BaseTrainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        nn.CrossEntropyLoss(),
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        scheduler=scheduler,
        gradient_accumulation_steps=accumulation,
        scheduler_step_mode=scheduler_step_mode,
    )


class TestBaseTrainer(unittest.TestCase):
    def test_load_checkpoint_validates_metadata_before_mutating_trainer(self) -> None:
        model = nn.Linear(1, 2)
        trainer = _make_trainer(model, [], [])
        original_model = {name: value.clone() for name, value in model.state_dict().items()}
        valid = {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "history": {"train_loss": [], "val_loss": [], "val_acc": []},
        }
        malformed = (
            ({**valid, "epoch": True}, ValueError, "epoch"),
            ({**valid, "model_state_dict": None}, TypeError, "state mappings"),
            ({**valid, "history": {"train_loss": []}}, ValueError, "history schema"),
            (
                {**valid, "history": {"train_loss": [float("nan")], "val_loss": [], "val_acc": []}},
                ValueError,
                "finite numbers",
            ),
        )

        for payload, error_type, message in malformed:
            with (
                self.subTest(payload=payload),
                mock.patch("dacp.utils.trainer.torch.load", return_value=payload),
                self.assertRaisesRegex(error_type, message),
            ):
                trainer.load_checkpoint("unused.pt")
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, original_model[name])

    def test_rejects_non_positive_accumulation(self) -> None:
        model = nn.Linear(2, 2)
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
                    _make_trainer(model, [], [], accumulation=value)

    def test_rejects_unknown_scheduler_step_mode(self) -> None:
        model = nn.Linear(2, 2)

        with self.assertRaisesRegex(ValueError, "scheduler_step_mode"):
            _make_trainer(model, [], [], scheduler_step_mode="batch")

    def test_empty_loaders_raise_explicit_contract_errors(self) -> None:
        model = nn.Linear(1, 2)
        trainer = _make_trainer(model, [], [])

        with self.assertRaisesRegex(ValueError, "train_loader"):
            trainer.train_one_epoch()
        with self.assertRaisesRegex(ValueError, "val_loader"):
            trainer.validate()

    def test_train_validates_epoch_and_checkpoint_intervals(self) -> None:
        model = nn.Linear(1, 2)
        trainer = _make_trainer(
            model,
            [(torch.tensor([[1.0]]), torch.tensor([0]))],
            [(torch.tensor([[1.0]]), torch.tensor([0]))],
        )

        for kwargs, name in (
            ({"epochs": 0}, "epochs"),
            ({"epochs": 1, "save_every": 0}, "save_every"),
            ({"epochs": 1, "save_every": 1.5}, "save_every"),
            ({"epochs": 1, "start_epoch": -1}, "start_epoch"),
            ({"epochs": 1, "start_epoch": 2}, "start_epoch"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, name):
                    trainer.train(verbose=False, **kwargs)

    def test_steps_tail_of_non_divisible_accumulation(self) -> None:
        model = nn.Linear(1, 2)
        batches = [
            (torch.tensor([[1.0]]), torch.tensor([0])),
            (torch.tensor([[2.0]]), torch.tensor([1])),
            (torch.tensor([[3.0]]), torch.tensor([0])),
        ]
        optimizer = _CountingSGD(model.parameters(), lr=0.1)
        trainer = _make_trainer(model, batches, batches[:1], optimizer, accumulation=2)

        trainer.train_one_epoch()

        self.assertEqual(optimizer.step_calls, 2)

    def test_partial_accumulation_uses_actual_tail_batch_count(self) -> None:
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        batches = [
            (torch.tensor([[0.0]]), torch.tensor([[0.0]])),
            (torch.tensor([[0.0]]), torch.tensor([[0.0]])),
            (torch.tensor([[1.0]]), torch.tensor([[0.0]])),
        ]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = BaseTrainer(
            model,
            batches,
            batches[:1],
            optimizer,
            nn.MSELoss(),
            device="cpu",
            checkpoint_dir="/tmp/ckpt-compress-trainer-tests",
            gradient_accumulation_steps=2,
        )

        trainer.train_one_epoch()

        torch.testing.assert_close(model.weight, torch.tensor([[0.8]]))

    def test_optimizer_step_scheduler_includes_partial_accumulation(self) -> None:
        model = nn.Linear(1, 2)
        batches = [
            (torch.tensor([[1.0]]), torch.tensor([0])),
            (torch.tensor([[2.0]]), torch.tensor([1])),
            (torch.tensor([[3.0]]), torch.tensor([0])),
        ]
        optimizer = _CountingSGD(model.parameters(), lr=0.1)
        scheduler = mock.Mock()
        trainer = _make_trainer(
            model,
            batches,
            batches[:1],
            optimizer,
            accumulation=2,
            scheduler=scheduler,
            scheduler_step_mode="optimizer_step",
        )

        trainer.train_one_epoch()

        self.assertEqual(optimizer.step_calls, 2)
        self.assertEqual(scheduler.step.call_count, 2)

    def test_epoch_scheduler_is_not_advanced_by_optimizer_updates(self) -> None:
        model = nn.Linear(1, 2)
        batches = [
            (torch.tensor([[1.0]]), torch.tensor([0])),
            (torch.tensor([[2.0]]), torch.tensor([1])),
        ]
        scheduler = mock.Mock()
        trainer = _make_trainer(model, batches, batches[:1], scheduler=scheduler)

        with mock.patch.object(trainer, "save_checkpoint"):
            trainer.train(epochs=2, verbose=False)

        self.assertEqual(scheduler.step.call_count, 2)

    def test_checkpoint_round_trip_preserves_scheduler_state(self) -> None:
        with TemporaryDirectory() as checkpoint_dir:
            model = nn.Linear(1, 2)
            batches = [(torch.tensor([[1.0]]), torch.tensor([0]))]
            optimizer = _CountingSGD(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
            trainer = _make_trainer(
                model,
                batches,
                batches,
                optimizer,
                scheduler=scheduler,
                scheduler_step_mode="optimizer_step",
                checkpoint_dir=checkpoint_dir,
            )
            trainer.train_one_epoch()
            checkpoint_path = trainer.save_checkpoint(1, 0.5)

            restored_model = nn.Linear(1, 2)
            restored_optimizer = _CountingSGD(restored_model.parameters(), lr=0.1)
            restored_scheduler = torch.optim.lr_scheduler.StepLR(
                restored_optimizer, step_size=1, gamma=0.5
            )
            restored = _make_trainer(
                restored_model,
                batches,
                batches,
                restored_optimizer,
                scheduler=restored_scheduler,
                scheduler_step_mode="optimizer_step",
                checkpoint_dir=checkpoint_dir,
            )

            epoch = restored.load_checkpoint(checkpoint_path)

            self.assertEqual(epoch, 1)
            self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())

    def test_shared_loss_helper_handles_language_model_dict_batches(self) -> None:
        model = _TinyLM()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[2, 3, 4]]),
        }
        trainer = _make_trainer(model, [batch], [batch])

        loss = trainer.train_one_epoch()
        val_loss, accuracy = trainer.validate()

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(torch.isfinite(torch.tensor(val_loss)))
        self.assertEqual(accuracy, 0.0)

    def test_mapping_batches_forward_attention_mask(self) -> None:
        model = _MaskRecordingLM()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
            "labels": torch.tensor([[2, 3, -100]]),
        }
        trainer = _make_trainer(model, [batch], [batch])

        trainer.train_one_epoch()
        trainer.validate()

        self.assertEqual(len(model.seen_attention_masks), 2)
        for seen in model.seen_attention_masks:
            torch.testing.assert_close(seen, batch["attention_mask"])

    def test_resume_uses_target_epoch_numbers(self) -> None:
        model = nn.Linear(1, 2)
        batches = [(torch.tensor([[1.0]]), torch.tensor([0]))]
        trainer = _make_trainer(model, batches, batches)

        with mock.patch.object(trainer, "save_checkpoint") as save:
            trainer.train(epochs=4, start_epoch=2, verbose=False)

        self.assertEqual([call.args[0] for call in save.call_args_list], [3, 4])
        self.assertEqual(len(trainer.history["train_loss"]), 2)

    def test_checkpoint_restores_early_stopping_state_and_legacy_fallback(self) -> None:
        model = nn.Linear(1, 2)
        trainer = _make_trainer(model, [], [])
        state = {
            "epoch": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "history": {
                "train_loss": [1.0, 1.0, 1.0],
                "val_loss": [0.5, 0.6, 0.7],
                "val_acc": [0.0, 0.0, 0.0],
            },
        }

        with mock.patch("dacp.utils.trainer.torch.load", return_value=state):
            trainer.load_checkpoint("legacy.pt")
        self.assertEqual(trainer.best_val_loss, 0.5)
        self.assertEqual(trainer.patience_counter, 2)

        state["best_val_loss"] = 0.4
        state["patience_counter"] = 1
        with mock.patch("dacp.utils.trainer.torch.load", return_value=state):
            trainer.load_checkpoint("current.pt")
        self.assertEqual(trainer.best_val_loss, 0.4)
        self.assertEqual(trainer.patience_counter, 1)


if __name__ == "__main__":
    unittest.main()
