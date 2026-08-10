import unittest

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


def _make_trainer(model, train_loader, val_loader, optimizer=None, accumulation=1):
    if optimizer is None:
        optimizer = _CountingSGD(model.parameters(), lr=0.1)
    return BaseTrainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        nn.CrossEntropyLoss(),
        device="cpu",
        checkpoint_dir="/tmp/ckpt-compress-trainer-tests",
        gradient_accumulation_steps=accumulation,
    )


class TestBaseTrainer(unittest.TestCase):
    def test_rejects_non_positive_accumulation(self) -> None:
        model = nn.Linear(2, 2)
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
                    _make_trainer(model, [], [], accumulation=value)

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


if __name__ == "__main__":
    unittest.main()
