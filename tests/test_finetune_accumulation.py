import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.scripts.finetune import finetune_bert_large, finetune_gpt2_1000steps


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class _TinyBert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask
        self.forward_calls += 1
        prediction = self.weight * input_ids.float().mean()
        return SimpleNamespace(loss=(prediction - labels.float().mean()).square())


class _TinyGpt(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        self.forward_calls += 1
        zeros = torch.zeros(*input_ids.shape, device=input_ids.device)
        return torch.stack((self.weight + zeros, -self.weight + zeros), dim=-1)


class TestStandaloneFinetuneAccumulation(unittest.TestCase):
    def test_bert_num_steps_counts_optimizer_updates(self) -> None:
        model = _TinyBert()
        optimizer = _CountingSGD(model.parameters(), lr=0.1)
        batch = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "labels": torch.tensor([1]),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            finetune_bert_large.train_n_steps(
                model,
                [batch],
                None,
                optimizer,
                "cpu",
                "classification",
                "accuracy",
                num_steps=2,
                gradient_accumulation_steps=3,
                log_interval=10,
                save_interval=10,
                checkpoint_dir=temp_dir,
            )
            checkpoint = torch.load(
                Path(temp_dir) / "checkpoint_step_2_final.pt",
                weights_only=True,
            )

        self.assertEqual(optimizer.step_calls, 2)
        self.assertEqual(model.forward_calls, 6)
        self.assertEqual(checkpoint["optimizer_steps"], 2)
        self.assertEqual(checkpoint["micro_steps"], 6)
        self.assertEqual(checkpoint["gradient_accumulation_steps"], 3)

    def test_gpt_num_steps_counts_optimizer_updates_and_forwards_mask(self) -> None:
        model = _TinyGpt()
        optimizer = _CountingSGD(model.parameters(), lr=0.1)
        batch = {
            "input_ids": torch.tensor([[0, 1]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "labels": torch.tensor([[0, 1]]),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            finetune_gpt2_1000steps.train_n_steps(
                model,
                [batch],
                optimizer,
                torch.nn.CrossEntropyLoss(),
                "cpu",
                num_steps=2,
                gradient_accumulation_steps=3,
                log_interval=10,
                save_interval=10,
                checkpoint_dir=temp_dir,
            )
            checkpoint = torch.load(
                Path(temp_dir) / "checkpoint_step_2_final.pt",
                weights_only=True,
            )

        self.assertEqual(optimizer.step_calls, 2)
        self.assertEqual(model.forward_calls, 6)
        self.assertEqual(checkpoint["optimizer_steps"], 2)
        self.assertEqual(checkpoint["micro_steps"], 6)
        self.assertEqual(checkpoint["gradient_accumulation_steps"], 3)


if __name__ == "__main__":
    unittest.main()
