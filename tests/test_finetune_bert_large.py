import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from experiments.scripts.finetune import finetune_bert_large


class _SpyBert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def forward(self, input_ids, **kwargs):
        self.calls.append((input_ids, kwargs))
        logits = torch.tensor(
            [[4.0, 0.0], [0.0, 4.0]],
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits)


class TestFinetuneBertLarge(unittest.TestCase):
    def test_parser_defaults_and_explicit_checkpoint_directory(self) -> None:
        args = finetune_bert_large.parse_args(["--dataset", "mnli"])
        self.assertEqual(args.checkpoint_dir, "./checkpoints/bert_large_mnli")

        explicit = finetune_bert_large.parse_args(
            ["--dataset", "mnli", "--checkpoint_dir", "custom/checkpoints"]
        )
        self.assertEqual(explicit.checkpoint_dir, "custom/checkpoints")

        with self.assertRaisesRegex(ValueError, "num_steps"):
            finetune_bert_large.default_checkpoint_dir("mnli", 0)

    def test_classification_moves_batch_and_forwards_token_type_ids(self) -> None:
        model = _SpyBert()
        token_type_ids = torch.tensor([[0, 1], [1, 0]])
        batch = {
            "input_ids": torch.ones(2, 2, dtype=torch.long),
            "attention_mask": torch.ones(2, 2, dtype=torch.long),
            "token_type_ids": token_type_ids,
            "labels": torch.tensor([0, 1]),
        }

        with mock.patch.object(
            finetune_bert_large,
            "move_batch_to_device",
            wraps=finetune_bert_large.move_batch_to_device,
        ) as move_batch:
            accuracy = finetune_bert_large.evaluate_classification(
                model,
                [batch],
                "cpu",
            )

        self.assertEqual(accuracy, 1.0)
        move_batch.assert_called_once_with(batch, "cpu")
        self.assertEqual(len(model.calls), 1)
        self.assertIn("token_type_ids", model.calls[0][1])
        torch.testing.assert_close(model.calls[0][1]["token_type_ids"], token_type_ids)


if __name__ == "__main__":
    unittest.main()
