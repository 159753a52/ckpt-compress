import math
import unittest
from unittest import mock

import torch

from experiments.scripts.finetune import (
    finetune_gpt2_medium,
    finetune_pythia_410m,
    finetune_vit_large,
)


class TestFinetuneCli(unittest.TestCase):
    def test_amp_can_be_disabled_and_is_never_enabled_on_cpu(self) -> None:
        modules = (
            finetune_gpt2_medium,
            finetune_pythia_410m,
            finetune_vit_large,
        )
        for module in modules:
            with self.subTest(module=module.__name__):
                self.assertFalse(module.parse_args(["--no-use_amp"]).use_amp)
                self.assertFalse(module.parse_args(["--device", "cpu"]).use_amp)
                self.assertTrue(module.parse_args(["--device", "cuda"]).use_amp)

    def test_pythia_evaluation_uses_masked_labels_and_restores_eval_mode(self) -> None:
        model = mock.Mock()
        model.training = False
        model.return_value.logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]]])
        batch = {
            "input_ids": torch.tensor([[0, 1, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "labels": torch.tensor([[0, 1, 0, -100]]),
        }

        ppl, average_loss = finetune_pythia_410m.evaluate_ppl(model, [batch], "cpu")

        summed_loss = torch.nn.functional.cross_entropy(
            model.return_value.logits[..., :-1, :].reshape(-1, 2),
            batch["labels"][..., 1:].reshape(-1),
            reduction="sum",
        ).item()
        self.assertAlmostEqual(average_loss, summed_loss / 2)
        self.assertAlmostEqual(ppl, math.exp(summed_loss / 2))
        model.assert_called_once_with(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        model.train.assert_called_once_with(False)

    def test_pythia_evaluation_rejects_batches_without_supervised_tokens(self) -> None:
        model = mock.Mock()
        model.training = True
        model.return_value.logits = torch.zeros(1, 2, 3)
        batch = {
            "input_ids": torch.tensor([[0, 0]]),
            "attention_mask": torch.tensor([[0, 0]]),
            "labels": torch.tensor([[-100, -100]]),
        }

        with self.assertRaisesRegex(ValueError, "no supervised"):
            finetune_pythia_410m.evaluate_ppl(model, [batch], "cpu")
        model.train.assert_called_once_with(True)


if __name__ == "__main__":
    unittest.main()
