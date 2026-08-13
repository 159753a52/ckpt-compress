import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from experiments.scripts import eval_checkpoints_ppl
from experiments.scripts.finetune import (
    finetune_bert_large,
    finetune_gpt2_medium,
    finetune_vit_large,
)


class _FixedLM(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        logits = torch.zeros(*input_ids.shape, 2)
        logits[..., 0] = 1.0
        return SimpleNamespace(logits=logits)


class _FixedClassifier(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, *args, **kwargs):
        del args, kwargs
        return SimpleNamespace(logits=self.logits)


class TestFinetuneEvaluation(unittest.TestCase):
    def test_gpt2_evaluators_share_token_weighted_contract(self) -> None:
        batch = {
            "input_ids": torch.tensor([[0, 1, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
            "labels": torch.tensor([[0, 1, 0]]),
        }
        for evaluator, returns_loss in (
            (eval_checkpoints_ppl.evaluate_ppl, True),
            (finetune_gpt2_medium.evaluate_ppl, False),
        ):
            with self.subTest(evaluator=evaluator.__module__):
                model = _FixedLM()
                model.eval()
                result = evaluator(model, [batch], "cpu", max_batches=1)
                ppl = result[0] if returns_loss else result
                self.assertAlmostEqual(ppl, math.exp(1.31326162815094))
                self.assertFalse(model.training)

                with self.assertRaisesRegex(ValueError, "supervised"):
                    evaluator(model, [], "cpu", max_batches=1)
                self.assertFalse(model.training)

    def test_bert_classification_and_regression_fail_closed(self) -> None:
        classification = _FixedClassifier(torch.tensor([[4.0, 0.0], [0.0, 4.0]]))
        classification.train()
        cls_batch = {
            "input_ids": torch.ones(2, 2, dtype=torch.long),
            "attention_mask": torch.ones(2, 2, dtype=torch.long),
            "labels": torch.tensor([0, 1]),
        }
        self.assertEqual(
            finetune_bert_large.evaluate_classification(
                classification,
                [cls_batch],
                "cpu",
            ),
            1.0,
        )
        self.assertTrue(classification.training)
        with self.assertRaisesRegex(ValueError, "at least one example"):
            finetune_bert_large.evaluate_classification(classification, [], "cpu")

        regression = _FixedClassifier(torch.tensor([[1.0], [2.0], [3.0]]))
        regression.eval()
        reg_batch = {
            "input_ids": torch.ones(3, 2, dtype=torch.long),
            "attention_mask": torch.ones(3, 2, dtype=torch.long),
            "labels": torch.tensor([2.0, 4.0, 6.0]),
        }
        self.assertAlmostEqual(
            finetune_bert_large.evaluate_regression(regression, [reg_batch], "cpu"),
            1.0,
        )
        self.assertFalse(regression.training)

        constant = _FixedClassifier(torch.ones(3, 1))
        with self.assertRaisesRegex(ValueError, "non-constant"):
            finetune_bert_large.evaluate_regression(constant, [reg_batch], "cpu")

    def test_vit_accuracy_restores_mode_and_rejects_empty_data(self) -> None:
        model = _FixedClassifier(torch.tensor([[4.0, 0.0], [0.0, 4.0]]))
        model.eval()
        batch = (torch.ones(2, 3, 2, 2), torch.tensor([0, 1]))

        self.assertEqual(
            finetune_vit_large.evaluate_acc(
                model,
                [batch],
                "cpu",
                max_batches=1,
                use_amp=False,
            ),
            1.0,
        )
        self.assertFalse(model.training)
        with self.assertRaisesRegex(ValueError, "at least one example"):
            finetune_vit_large.evaluate_acc(
                model,
                [],
                "cpu",
                max_batches=1,
                use_amp=False,
            )


if __name__ == "__main__":
    unittest.main()
