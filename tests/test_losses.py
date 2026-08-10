import unittest

import torch

from experiments.lib.losses import make_task_loss


class _TensorOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _Classifier(torch.nn.Module):
    def forward(self, inputs, attention_mask=None):
        del attention_mask
        return _TensorOutput(inputs)


class TestTaskLossFactory(unittest.TestCase):
    def test_rejects_unknown_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown task_type"):
            make_task_loss("unknown")

    def test_classification_and_cv_use_cross_entropy(self) -> None:
        model = _Classifier()
        logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
        labels = torch.tensor([0, 1])
        expected = torch.nn.functional.cross_entropy(logits, labels)

        cls_loss = make_task_loss("cls")(
            model,
            {"input_ids": logits, "labels": labels},
        )
        cv_loss = make_task_loss("cv")(
            model,
            {"images": logits, "labels": labels},
        )

        torch.testing.assert_close(cls_loss, expected)
        torch.testing.assert_close(cv_loss, expected)

    def test_lm_shifts_logits_and_labels(self) -> None:
        model = _Classifier()
        logits = torch.tensor(
            [[[2.0, -1.0], [-1.0, 2.0], [2.0, -1.0]]]
        )
        labels = torch.tensor([[0, 1, 0]])
        expected = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, 2),
            labels[:, 1:].reshape(-1),
        )

        loss = make_task_loss("lm")(
            model,
            {"input_ids": logits, "labels": labels},
        )

        torch.testing.assert_close(loss, expected)

    def test_reg_keeps_legacy_classification_mapping(self) -> None:
        model = _Classifier()
        logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
        labels = torch.tensor([0, 1])

        loss = make_task_loss("reg")(
            model,
            {"input_ids": logits, "labels": labels},
        )

        torch.testing.assert_close(
            loss,
            torch.nn.functional.cross_entropy(logits, labels),
        )


if __name__ == "__main__":
    unittest.main()
