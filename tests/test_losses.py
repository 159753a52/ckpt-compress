import unittest
from unittest import mock

import torch

from experiments.lib.losses import (
    compute_task_loss,
    extract_logits,
    make_task_loss,
    perplexity_from_loss,
    transformer_batch_kwargs,
)


class _TensorOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _Classifier(torch.nn.Module):
    def forward(self, inputs, attention_mask=None):
        del attention_mask
        return _TensorOutput(inputs)


class _SpyClassifier(torch.nn.Module):
    def forward(self, inputs, **kwargs):
        self.kwargs = kwargs
        return _TensorOutput(inputs)


class TestTaskLossFactory(unittest.TestCase):
    def test_extract_logits_supports_tensor_and_model_output_contracts(self) -> None:
        logits = torch.tensor([[1.0, 2.0]])

        self.assertIs(extract_logits(logits), logits)
        self.assertIs(extract_logits(_TensorOutput(logits)), logits)

    def test_factory_reuses_stateless_task_loss_closures(self) -> None:
        self.assertIs(make_task_loss("lm"), make_task_loss("lm"))
        self.assertIs(make_task_loss("cls"), make_task_loss("cls"))

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

    def test_transformer_batch_kwargs_is_optional(self) -> None:
        token_type_ids = torch.tensor([[0, 1]])
        self.assertEqual(transformer_batch_kwargs({}), {})
        self.assertIs(
            transformer_batch_kwargs({"token_type_ids": token_type_ids})["token_type_ids"],
            token_type_ids,
        )

    def test_classification_loss_forwards_token_type_ids(self) -> None:
        model = _SpyClassifier()
        token_type_ids = torch.tensor([[0, 1], [1, 0]])

        make_task_loss("cls")(
            model,
            {
                "input_ids": torch.tensor([[2.0, -1.0], [-1.0, 2.0]]),
                "token_type_ids": token_type_ids,
                "labels": torch.tensor([0, 1]),
            },
        )

        self.assertIs(model.kwargs["token_type_ids"], token_type_ids)

    def test_lm_does_not_forward_token_type_ids(self) -> None:
        model = mock.Mock(return_value=torch.zeros(1, 3, 2))

        make_task_loss("lm")(
            model,
            {
                "input_ids": torch.tensor([[0, 1, 0]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "token_type_ids": torch.zeros(1, 3, dtype=torch.long),
                "labels": torch.tensor([[0, 1, 0]]),
            },
        )

        self.assertNotIn("token_type_ids", model.call_args.kwargs)

    def test_lm_shifts_logits_and_labels(self) -> None:
        logits = torch.tensor([[[2.0, -1.0], [-1.0, 2.0], [2.0, -1.0]]])
        model = mock.Mock(return_value=logits)
        input_ids = torch.tensor([[0, 1, 0]])
        labels = torch.tensor([[0, 1, 0]])
        expected = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, 2),
            labels[:, 1:].reshape(-1),
        )

        loss = make_task_loss("lm")(
            model,
            {"input_ids": input_ids, "labels": labels},
        )

        torch.testing.assert_close(loss, expected)

    def test_lm_forwards_attention_mask_and_ignores_masked_labels(self) -> None:
        logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]]])
        model = mock.Mock(return_value=logits)
        input_ids = torch.tensor([[0, 1, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])
        labels = torch.tensor([[0, 1, 0, 1]])

        loss = compute_task_loss(
            model,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            },
            "lm",
            "cpu",
        )

        expected = torch.nn.functional.cross_entropy(
            logits[..., :-1, :].reshape(-1, 2),
            torch.tensor([1, 0, -100]),
        )
        torch.testing.assert_close(loss, expected)
        model.assert_called_once_with(input_ids, attention_mask=attention_mask)

    def test_perplexity_conversion_is_exact_and_fail_closed(self) -> None:
        self.assertAlmostEqual(perplexity_from_loss(3.0), torch.exp(torch.tensor(3.0)).item())
        self.assertGreater(perplexity_from_loss(21.0), torch.exp(torch.tensor(20.0)).item())
        for invalid in (True, -1.0, float("nan"), float("inf"), 1000.0):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                perplexity_from_loss(invalid)

    def test_regression_uses_mse_and_forwards_attention_mask(self) -> None:
        model = _Classifier()
        predictions = torch.tensor([[2.0], [4.0]])
        labels = torch.tensor([1.0, 3.0])
        attention_mask = torch.ones(2, 1, dtype=torch.long)

        loss = make_task_loss("reg")(
            model,
            {
                "input_ids": predictions,
                "attention_mask": attention_mask,
                "labels": labels,
            },
        )

        torch.testing.assert_close(loss, torch.tensor(1.0))

    def test_regression_validates_label_dtype_and_prediction_shape(self) -> None:
        model = _Classifier()
        with self.assertRaisesRegex(TypeError, "floating-point"):
            make_task_loss("reg")(
                model,
                {"input_ids": torch.ones(2, 1), "labels": torch.ones(2, dtype=torch.long)},
            )
        with self.assertRaisesRegex(ValueError, "match label shape"):
            make_task_loss("reg")(
                model,
                {"input_ids": torch.ones(2, 2), "labels": torch.ones(2)},
            )

    def test_compute_task_loss_supports_regression(self) -> None:
        loss = compute_task_loss(
            _Classifier(),
            {
                "input_ids": torch.tensor([[2.0], [4.0]]),
                "attention_mask": torch.ones(2, 1),
                "labels": torch.tensor([1.0, 3.0]),
            },
            "reg",
            "cpu",
        )

        torch.testing.assert_close(loss, torch.tensor(1.0))

    def test_compute_task_loss_normalizes_cv_tuple_batches(self) -> None:
        model = _Classifier()
        logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
        labels = torch.tensor([0, 1])

        loss = compute_task_loss(model, (logits, labels), "cv", "cpu")

        torch.testing.assert_close(
            loss,
            torch.nn.functional.cross_entropy(logits, labels),
        )

    def test_compute_task_loss_rejects_unsupported_batch_and_task(self) -> None:
        model = _Classifier()
        with self.assertRaisesRegex(ValueError, "Unknown training task_type"):
            compute_task_loss(model, {}, "unknown", "cpu")
        with self.assertRaisesRegex(TypeError, "mappings"):
            compute_task_loss(model, torch.tensor([1.0]), "lm", "cpu")


if __name__ == "__main__":
    unittest.main()
