import math
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

from experiments.lib.evaluation import compute_quality_drop, evaluate, evaluate_downstream


class _VariableLossLM(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        logits = torch.zeros(*input_ids.shape, 2, device=input_ids.device)
        if input_ids[0, 0].item() == 0:
            logits[..., 0, 0] = 3.0
        else:
            logits[..., 0, 0] = 3.0
            logits[..., 1, 1] = 3.0
        return logits


class TestEvaluation(unittest.TestCase):
    def test_evaluate_restores_model_mode_on_success_and_failure(self) -> None:
        model = _VariableLossLM()
        model.train()
        evaluate(
            model,
            [{"input_ids": torch.tensor([[0, 1]]), "labels": torch.tensor([[0, 1]])}],
            "lm",
            device="cpu",
        )
        self.assertTrue(model.training)

        model.eval()
        with self.assertRaisesRegex(ValueError, "Unknown task_type"):
            evaluate(model, [], "unknown", device="cpu")
        self.assertFalse(model.training)

    def test_lm_metrics_are_weighted_by_supervised_tokens(self) -> None:
        batches = [
            {
                "input_ids": torch.tensor([[0, 1, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0]]),
                "labels": torch.tensor([[0, 1, -100]]),
            },
            {
                "input_ids": torch.tensor([[1, 0, 1]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "labels": torch.tensor([[1, 0, 1]]),
            },
        ]

        metrics = evaluate(_VariableLossLM(), batches, "lm", device="cpu")

        first_loss = torch.nn.functional.cross_entropy(
            torch.tensor([[3.0, 0.0]]),
            torch.tensor([1]),
            reduction="sum",
        )
        second_loss = torch.nn.functional.cross_entropy(
            torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
            torch.tensor([0, 1]),
            reduction="sum",
        )
        expected = (first_loss + second_loss).item() / 3
        self.assertAlmostEqual(metrics["loss"], expected)
        self.assertAlmostEqual(metrics["perplexity"], math.exp(expected))

    def test_lm_metrics_reject_empty_or_fully_masked_evaluation(self) -> None:
        with self.assertRaisesRegex(ValueError, "supervised token"):
            evaluate(_VariableLossLM(), [], "lm", device="cpu")
        with self.assertRaisesRegex(ValueError, "no supervised tokens"):
            evaluate(
                _VariableLossLM(),
                [
                    {
                        "input_ids": torch.tensor([[0, 0]]),
                        "labels": torch.tensor([[-100, -100]]),
                    }
                ],
                "lm",
                device="cpu",
            )

    def test_empty_classification_and_regression_batches_fail_closed(self) -> None:
        model = torch.nn.Linear(1, 1)
        for task_type, message in (
            ("cls", "classification evaluation"),
            ("reg", "regression evaluation"),
        ):
            with (
                self.subTest(task_type=task_type),
                self.assertRaisesRegex(
                    ValueError,
                    message,
                ),
            ):
                evaluate(model, [], task_type, device="cpu")

    def test_classification_loss_is_weighted_by_examples(self) -> None:
        class IdentityClassifier(torch.nn.Module):
            def forward(self, inputs, attention_mask=None):
                del attention_mask
                return inputs

        large_logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0]])
        small_logits = torch.tensor([[4.0, 0.0]])
        batches = [
            {"input_ids": large_logits, "labels": torch.tensor([0, 0, 0])},
            {"input_ids": small_logits, "labels": torch.tensor([1])},
        ]

        metrics = evaluate(IdentityClassifier(), batches, "cls", device="cpu")

        expected = (
            torch.nn.functional.cross_entropy(
                large_logits,
                batches[0]["labels"],
                reduction="sum",
            )
            + torch.nn.functional.cross_entropy(
                small_logits,
                batches[1]["labels"],
                reduction="sum",
            )
        ).item() / 4
        self.assertAlmostEqual(metrics["loss"], expected)
        self.assertEqual(metrics["accuracy"], 0.75)

    def test_classification_evaluation_forwards_token_type_ids(self) -> None:
        class SpyClassifier(torch.nn.Module):
            def forward(self, inputs, **kwargs):
                self.kwargs = kwargs
                return inputs

        model = SpyClassifier()
        token_type_ids = torch.tensor([[0, 1], [1, 0]])
        evaluate(
            model,
            [
                {
                    "input_ids": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
                    "token_type_ids": token_type_ids,
                    "labels": torch.tensor([0, 1]),
                }
            ],
            "cls",
            device="cpu",
        )
        self.assertIs(model.kwargs["token_type_ids"], token_type_ids)

    def test_lm_evaluation_does_not_forward_token_type_ids(self) -> None:
        class SpyLM(torch.nn.Module):
            def forward(self, inputs, **kwargs):
                self.kwargs = kwargs
                return torch.zeros(inputs.shape[0], inputs.shape[1], 2)

        model = SpyLM()
        evaluate(
            model,
            [
                {
                    "input_ids": torch.tensor([[0, 1]]),
                    "token_type_ids": torch.zeros(1, 2, dtype=torch.long),
                    "labels": torch.tensor([[0, 1]]),
                }
            ],
            "lm",
            device="cpu",
        )
        self.assertNotIn("token_type_ids", model.kwargs)

    def test_regression_loss_is_weighted_by_elements(self) -> None:
        class IdentityRegressor(torch.nn.Module):
            def forward(self, inputs, attention_mask=None):
                del attention_mask
                return inputs

        batches = [
            {
                "input_ids": torch.tensor([[0.0], [1.0], [2.0]]),
                "labels": torch.tensor([0.0, 1.0, 2.0]),
            },
            {
                "input_ids": torch.tensor([[2.0]]),
                "labels": torch.tensor([0.0]),
            },
        ]

        metrics = evaluate(IdentityRegressor(), batches, "reg", device="cpu")

        self.assertAlmostEqual(metrics["loss"], 1.0)

    def test_regression_rejects_constant_or_non_finite_pearson_inputs(self) -> None:
        class IdentityRegressor(torch.nn.Module):
            def forward(self, inputs, attention_mask=None):
                del attention_mask
                return inputs

        invalid_batches = (
            {
                "input_ids": torch.tensor([[1.0], [1.0]]),
                "labels": torch.tensor([0.0, 1.0]),
            },
            {
                "input_ids": torch.tensor([[0.0], [1.0]]),
                "labels": torch.tensor([1.0, 1.0]),
            },
            {
                "input_ids": torch.tensor([[0.0], [float("nan")]]),
                "labels": torch.tensor([0.0, 1.0]),
            },
        )
        for batch in invalid_batches:
            with (
                self.subTest(batch=batch),
                self.assertRaisesRegex(ValueError, "non-constant|finite"),
            ):
                evaluate(IdentityRegressor(), [batch], "reg", device="cpu")

    def test_quality_drop_preserves_normalized_direction(self) -> None:
        self.assertAlmostEqual(
            compute_quality_drop({"loss": 2.0}, {"loss": 2.2}, "lm"),
            10.0,
        )
        self.assertAlmostEqual(
            compute_quality_drop({"accuracy": 0.8}, {"accuracy": 0.76}, "cls"),
            5.0,
        )
        self.assertAlmostEqual(
            compute_quality_drop({"pearson": -0.5}, {"pearson": -0.4}, "reg"),
            -20.0,
        )

    def test_quality_drop_zero_baseline_is_explicit(self) -> None:
        self.assertEqual(
            compute_quality_drop({"loss": 0.0}, {"loss": 0.0}, "lm"),
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "undefined for a zero baseline"):
            compute_quality_drop({"loss": 0.0}, {"loss": 0.1}, "lm")
        with self.assertRaisesRegex(ValueError, "undefined for a zero baseline"):
            compute_quality_drop({"accuracy": 0.0}, {"accuracy": 0.1}, "cls")

    def test_quality_drop_rejects_missing_or_unknown_contracts(self) -> None:
        with self.assertRaisesRegex(KeyError, "baseline metrics must contain 'loss'"):
            compute_quality_drop({}, {"loss": 1.0}, "lm")
        with self.assertRaisesRegex(ValueError, "Unknown task_type"):
            compute_quality_drop({"loss": 1.0}, {"loss": 1.0}, "unknown")

    def test_downstream_rejects_invalid_arguments_before_loading_optional_stack(self) -> None:
        invalid = (
            ({"tasks": []}, "tasks"),
            ({"tasks": ["piqa", "piqa"]}, "tasks"),
            ({"tasks": [""]}, "tasks"),
            ({"batch_size": 0}, "batch_size"),
            ({"batch_size": True}, "batch_size"),
            ({"num_fewshot": -1}, "num_fewshot"),
        )
        for arguments, message in invalid:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, message):
                evaluate_downstream(object(), "tokenizer", **arguments)

    def test_downstream_requires_finite_accuracy_for_every_task(self) -> None:
        lm_eval = ModuleType("lm_eval")
        huggingface = ModuleType("lm_eval.models.huggingface")
        huggingface.HFLM = mock.Mock(return_value=object())
        tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = SimpleNamespace(
            from_pretrained=mock.Mock(return_value=tokenizer)
        )

        cases = (
            ({"results": {}}, "missing results"),
            ({"results": {"piqa": {}}}, "no accuracy metric"),
            ({"results": {"piqa": {"acc,none": float("nan")}}}, "finite and in"),
            ({"results": {"piqa": {"acc,none": 1.1}}}, "finite and in"),
        )
        with mock.patch.dict(
            sys.modules,
            {
                "lm_eval": lm_eval,
                "lm_eval.models.huggingface": huggingface,
                "transformers": transformers,
            },
        ):
            for response, message in cases:
                lm_eval.simple_evaluate = mock.Mock(return_value=response)
                with self.subTest(response=response), self.assertRaisesRegex(ValueError, message):
                    evaluate_downstream(object(), "tokenizer", tasks=["piqa"], device="cpu")

            lm_eval.simple_evaluate = mock.Mock(
                return_value={
                    "results": {
                        "piqa": {"acc,none": 0.75},
                        "arc_easy": {"acc_norm,none": 0.5},
                    }
                }
            )
            self.assertEqual(
                evaluate_downstream(
                    object(),
                    "tokenizer",
                    tasks=["piqa", "arc_easy"],
                    device="cpu",
                ),
                {"piqa": 0.75, "arc_easy": 0.5, "avg_acc": 0.625},
            )


if __name__ == "__main__":
    unittest.main()
