import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import experiments.lib.models as models


class TestModelCheckpointLoading(unittest.TestCase):
    def test_model_creation_dispatches_by_model_family(self) -> None:
        sentinel = object()
        routes = (
            ("gpt2-small", "_create_gpt2", ("gpt2-small", False)),
            ("bert-base", "_create_bert", ("bert-base", False, "sst2")),
            ("resnet18", "_create_resnet", ("resnet18", False)),
            ("pythia-410m", "_create_pythia", (False,)),
            ("gpt2-large", "_create_gpt2_large", (False,)),
            ("pythia-1b", "_create_pythia_1b", (False,)),
            ("vit-b-16", "_create_vit", ("vit-b-16", False)),
        )

        for model_name, helper_name, expected_args in routes:
            with self.subTest(model_name=model_name), mock.patch.object(
                models, helper_name, return_value=sentinel
            ) as helper:
                result = models._create_model(
                    model_name,
                    False,
                    dataset_name="sst2" if helper_name == "_create_bert" else None,
                )

            self.assertIs(result, sentinel)
            helper.assert_called_once_with(*expected_args)

    def test_model_creation_rejects_unknown_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            models._create_model("unknown", pretrained=False)

    def test_extract_checkpoint_state_supports_wrapped_and_raw_payloads(self) -> None:
        state = {"weight": torch.ones(1)}

        for payload in (
            {"model_state_dict": state},
            {"state_dict": state},
            {"model": state},
            state,
        ):
            self.assertIs(models._extract_checkpoint_state(payload), state)

    def test_non_strict_loading_reports_incompatible_keys(self) -> None:
        model = mock.Mock()
        model.load_state_dict.return_value = SimpleNamespace(
            missing_keys=["head.weight"],
            unexpected_keys=["module.extra"],
        )
        state = {"weight": torch.ones(1)}

        with mock.patch.object(models.torch, "load", return_value={"state_dict": state}), \
             mock.patch("builtins.print") as print_mock:
            models._load_checkpoint(model, "checkpoint.pt")

        model.load_state_dict.assert_called_once_with(state, strict=False)
        self.assertTrue(
            any("检查点键不匹配" in str(call) for call in print_mock.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
