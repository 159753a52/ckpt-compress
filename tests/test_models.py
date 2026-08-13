import tempfile
import unittest
from pathlib import Path
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
            with (
                self.subTest(model_name=model_name),
                mock.patch.object(models, helper_name, return_value=sentinel) as helper,
            ):
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

    def test_pythia_variants_reuse_local_config_when_available(self) -> None:
        fake_config = object()
        fake_model = object()
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "pythia-410m"
            snapshot.mkdir()
            with (
                mock.patch.object(models, "resolve_model_source", return_value=str(snapshot)),
                mock.patch(
                    "transformers.GPTNeoXConfig.from_pretrained", return_value=fake_config
                ) as load_config,
                mock.patch("transformers.GPTNeoXForCausalLM", return_value=fake_model) as construct,
            ):
                self.assertIs(models._create_pythia_variant("pythia-410m", False), fake_model)

        load_config.assert_called_once_with(str(snapshot), local_files_only=True)
        construct.assert_called_once_with(fake_config)

    def test_pythia_variant_rejects_unknown_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported Pythia"):
            models._create_pythia_variant("pythia-unknown", False)

    def test_extract_checkpoint_state_supports_wrapped_and_raw_payloads(self) -> None:
        state = {"weight": torch.ones(1)}

        for payload in (
            {"model_state_dict": state},
            {"state_dict": state},
            {"model": state},
            state,
        ):
            self.assertIs(models._extract_checkpoint_state(payload), state)

    def test_checkpoint_loading_is_strict_and_tensor_only_by_default(self) -> None:
        model = mock.Mock()
        model.load_state_dict.return_value = SimpleNamespace(
            missing_keys=[],
            unexpected_keys=[],
        )
        state = {"weight": torch.ones(1)}

        with mock.patch.object(models.torch, "load", return_value={"state_dict": state}) as load:
            models._load_checkpoint(model, "checkpoint.pt")

        model.load_state_dict.assert_called_once_with(state, strict=True)
        self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_checkpoint_loading_rejects_invalid_state_payloads(self) -> None:
        model = mock.Mock()
        for payload in (None, {"model_state_dict": []}, {"weight": "not-a-tensor"}):
            with (
                self.subTest(payload=payload),
                mock.patch.object(models.torch, "load", return_value=payload),
                self.assertRaisesRegex(TypeError, "mapping|string keys"),
            ):
                models._load_checkpoint(model, "checkpoint.pt")
            model.load_state_dict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
