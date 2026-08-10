import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import experiments.lib.models as models


class TestModelCheckpointLoading(unittest.TestCase):
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
