import unittest

import torch
import torch.nn as nn

from dacp.models._wrapper import DelegatingModel


class _Wrapper(DelegatingModel):
    pass


class _LegacyAttributeWrapper(DelegatingModel):
    @property
    def _wrapped_model(self) -> nn.Module:
        return self.resnet


class TestDelegatingModel(unittest.TestCase):
    def test_delegates_state_and_parameter_contract(self) -> None:
        inner = nn.Linear(2, 1)
        wrapper = _Wrapper()
        wrapper.model = inner

        self.assertEqual(
            [name for name, _ in wrapper.named_parameters()],
            [name for name, _ in inner.named_parameters()],
        )
        self.assertEqual(set(wrapper.state_dict()), set(inner.state_dict()))

        replacement = {
            name: torch.full_like(value, 2.0) for name, value in inner.state_dict().items()
        }
        wrapper.load_state_dict(replacement)
        for name, value in inner.state_dict().items():
            self.assertTrue(torch.equal(value, replacement[name]))

    def test_supports_wrappers_with_a_legacy_inner_attribute(self) -> None:
        inner = nn.Linear(2, 1)
        wrapper = _LegacyAttributeWrapper()
        wrapper.resnet = inner

        self.assertEqual(
            [name for name, _ in wrapper.named_parameters()],
            [name for name, _ in inner.named_parameters()],
        )
        self.assertIs(wrapper.train(False), wrapper)
        self.assertFalse(wrapper.training)
        self.assertFalse(inner.training)
        self.assertIs(wrapper.eval(), wrapper)
        self.assertFalse(wrapper.training)


if __name__ == "__main__":
    unittest.main()
