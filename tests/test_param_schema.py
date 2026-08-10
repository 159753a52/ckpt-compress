import unittest

import torch

from dacp.pruning.param_schema import (
    get_prunable_types,
    infer_layer_type,
    register_type_rule,
)


class TestParameterTypeSchema(unittest.TestCase):
    def test_builtin_rules_and_prunable_types_share_one_schema(self) -> None:
        self.assertEqual(get_prunable_types("resnet"), ["conv", "fc"])
        self.assertEqual(
            infer_layer_type("layer1.conv1.weight", torch.ones(2, 2), "resnet"),
            "conv",
        )

    def test_custom_family_registers_rule_and_types_together(self) -> None:
        register_type_rule(
            "test-custom-conv",
            lambda name, tensor: "skip" if tensor.dim() < 2 else "conv",
            ("conv",),
        )

        self.assertEqual(get_prunable_types("test-custom-conv"), ["conv"])
        self.assertEqual(
            infer_layer_type("weight", torch.ones(2, 2), "test-custom-conv"),
            "conv",
        )

    def test_unknown_family_fails_consistently(self) -> None:
        with self.assertRaisesRegex(ValueError, "No type schema"):
            get_prunable_types("missing-family")
        with self.assertRaisesRegex(ValueError, "No type schema"):
            infer_layer_type("weight", torch.ones(2, 2), "missing-family")

    def test_rule_cannot_return_an_unregistered_type(self) -> None:
        register_type_rule(
            "test-invalid-output",
            lambda _name, _tensor: "mlp",
            ("conv",),
        )

        with self.assertRaisesRegex(ValueError, "unregistered type"):
            infer_layer_type(
                "weight",
                torch.ones(2, 2),
                "test-invalid-output",
            )

    def test_registration_validates_reserved_and_duplicate_types(self) -> None:
        for types in (("skip",), ("conv", "conv"), ()):
            with self.subTest(types=types):
                with self.assertRaises(ValueError):
                    register_type_rule(
                        "test-invalid-registration",
                        lambda _name, _tensor: "skip",
                        types,
                    )


if __name__ == "__main__":
    unittest.main()
