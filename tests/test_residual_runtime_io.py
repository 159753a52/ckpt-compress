import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import experiments.lib.residual_runtime as residual_runtime
from experiments.lib.residual_runtime import (
    batch_hash,
    checkpoint_optimizer_state,
    checkpoint_state,
    load_token_batches,
    load_training_checkpoint,
    optimizer_state_to_cpu,
    write_json,
)


class IntegerTokenizer:
    def encode(self, line: str) -> list[int]:
        return [int(value) for value in line.split()]


class TestResidualRuntimeIO(unittest.TestCase):
    def test_checkpoint_state_supports_all_existing_payload_schemas(self) -> None:
        nested_schemas = ("model_state_dict", "state_dict", "model")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, key in enumerate(nested_schemas, start=1):
                with self.subTest(key=key):
                    path = root / f"{key}.pt"
                    torch.save({key: {"weight": torch.tensor([index])}}, path)
                    state = checkpoint_state(path)
                    self.assertTrue(
                        torch.equal(state["weight"], torch.tensor([index]))
                    )

            raw_path = root / "raw.pt"
            torch.save({"weight": torch.tensor([4])}, raw_path)
            self.assertTrue(
                torch.equal(checkpoint_state(raw_path)["weight"], torch.tensor([4]))
            )

            priority_path = root / "priority.pt"
            torch.save(
                {
                    "model_state_dict": {"weight": torch.tensor([1])},
                    "state_dict": {"weight": torch.tensor([2])},
                    "model": {"weight": torch.tensor([3])},
                },
                priority_path,
            )
            self.assertTrue(
                torch.equal(
                    checkpoint_state(priority_path)["weight"],
                    torch.tensor([1]),
                )
            )

    def test_checkpoint_extractors_preserve_existing_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsupported = root / "unsupported.pt"
            missing_model = root / "missing_model.pt"
            missing_optimizer = root / "missing_optimizer.pt"
            torch.save([1, 2, 3], unsupported)
            torch.save({"step": 5}, missing_model)
            torch.save({"model_state_dict": {"weight": torch.ones(1)}}, missing_optimizer)

            with self.assertRaisesRegex(TypeError, "Unsupported checkpoint payload"):
                checkpoint_state(unsupported)
            with self.assertRaisesRegex(KeyError, "No model state dict found"):
                checkpoint_state(missing_model)
            with self.assertRaisesRegex(KeyError, "No optimizer state dict found"):
                checkpoint_optimizer_state(missing_optimizer)

    def test_training_checkpoint_uses_one_cpu_deserialization(self) -> None:
        path = Path("checkpoint.pt")
        model_state = {"weight": torch.ones(1)}
        optimizer_state = {"state": {}, "param_groups": []}
        payload = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "step": "17",
        }

        with mock.patch.object(
            residual_runtime.torch,
            "load",
            return_value=payload,
        ) as load_mock:
            checkpoint = load_training_checkpoint(path)

        load_mock.assert_called_once_with(
            path,
            map_location="cpu",
            weights_only=False,
        )
        self.assertIs(checkpoint.model_state, model_state)
        self.assertIs(checkpoint.optimizer_state, optimizer_state)
        self.assertEqual(checkpoint.step, 17)

    def test_optimizer_state_to_cpu_deeply_isolates_the_clone(self) -> None:
        state_tensor = torch.tensor([1.0], requires_grad=True)
        source = {
            "state": {
                0: {
                    "exp_avg": state_tensor,
                    "metadata": {"labels": ["source"]},
                }
            },
            "param_groups": [{"params": [0], "options": {"betas": [0.9, 0.99]}}],
        }

        cloned = optimizer_state_to_cpu(source)
        cloned_tensor = cloned["state"][0]["exp_avg"]
        self.assertEqual(cloned_tensor.device.type, "cpu")
        self.assertFalse(cloned_tensor.requires_grad)
        self.assertNotEqual(cloned_tensor.data_ptr(), state_tensor.data_ptr())

        source["state"][0]["metadata"]["labels"].append("changed")
        source["param_groups"][0]["options"]["betas"].append(1.0)
        with torch.no_grad():
            state_tensor.fill_(9.0)
        self.assertEqual(cloned["state"][0]["metadata"]["labels"], ["source"])
        self.assertEqual(cloned["param_groups"][0]["options"]["betas"], [0.9, 0.99])
        self.assertTrue(torch.equal(cloned_tensor, torch.tensor([1.0])))

        cloned["state"][0]["metadata"]["labels"].append("clone")
        self.assertEqual(
            source["state"][0]["metadata"]["labels"],
            ["source", "changed"],
        )

    def test_write_json_atomically_replaces_and_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("old result\n", encoding="utf-8")

            write_json(path, {"z": 1, "a": [2]})
            text = path.read_text(encoding="utf-8")
            self.assertEqual(json.loads(text), {"a": [2], "z": 1})
            self.assertLess(text.index('"a"'), text.index('"z"'))
            self.assertTrue(text.endswith("\n"))
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

            stable = path.read_bytes()
            with self.assertRaises(ValueError):
                write_json(path, {"invalid": float("nan")})
            self.assertEqual(path.read_bytes(), stable)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_token_batch_offset_and_hash_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.txt"
            path.write_text("0 1 2 3\n4 5 6 7\n", encoding="utf-8")
            batches = load_token_batches(
                path,
                IntegerTokenizer(),
                batch_size=1,
                seq_length=2,
                num_batches=2,
                batch_offset=1,
            )

            expected = (torch.tensor([[2, 3]]), torch.tensor([[4, 5]]))
            for batch, input_ids in zip(batches, expected):
                self.assertTrue(torch.equal(batch["input_ids"], input_ids))
                self.assertTrue(torch.equal(batch["labels"], input_ids))
                self.assertNotEqual(
                    batch["input_ids"].data_ptr(),
                    batch["labels"].data_ptr(),
                )

            digest = hashlib.sha256()
            for input_ids in expected:
                digest.update(input_ids.contiguous().numpy().tobytes())
            self.assertEqual(batch_hash(batches), digest.hexdigest())
            self.assertNotEqual(batch_hash(batches), batch_hash(list(reversed(batches))))

            with self.assertRaisesRegex(RuntimeError, "Only found 8 tokens"):
                load_token_batches(
                    path,
                    IntegerTokenizer(),
                    batch_size=1,
                    seq_length=2,
                    num_batches=4,
                    batch_offset=1,
                )


if __name__ == "__main__":
    unittest.main()
