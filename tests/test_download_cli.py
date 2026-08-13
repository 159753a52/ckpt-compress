import json
from pathlib import Path
from unittest import mock

import pytest

from scripts import download_data, download_models


def _write_cifar_cache(root: Path, dataset_name: str) -> None:
    directory_name, filenames = download_data.CIFAR_DATASET_FILES[dataset_name]
    dataset_dir = root / directory_name
    dataset_dir.mkdir(parents=True)
    for filename in filenames:
        (dataset_dir / filename).write_bytes(b"payload")


def _write_hf_cache(root: Path, dataset_name: str, *, builder_name: str | None = None) -> None:
    spec = download_data.HF_DATASET_SPECS[dataset_name]
    cache_dir = root / spec.builder_name / spec.config_name / "1.0.0" / "complete"
    cache_dir.mkdir(parents=True)
    splits = {}
    for split in spec.required_splits:
        filename = f"{dataset_name}-{split}.arrow"
        (cache_dir / filename).write_bytes(b"arrow")
        splits[split] = {
            "name": split,
            "num_bytes": 5,
            "num_examples": 1,
            "filename": filename,
        }
    (cache_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "builder_name": builder_name or spec.builder_name,
                "config_name": spec.config_name,
                "splits": splits,
            }
        ),
        encoding="utf-8",
    )


def test_model_download_success_and_failure_exit_codes(tmp_path: Path) -> None:
    with mock.patch.object(download_models, "download_gpt2_small", return_value=True) as download:
        assert download_models.main(["--model", "gpt2-small", "--cache_dir", str(tmp_path)]) == 0
    download.assert_called_once_with(str(tmp_path))

    with mock.patch.object(download_models, "download_gpt2_small", return_value=False):
        assert download_models.main(["--model", "gpt2-small", "--cache_dir", str(tmp_path)]) == 1


def test_data_download_success_and_failure_exit_codes(tmp_path: Path) -> None:
    with mock.patch.object(download_data, "download_cifar10", return_value=True) as download:
        assert download_data.main(["--dataset", "cifar10", "--data_dir", str(tmp_path)]) == 0
    download.assert_called_once_with(str(tmp_path))

    with mock.patch.object(download_data, "download_cifar10", return_value=False):
        assert download_data.main(["--dataset", "cifar10", "--data_dir", str(tmp_path)]) == 1


@pytest.mark.parametrize(
    ("module", "selection", "all_args"),
    [
        (download_models, "--model", ["--all"]),
        (download_data, "--dataset", ["--all"]),
    ],
)
def test_all_and_specific_selection_are_mutually_exclusive(
    module: object, selection: str, all_args: list[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        module.main([selection, "unused", *all_args])  # type: ignore[attr-defined]
    assert exc_info.value.code != 0


@pytest.mark.parametrize(
    ("module", "option", "value"),
    [
        (download_models, "--model", ""),
        (download_models, "--model", "gpt2-small,"),
        (download_models, "--model", "unknown"),
        (download_data, "--dataset", ""),
        (download_data, "--dataset", "cifar10,"),
        (download_data, "--dataset", "unknown"),
        (download_data, "--verify", "unknown"),
    ],
)
def test_unknown_and_empty_selections_exit_nonzero(module: object, option: str, value: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        module.main([option, value])  # type: ignore[attr-defined]
    assert exc_info.value.code != 0


@pytest.mark.parametrize("dataset_name", download_data.DATASET_NAMES)
def test_verify_dataset_requires_matching_completed_cache(
    tmp_path: Path, dataset_name: str
) -> None:
    assert not download_data.verify_dataset(dataset_name, str(tmp_path))

    if dataset_name in download_data.CIFAR_DATASET_FILES:
        _write_cifar_cache(tmp_path, dataset_name)
    else:
        _write_hf_cache(tmp_path, dataset_name)
    assert download_data.verify_dataset(dataset_name, str(tmp_path))


def test_verify_cifar_requires_all_key_files(tmp_path: Path) -> None:
    _write_cifar_cache(tmp_path, "cifar10")
    (tmp_path / "cifar-10-batches-py" / "test_batch").unlink()

    assert not download_data.verify_dataset("cifar10", str(tmp_path))


def test_verify_hf_rejects_wrong_builder_and_missing_split_file(tmp_path: Path) -> None:
    _write_hf_cache(tmp_path, "sst2", builder_name="wikitext")
    assert not download_data.verify_dataset("sst2", str(tmp_path))

    wrong_root = tmp_path / "wrong-builder"
    _write_hf_cache(wrong_root, "sst2")
    (wrong_root / "glue" / "sst2" / "1.0.0" / "complete" / "sst2-validation.arrow").unlink()
    assert not download_data.verify_dataset("sst2", str(wrong_root))


def test_verify_not_found_returns_nonzero(tmp_path: Path) -> None:
    assert download_data.main(["--verify", "cifar10", "--data_dir", str(tmp_path)]) == 1
