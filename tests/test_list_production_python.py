import subprocess
from pathlib import Path

from scripts.list_production_python import production_python_files, run_mypy


def _write(path: Path, content: str = "VALUE = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_production_files_follow_git_and_filter_non_production(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    included = ["dacp/live.py", "space dir/kept file.py"]
    excluded = [
        "tests/test_live.py",
        "baselines/excp/upstream/reference.py",
    ]
    deleted = "experiments/deleted.py"
    untracked = "scripts/untracked.py"
    ignored = "build/generated.py"

    for relative_path in [*included, *excluded, deleted, untracked, ignored]:
        _write(tmp_path / relative_path)
    (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", *included, *excluded, deleted],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / deleted).unlink()

    assert production_python_files(tmp_path) == sorted([*included, untracked])


def test_run_mypy_passes_spaced_paths_in_process(tmp_path: Path, capsys) -> None:
    calls: list[list[str]] = []

    def fake_runner(arguments: list[str]) -> tuple[str, str, int]:
        calls.append(arguments)
        assert Path.cwd() == tmp_path
        return "checked\n", "diagnostic\n", 7

    original_cwd = Path.cwd()
    status = run_mypy(
        tmp_path,
        ["space dir/kept file.py", "dacp/live.py"],
        ["--explicit-package-bases"],
        runner=fake_runner,
    )

    assert status == 7
    assert Path.cwd() == original_cwd
    assert calls == [
        [
            "--explicit-package-bases",
            "space dir/kept file.py",
            "dacp/live.py",
        ]
    ]
    captured = capsys.readouterr()
    assert captured.out == "checked\n"
    assert captured.err == "diagnostic\n"
