"""List or type-check the repository's publishable production Python files.

The file selection is intentionally derived from Git rather than a maintained
directory list.  Tracked files and non-ignored untracked files are included so
new modules cannot escape pre-commit checks.  Tests and the untouched ExCP
reference implementation are excluded.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import cast

EXCLUDED_PREFIXES = ("tests/", "baselines/excp/upstream/")
MypyRunner = Callable[[list[str]], tuple[str, str, int]]


def repository_root(start: Path | None = None) -> Path:
    """Return the Git worktree root containing *start*."""
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return Path(completed.stdout.strip()).resolve()


def production_python_files(repo_root: Path) -> list[str]:
    """Return existing, publishable production Python paths under *repo_root*."""
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.py",
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tracked_paths = (os.fsdecode(raw) for raw in completed.stdout.split(b"\0") if raw)

    selected: list[str] = []
    for raw_path in tracked_paths:
        path = PurePosixPath(raw_path).as_posix()
        if path.startswith(EXCLUDED_PREFIXES):
            continue
        if (repo_root / Path(path)).is_file():
            selected.append(path)
    return sorted(selected)


def run_mypy(
    repo_root: Path,
    files: Sequence[str],
    mypy_args: Sequence[str],
    *,
    runner: MypyRunner | None = None,
) -> int:
    """Run mypy in-process, avoiding platform command-line length limits."""
    if not files:
        raise RuntimeError("no production Python files were found")
    if runner is None:
        from mypy.api import run as mypy_runner

        executor = cast(MypyRunner, mypy_runner)
    else:
        executor = runner

    previous_cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        stdout, stderr, exit_status = executor([*mypy_args, *files])
    finally:
        os.chdir(previous_cwd)

    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, end="", file=sys.stderr)
    return int(exit_status)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("list", "mypy"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--repo",
            type=Path,
            help="Git worktree to inspect (defaults to the current worktree)",
        )
        if command == "mypy":
            subparser.add_argument(
                "mypy_args",
                nargs=argparse.REMAINDER,
                help="arguments passed to mypy; place them after --",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo.resolve() if args.repo else repository_root()
    files = production_python_files(repo_root)
    if args.command == "list":
        print(*files, sep="\n")
        return 0

    mypy_args = args.mypy_args
    if mypy_args[:1] == ["--"]:
        mypy_args = mypy_args[1:]
    return run_mypy(repo_root, files, mypy_args)


if __name__ == "__main__":
    raise SystemExit(main())
