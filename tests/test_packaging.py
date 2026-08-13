import tomllib
from pathlib import Path

from setuptools import find_namespace_packages

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_config() -> dict:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_runtime_dependencies_cover_direct_library_imports() -> None:
    dependencies = _project_config()["project"]["dependencies"]

    assert any(requirement.startswith("torchvision") for requirement in dependencies)
    assert any(requirement.startswith("scikit-learn") for requirement in dependencies)


def test_package_discovery_excludes_vendored_excp_source() -> None:
    discovery = _project_config()["tool"]["setuptools"]["packages"]["find"]
    packages = find_namespace_packages(
        where=str(PROJECT_ROOT / discovery["where"][0]),
        include=discovery["include"],
        exclude=discovery["exclude"],
    )

    assert "dacp" in packages
    assert "baselines.excp" in packages
    assert "baselines.inshrinkerator" in packages
    assert not any(
        package == "baselines.excp.upstream" or package.startswith("baselines.excp.upstream.")
        for package in packages
    )
