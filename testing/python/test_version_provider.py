from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import shutil
import subprocess
import uuid
import warnings
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_runtime_compute_version(module_file: Path):
    source_path = REPO_ROOT / "tilelang" / "__init__.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_compute_version")
    function_module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(function_module)
    namespace = {
        "Path": Path,
        "warnings": warnings,
        "__file__": str(module_file),
    }
    exec(compile(function_module, str(source_path), "exec"), namespace)
    return namespace["_compute_version"]


def _clear_version_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CUDA_VERSION",
        "NO_GIT_VERSION",
        "NO_TOOLCHAIN_VERSION",
        "NO_VERSION_LABEL",
        "TILELANG_BUILD_WHEEL_WITH_DATE",
        "USE_CUDA",
        "USE_ROCM",
    ):
        monkeypatch.delenv(name, raising=False)


def test_installed_package_uses_tilelang_musa_distribution_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    requested_distributions = []

    def fake_version(distribution_name: str) -> str:
        requested_distributions.append(distribution_name)
        return "1.2.3+musa"

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    compute_version = _load_runtime_compute_version(tmp_path / "site-packages" / "tilelang" / "__init__.py")

    assert compute_version() == "1.2.3+musa"
    assert requested_distributions == ["tilelang_musa"]


def _load_provider(root: Path):
    spec = importlib.util.spec_from_file_location(
        f"_tilelang_test_version_provider_{uuid.uuid4().hex}",
        root / "version_provider.py",
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_provider_tree(root: Path) -> None:
    shutil.copy(REPO_ROOT / "version_provider.py", root / "version_provider.py")
    (root / "VERSION").write_text("1.2.3\n", encoding="utf-8")


def test_extracted_sdist_uses_plain_version_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_version_env(monkeypatch)
    _make_provider_tree(tmp_path)
    (tmp_path / ".git_commit.txt").write_text("abcdef1234567890\n", encoding="utf-8")

    provider = _load_provider(tmp_path)

    assert provider.dynamic_metadata("version") == "1.2.3"


def test_extracted_sdist_can_opt_into_version_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_version_env(monkeypatch)
    monkeypatch.setenv("NO_VERSION_LABEL", "OFF")
    monkeypatch.setenv("NO_TOOLCHAIN_VERSION", "ON")
    _make_provider_tree(tmp_path)
    (tmp_path / ".git_commit.txt").write_text("abcdef1234567890\n", encoding="utf-8")

    provider = _load_provider(tmp_path)

    assert provider.dynamic_metadata("version") == "1.2.3+gitabcdef12"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required for checkout version test")
def test_git_checkout_uses_version_label_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_version_env(monkeypatch)
    monkeypatch.setenv("NO_TOOLCHAIN_VERSION", "ON")
    _make_provider_tree(tmp_path)

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "TileLang Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "tilelang-test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "VERSION", "version_provider.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        encoding="utf-8",
    ).stdout.strip()

    provider = _load_provider(tmp_path)

    assert provider.dynamic_metadata("version") == f"1.2.3+git{commit[:8]}"
    assert (tmp_path / ".git_commit.txt").read_text(encoding="utf-8") == commit
