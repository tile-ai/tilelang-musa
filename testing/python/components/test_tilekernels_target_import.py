import ast
from importlib import import_module
from pathlib import Path

import pytest


def test_benchmark_tilekernels_quant_common_imports():
    module = import_module("benchmark.mp31.tilekernels.quant.common")

    assert module.determine_target.__module__ == "tilelang.backend.target"


@pytest.mark.parametrize(
    "relative_path",
    [
        "benchmark/mp31/tilekernels/quant/common.py",
        "tilekernels/tile_kernels/quant/common.py",
    ],
)
def test_tilekernels_quant_common_uses_current_target_module(relative_path):
    tree = ast.parse(Path(relative_path).read_text(encoding="utf-8"))

    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "tilelang.backend.target"
        and any(alias.name == "determine_target" for alias in node.names)
        for node in ast.walk(tree)
    )
