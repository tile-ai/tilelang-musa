import gc
import inspect
import os
import sys
from pathlib import Path

import pytest
import torch

# Force TileLang kernels in this extracted test tree onto MUSA MP31.
os.environ.setdefault('TILELANG_TARGET', 'musa -arch=mp_31')

_TESTS_ROOT = Path(__file__).resolve().parent
_PACKAGE_ROOT = _TESTS_ROOT.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


def _has_musa() -> bool:
    return hasattr(torch, 'musa') and torch.musa.is_available()


def _patch_tilelang_shfl_sync() -> None:
    try:
        from tilelang import language as T
    except ImportError:
        return

    shfl_sync = getattr(T, 'shfl_sync', None)
    if shfl_sync is None:
        return

    params = tuple(inspect.signature(shfl_sync).parameters)
    if params[:3] != ('mask', 'value', 'srcLane'):
        return

    def compat_shfl_sync(*args, **kwargs):
        if len(args) == 2 and not kwargs:
            value, src_lane = args
            return shfl_sync(0xFFFFFFFF, value, src_lane)
        return shfl_sync(*args, **kwargs)

    T.shfl_sync = compat_shfl_sync


_patch_tilelang_shfl_sync()


def _reset_compiler_state() -> None:
    compiler = getattr(torch, 'compiler', None)
    if compiler is not None and hasattr(compiler, 'reset'):
        compiler.reset()
        return

    dynamo = getattr(torch, '_dynamo', None)
    if dynamo is not None and hasattr(dynamo, 'reset'):
        dynamo.reset()


def _clear_musa_cache() -> None:
    if not _has_musa():
        return

    torch.musa.empty_cache()
    if hasattr(torch.musa, 'ipc_collect'):
        torch.musa.ipc_collect()


@pytest.fixture(autouse=True)
def cleanup_test_state():
    yield
    gc.collect()
    _reset_compiler_state()
    _clear_musa_cache()


def pytest_collection_modifyitems(config, items):
    if not (_PACKAGE_ROOT / 'tile_kernels').is_dir():
        skip_marker = pytest.mark.skip(
            reason='Local TileKernels source tree not found under testing/musa/mp31/tilekernels.'
        )
        for item in items:
            item.add_marker(skip_marker)
        return

    if _has_musa():
        return

    skip_marker = pytest.mark.skip(reason='MUSA runtime is required for mp31/tilekernels tests.')
    for item in items:
        item.add_marker(skip_marker)

# Root-level conftest
#
# Loads the benchmark plugin (CLI options, markers, fixtures).
# The plugin lives in a file deliberately NOT named conftest.py to
# avoid pluggy's duplicate-registration error.

pytest_plugins = [
    'tests.pytest_random_plugin',
    'tests.pytest_benchmark_plugin',
]
