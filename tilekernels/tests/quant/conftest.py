import os
from pathlib import Path

import pytest
import torch


MUSA_FP4_SKIP_REASON = 'MUSA backend does not yet support native FP4/e2m1 codegen'


def _is_musa_target() -> bool:
    return os.environ.get('TILELANG_TARGET', '').strip().startswith('musa')


def _uses_native_e2m1(params: dict) -> bool:
    return params.get('fmt') == 'e2m1' or params.get('in_dtype') == torch.int8


def pytest_collection_modifyitems(items):
    if not _is_musa_target():
        return

    skip_musa_fp4 = pytest.mark.skip(reason=MUSA_FP4_SKIP_REASON)
    for item in items:
        path = Path(str(item.path))
        if path.name == 'test_per_block_cast_lossless.py':
            item.add_marker(skip_musa_fp4)
            continue

        callspec = getattr(item, 'callspec', None)
        if callspec is None:
            continue
        params = callspec.params.get('params')
        if isinstance(params, dict) and _uses_native_e2m1(params):
            item.add_marker(skip_musa_fp4)
