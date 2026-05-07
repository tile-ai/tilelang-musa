import os
import random
import pytest


DEFAULT_TEST_SEED = int(os.environ.get("TL_TEST_SEED", "0"))


def seed_everything(seed: int = DEFAULT_TEST_SEED):
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)


@pytest.fixture(scope="function", autouse=True)
def _reset_random_seed():
    seed_everything(DEFAULT_TEST_SEED)


@pytest.fixture(scope="session", autouse=True)
def _configure_torch_backend():
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.mudnn.deterministic = True
        torch.backends.mudnn.allow_tf32 = False


collect_ignore = [
    "python/cache",
    "python/carver",
    "python/cpu",
    "python/autotune",
    "python/tilelibrary",
    "python/kernel/test_tilelang_kernel_bf16_gemm_mma.py",
    "python/kernel/test_tilelang_kernel_fp8_gemm_mma.py",
    "python/kernel/test_tilelang_kernel_gemm_mma_intrinsic.py",
    "python/kernel/test_tilelang_kernel_int4_gemm_mma.py",
    "python/language/test_tilelang_language_cooperative.py",
    "python/language/test_tilelang_language_pdl.py",
    "python/language/test_tilelang_language_cluster.py",
    "python/language/test_tilelang_language_async_copy_gemm_sm80.py",
    "python/language/test_tilelang_language_intrinsics_codegen.py",
    "python/transform/test_tilelang_transform_lower_ldgstg.py",
    "python/jit/test_tilelang_jit_parcompile.py",
]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Ensure that at least one test is collected. Error out if all tests are skipped."""
    known_types = {
        "failed",
        "passed",
        "skipped",
        "deselected",
        "xfailed",
        "xpassed",
        "warnings",
        "error",
    }
    if sum(len(terminalreporter.stats.get(k, [])) for k in known_types.difference({"skipped", "deselected"})) == 0:
        terminalreporter.write_sep(
            "!",
            (f"Error: No tests were collected. {dict(sorted((k, len(v)) for k, v in terminalreporter.stats.items()))}"),
        )
        pytest.exit("No tests were collected.", returncode=5)
