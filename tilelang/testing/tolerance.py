from __future__ import annotations

from collections.abc import Mapping

import torch


ToleranceMap = Mapping[torch.dtype, tuple[float, float]]


BASELINE_PROFILE: ToleranceMap = {
    torch.float32: (1e-5, 1e-6),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (7.9e-3, 7.9e-3),
    torch.float8_e4m3fn: (1.25e-1, 1.25e-1),
}


TOLERANCE_PROFILES: dict[str, ToleranceMap] = {
    "baseline": BASELINE_PROFILE,
    # Contract check for kernels that cast outputs to input dtype after higher-precision accumulation.
    "gemm_contract": BASELINE_PROFILE,
    # Algorithmic check against a float32 oracle before final output cast.
    "gemm_algorithm": {
        torch.float32: (1e-5, 1e-6),
        torch.float16: (2e-2, 2e-2),
        torch.bfloat16: (5e-2, 5e-2),
        torch.float8_e4m3fn: (2.5e-1, 2.5e-1),
    },
}


def get_tolerance(
    dtype: torch.dtype,
    profile: str = "baseline",
    *,
    overrides: ToleranceMap | None = None,
) -> tuple[float, float]:
    """Return (rtol, atol) for a dtype under the selected tolerance profile.

    Parameters
    ----------
    dtype : torch.dtype
        Tensor dtype to query.
    profile : str
        Tolerance profile name. Built-in values include:
        - ``baseline``
        - ``gemm_contract``
        - ``gemm_algorithm``
    overrides : Mapping[torch.dtype, tuple[float, float]] | None
        Optional per-test local overrides. If provided and ``dtype`` exists in
        this mapping, the override value takes precedence over profile defaults.
    """
    if profile not in TOLERANCE_PROFILES:
        supported = ", ".join(sorted(TOLERANCE_PROFILES))
        raise ValueError(f"Unknown tolerance profile: {profile}. Supported profiles: {supported}")

    if overrides is not None and dtype in overrides:
        return overrides[dtype]

    profile_map = TOLERANCE_PROFILES[profile]
    if dtype not in profile_map:
        raise KeyError(f"dtype {dtype} is not configured for tolerance profile '{profile}'")

    return profile_map[dtype]


def list_tolerance_profiles() -> tuple[str, ...]:
    """List all built-in tolerance profile names."""
    return tuple(sorted(TOLERANCE_PROFILES))
