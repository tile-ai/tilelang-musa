from __future__ import annotations

import torch


FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
)


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    """Return True when dtype is one of PyTorch float8 dtypes."""
    return dtype in FP8_DTYPES


def cast_to_output_dtype(
    tensor: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    saturate_fp8: bool = True,
) -> torch.Tensor:
    """Cast tensor to out_dtype with optional satfinite behavior for float8 outputs.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to cast.
    out_dtype : torch.dtype
        Destination dtype.
    saturate_fp8 : bool
        When True and out_dtype is float8, values are clamped to finfo range
        before cast. This avoids NaN overflow artifacts and matches common
        satfinite writeback behavior in GPU kernels.
    """
    if saturate_fp8 and is_fp8_dtype(out_dtype):
        finfo = torch.finfo(out_dtype)
        return tensor.clamp(finfo.min, finfo.max).to(out_dtype)
    return tensor.to(out_dtype)


def matmul_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    accum_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype | None = None,
    saturate_fp8: bool = True,
) -> torch.Tensor:
    """Compute a matmul reference with explicit accumulation/output dtypes.

    This helper is intended for kernel tests where inputs may be low precision
    (e.g. float8), accumulation should be in fp32, and output writeback
    semantics (including satfinite fp8 casts) should be controlled explicitly.
    """
    ref = torch.matmul(a.to(accum_dtype), b.to(accum_dtype))
    if out_dtype is None:
        return ref
    return cast_to_output_dtype(ref, out_dtype, saturate_fp8=saturate_fp8)
