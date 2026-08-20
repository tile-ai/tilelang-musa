"""MUSA dialect of ``T.Kernel`` with SIMT launch dimensions."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.kernel import KernelLaunchFrame, kernel_launch_factory, launch_kernel

__all__ = ["Kernel"]


@kernel_launch_factory
def Kernel(
    *blocks: int | tirx.PrimExpr,
    threads: int | list[int] | tuple[int, ...] | None = None,
) -> KernelLaunchFrame:
    """Construct a MUSA kernel launch frame.

    Parameters
    ----------
    *blocks : int | PrimExpr
        Grid extent along each axis (one to three dimensions).
    threads : int | list[int] | tuple[int, ...], optional
        Threads per block as a count or up to three per-dimension extents.
        The MUSA pipeline supplies its default when omitted.
    """
    return launch_kernel(blocks, threads=threads)
