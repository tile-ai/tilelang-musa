"""MUSA device-side random number generation DSL APIs (murand)."""

from __future__ import annotations

from tvm import tirx
import tilelang.language.common as T

__all__ = ["rng_init", "rng_rand", "rng_rand_float"]

_SUPPORTED_GENERATORS = (
    "curandStateMRG32k3a_t",
    "curandStatePhilox4_32_10_t",
    "curandStateXORWOW_t",
    "murandStateMRG32k3a_t",
    "murandStatePhilox4_32_10_t",
    "murandStateXORWOW_t",
)


def rng_init(seed, seq=None, off=0, generator="curandStatePhilox4_32_10_t") -> tirx.PrimExpr:
    """Initialize MUSA murand random number generator state.

    Parameters
    ----------
    seed : PrimExpr
        Random seed value.
    seq : PrimExpr, optional
        Sequence number for parallel random number generation. Defaults to
        the linear thread index ``tx + bx * threads``.
    off : PrimExpr, optional
        Offset number for parallel random number generation.
    generator : str, optional
        Generator state type name. Both CUDA-compatible ``curandState*`` names
        and native ``murandState*`` names are accepted.

    Returns
    -------
    state : PrimExpr
        The RNG initialization call expression.
    """
    if generator not in _SUPPORTED_GENERATORS:
        raise ValueError(
            f"Unsupported MUSA RNG generator: {generator!r}. "
            f"Expected one of {_SUPPORTED_GENERATORS}."
        )
    seed = tirx.convert(seed)
    if seq is None:
        bx = T.get_block_binding()
        ex = T.get_thread_extent()
        tx = T.get_thread_binding()
        seq = tirx.convert(tx + bx * ex)
    else:
        seq = tirx.convert(seq)
    off = tirx.convert(off)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.rng_init"), seed, seq, off, generator)


def rng_rand() -> tirx.PrimExpr:
    """Generate a 32-bit unsigned random integer via ``murand``."""
    return tirx.call_intrin("uint32", tirx.op.Op.get("tl.rng_rand"))


def rng_rand_float(bit=32, dist="uniform") -> tirx.PrimExpr:
    """Generate a random floating-point value via ``murand_uniform``/``murand_normal``.

    Parameters
    ----------
    bit : int, optional
        Bitwidth of the generated float (``32`` or ``64``).
    dist : str, optional
        Distribution (``"uniform"`` or ``"normal"``).
    """
    if bit not in (32, 64):
        raise ValueError(f"Unsupported bitwidth for rng_rand_float: {bit}. Expected 32 or 64.")
    if dist not in ("uniform", "normal"):
        raise ValueError(
            f"Unsupported distribution for rng_rand_float: {dist!r}. "
            "Expected 'uniform' or 'normal'."
        )
    return tirx.call_intrin(f"float{bit}", tirx.op.Op.get("tl.rng_rand_float"), dist)
