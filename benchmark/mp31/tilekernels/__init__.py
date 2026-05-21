from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "get_runtime_device_type": ("tilekernels.config", "get_runtime_device_type"),
    "_resolve_big_fuse_config": ("tilekernels.mhc.pre_big_fuse", "_resolve_big_fuse_config"),
    "_mhc_pre_big_fuse": ("tilekernels.mhc.pre_big_fuse_kernel", "_mhc_pre_big_fuse"),
    "expand_to_fused": ("tilekernels.moe.expand_to_fused_kernel", "expand_to_fused"),
    "reduce_fused": ("tilekernels.moe.reduce_fused_kernel", "reduce_fused"),
    "topk_gate": ("tilekernels.moe.topk_gate_kernel", "topk_gate"),
    "topk_sum_and_topk_group_idx": (
        "tilekernels.moe.topk_sum_and_topk_group_idx_kernel",
        "topk_sum_and_topk_group_idx",
    ),
    "per_block_cast": ("tilekernels.quant.per_block_cast_kernel", "per_block_cast"),
    "per_token_cast": ("tilekernels.quant.per_token_cast_kernel", "per_token_cast"),
    "swiglu_forward_and_per_token_cast": (
        "tilekernels.quant.swiglu_forward_and_per_token_cast_kernel",
        "swiglu_forward_and_per_token_cast",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
