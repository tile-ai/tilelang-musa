from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "get_runtime_device_type": (".config", "get_runtime_device_type"),
    "_resolve_big_fuse_config": (".mhc.pre_big_fuse", "_resolve_big_fuse_config"),
    "_mhc_pre_big_fuse": (".mhc.pre_big_fuse_kernel", "_mhc_pre_big_fuse"),
    "expand_to_fused": (".moe.expand_to_fused_kernel", "expand_to_fused"),
    "reduce_fused": (".moe.reduce_fused_kernel", "reduce_fused"),
    "topk_gate": (".moe.topk_gate_kernel", "topk_gate"),
    "topk_sum_and_topk_group_idx": (
        ".moe.topk_sum_and_topk_group_idx_kernel",
        "topk_sum_and_topk_group_idx",
    ),
    "per_block_cast": (".quant.per_block_cast_kernel", "per_block_cast"),
    "per_token_cast": (".quant.per_token_cast_kernel", "per_token_cast"),
    "swiglu_forward_and_per_token_cast": (
        ".quant.swiglu_forward_and_per_token_cast_kernel",
        "swiglu_forward_and_per_token_cast",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
