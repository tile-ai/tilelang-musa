from __future__ import annotations

import tvm
from tvm.target import Target

from .arch_base import TileDevice
from .driver import musa_driver


def check_mp_version(arch: str) -> int:
    mp_version = arch.replace("mp_", "")
    return int(mp_version) if mp_version.isdigit() else -1


def is_musa_arch(arch: TileDevice) -> bool:
    return isinstance(arch, MUSA)


def is_mp31_arch(arch: TileDevice) -> bool:
    return is_musa_arch(arch) and arch.mp_version == 31


def has_musa_mma_support(arch: TileDevice) -> bool:
    return is_musa_arch(arch) and arch.mp_version == 31


mp31_tensorcore_supported = [
    ("bfloat16", "float32"),
    ("float16", "float32"),
    ("float16", "float16"),
    ("int8", "int32"),
]


def is_musa_tensorcore_supported_precision(in_dtype: str, accum_dtype: str, arch: TileDevice) -> bool:
    if is_mp31_arch(arch):
        return (in_dtype, accum_dtype) in mp31_tensorcore_supported
    raise ValueError(f"Unsupported MUSA architecture: {arch}")


class TensorInstruction:
    def __init__(self, name: str, shape: list[int]):
        self.name = name
        self.shape = shape


class MUSA(TileDevice):
    def __init__(self, target: Target | str):
        if isinstance(target, str):
            from tilelang.backend.target import determine_target

            target = determine_target(target, return_object=True)
        self.target = target
        self.mp_version = check_mp_version(str(self.target.attrs["arch"]))
        device = tvm.runtime.musa(0)
        if not device.exist:
            raise RuntimeError("Cannot find MUSA device 0.")
        self.name = musa_driver.get_device_name()
        self.device = device
        self.platform = "MUSA"
        self.smem_cap = musa_driver.get_shared_memory_per_block() or device.max_shared_memory_per_block
        self.compute_max_core = device.multi_processor_count
        self.warp_size = device.warp_size
        self.compute_capability = str(self.target.attrs["arch"])
        self.reg_cap = musa_driver.get_registers_per_block() or 0
        self.max_smem_usage = musa_driver.get_max_dynamic_shared_size_bytes() or self.smem_cap
        self.sm_partition = 4
        self.l2_cache_size_bytes = musa_driver.get_persisting_l2_cache_max_size() or 0
        self.transaction_size = [32, 128]
        self.bandwidth = [750, 12080]
        self.available_tensor_instructions: tuple[TensorInstruction, ...] | None = None

    def get_avaliable_tensorintrin_shapes(self):
        self.available_tensor_instructions = (
            TensorInstruction("mma", [16, 16]),
            TensorInstruction("wmma", [16, 16]),
        )
        return [instruction.shape for instruction in self.available_tensor_instructions]

    def __repr__(self):
        return f"MUSA({self.target})"


__all__ = [
    "check_mp_version",
    "is_musa_arch",
    "is_mp31_arch",
    "has_musa_mma_support",
    "is_musa_tensorcore_supported_precision",
    "MUSA",
]
