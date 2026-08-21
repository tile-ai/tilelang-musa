from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen


_build_musa = global_func_device_codegen("target.build.musa")


DEVICE_CODEGEN = DeviceCodegen(
    "musa",
    build=_build_musa,
    build_without_compile=_build_musa,
)
