from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import register_target_detector


def _target_ffi_api():
    from tilelang import _ffi_api

    return _ffi_api


def check_musa_availability() -> bool:
    try:
        from tilelang.contrib import mcc

        mcc.find_musa_path()
        return True
    except Exception:
        return False


def _detect_musa_target() -> str | None:
    if check_musa_availability():
        return "musa"
    return None


def target_is_musa(target: Target) -> bool:
    return _target_ffi_api().TargetIsMusa(target)


def target_is_qy2(target: Target) -> bool:
    return _target_ffi_api().TargetIsQY2(target)


def target_is_ph1(target: Target) -> bool:
    return _target_ffi_api().TargetIsPH1(target)


register_target_detector("musa", _detect_musa_target, override=True)
