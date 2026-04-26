import functools
import torch

_num_sms = 0


def _get_device_module():
    if hasattr(torch, 'musa') and torch.musa.is_available():
        return torch.musa
    raise RuntimeError('MUSA is required for mp31/tilekernels')


def get_device() -> torch.device:
    if hasattr(torch, 'musa') and torch.musa.is_available():
        return torch.device('musa')
    raise RuntimeError('MUSA is required for mp31/tilekernels')


def get_runtime_device_type() -> str:
    _get_device_module()
    return 'musa'


@functools.lru_cache(maxsize=None)
def get_device_num_sms() -> int:
    device_mod = _get_device_module()
    prop = device_mod.get_device_properties(device_mod.current_device())
    return prop.multi_processor_count


def set_num_sms(num_sms: int) -> None:
    global _num_sms
    assert 0 < num_sms <= get_device_num_sms()
    _num_sms = num_sms


def get_num_sms() -> int:
    global _num_sms
    if _num_sms == 0:
        return get_device_num_sms()
    return _num_sms


@functools.lru_cache(maxsize=None)
def get_max_smem_per_sm() -> int:
    return 192 * 1024
