# dlx/utils/backend.py
import os
import random
import numpy as np

_has_cupy = False
try:
    import cupy
    if cupy.cuda.is_available():
        _has_cupy = True
except (ImportError, ModuleNotFoundError):
    pass


class Device:
    def __init__(self, device_type: str = "cpu"):
        if device_type == "cuda" and not _has_cupy:
            raise RuntimeError("CUDA requested but cupy is not available")
        if device_type not in ("cpu", "cuda"):
            raise ValueError(f"Unknown device: {device_type}")
        self._type = device_type

    @property
    def xp(self):
        """The array module (numpy or cupy) for this device."""
        if self._type == "cuda":
            import cupy
            return cupy
        return np

    @property
    def type(self) -> str:
        return self._type

    def __eq__(self, other):
        if isinstance(other, Device):
            return self._type == other._type
        if isinstance(other, str):
            return self._type == other
        return NotImplemented

    def __hash__(self):
        return hash(self._type)

    def __repr__(self):
        return f"device('{self._type}')"

# singletons
cpu = Device("cpu")
cuda = Device("cuda") if _has_cupy else None

_force_cpu = os.environ.get("DLX_FORCE_CPU", "") == "1"
_default_device: Device = cuda if (_has_cupy and not _force_cpu) else cpu

def get_default_device() -> Device:
    return _default_device

def set_default_device(device: "str | Device"):
    global _default_device
    if isinstance(device, str):
        device = Device(device)
    _default_device = device

xp = _default_device.xp


def set_seed(seed):
    import torch
    random.seed(seed)
    _default_device.xp.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False