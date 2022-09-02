from papercandy.core import config as _cfg

try:
    if _cfg.CONFIG().CURRENT.get_predefined("gpu_acceleration"):
        raise ImportError
    import cupy as _np

    def cupy_is_available() -> bool:
        return True
except ImportError:
    import numpy as _np

    def cupy_is_available() -> bool:
        return False


class _SimulatedCOOTA(object):
    class Generator(object):
        def generate(self, size: int, *args, parse: bool = True):
            raise NotImplementedError


try:
    import coota as _coota
    _coota_is_available: bool = True
except ImportError:
    _coota = _SimulatedCOOTA
    _coota_is_available: bool = False


def coota_is_available() -> bool:
    return _coota_is_available
