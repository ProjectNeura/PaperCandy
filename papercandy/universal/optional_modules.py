from papercandy.universal import config as _cfg


try:
    if _cfg.ConfigContainer().CURRENT.get_predefined("gpu_acceleration", True):
        raise ImportError
    import cupy as _np
except ImportError:
    import numpy as _np


try:
    import coota as _coota
except ImportError:
    _coota = None


def coota_is_available() -> bool:
    return _coota is not None
