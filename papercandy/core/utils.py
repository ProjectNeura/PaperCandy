import numpy as _np
from typing import Any


def angle2radian(angle: float) -> float:
    return _np.pi * angle / 180


def radian2angle(radian: float) -> float:
    return radian * 180 / _np.pi


def assume_type_matches(obj: Any) -> Any:
    return obj
