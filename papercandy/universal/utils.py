from typing import Any


from papercandy.universal.optional_modules import _np


def angle2radian(angle: float) -> float:
    return _np.pi * angle / 180


def radian2angle(radian: float) -> float:
    return radian * 180 / _np.pi


def assume_type_matches(obj: Any) -> Any:
    return obj
