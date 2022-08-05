from numpy import pi
from typing import Any


def angle2radian(angle: float) -> float:
    return pi * angle / 180


def radian2angle(radian: float) -> float:
    return radian * 180 / pi


def assume_type_matches(obj: Any) -> Any:
    return obj
