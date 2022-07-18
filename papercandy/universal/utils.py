from numpy import pi


def angle2radian(angle: float) -> float:
    return pi * angle / 180


def radian2angle(radian: float) -> float:
    return radian * 180 / pi
