from enum import IntEnum


def perp(deg):
    """The perpendicular of an angle given in degrees"""
    return (deg + 90) % 180


class Sign(IntEnum):
    Minus = -1
    Plus = 1


def phi_name(sign: Sign):
    if sign is Sign.Minus:
        return "Φ-"
    if sign is Sign.Plus:
        return "Φ+"
    return "?"
