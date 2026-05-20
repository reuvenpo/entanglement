from enum import IntEnum

import numpy as np


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


def phi_name_latex(sign: Sign):
    if sign is Sign.Minus:
        return r"$\Phi^-$"
    if sign is Sign.Plus:
        return r"$\Phi^+$"
    return "?"


def rad_to_deg(rad):
    return rad * 180 / np.pi


def mod_dist(to, from_, mod):
    """Distance in modulo space. Result in [-mod/2, mod/2]. Sign indicates direction."""
    return (to - from_ + mod / 2) % mod - mod / 2
