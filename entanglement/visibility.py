"""This file includes function used in the visibility experiment"""
import numpy as np
import scipy as sp

from .typing import NDFloat64Array


def estimate(c_min, c_max):
    """An estimate of the visibility"""
    return (c_max - c_min)/(c_max + c_min)


def estimate_err(c_min, c_max):
    """The error in the estimate of the visibility based on the gaussian error propagation rule.

    The errors in c_min and c_max are assumed to equal their square roots.
    """
    return 2 * (c_min*c_max)**(1/2) / (c_min + c_max)**(3/2)


def fit(beta, a, beta_c, v, p):
    """The function to fit to find the visibility.

    To find the visibility, fit this function to a correlation curve using least squares minimization.

    The parameters are:
    :param beta: The relative angle between the polarizers
    :param a: Curve amplitude
    :param beta_c: Curve center
    :param v: Visibility
    :param p: Periodicity
    :return: The fitted coincidence count at `beta`
    """
    return a / 2 * (1 - v * np.sin((beta - beta_c) / p))


def find_with_estimate(coincidence: NDFloat64Array):
    return estimate(coincidence.min(), coincidence.max()), estimate_err(coincidence.min(), coincidence.max())


def find_with_least_squares(beta, coincidence: NDFloat64Array):
    res = sp.optimize.curve_fit(
        fit,
        beta,
        coincidence,
        [coincidence.max(), np.pi, 1, 1],
        bounds=([-np.inf, -np.inf, 0, -np.inf], [np.inf, np.inf, 1, np.inf]),
    )
    popt = res[0]
    pcov = res[1]
    return popt, pcov.diagonal()**(1/2)
