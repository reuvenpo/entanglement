"""This file includes function used in the visibility experiment"""
import os.path

import numpy as np
import scipy as sp

from .typing import NDFloat64Array
from . import parse
from . import plot
from . import statistics


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


def run_visibility(path):
    corr = parse.Correlation.from_file(path)
    file_name, _ = os.path.splitext(os.path.basename(path))
    print(file_name)

    v_estimates = []
    v_estimate_errs = []
    for x in [corr.alpha_0, corr.alpha_45, corr.alpha_90, corr.alpha_135]:
        v_estimate, v_estimate_err = find_with_estimate(x)
        v_estimates.append(v_estimate)
        v_estimate_errs.append(v_estimate_err)

    alpha_0_fit_params, alpha_0_fit_errs = find_with_least_squares(corr.beta, corr.alpha_0)
    alpha_45_fit_params, alpha_45_fit_errs = find_with_least_squares(corr.beta, corr.alpha_45)
    alpha_90_fit_params, alpha_90_fit_errs = find_with_least_squares(corr.beta, corr.alpha_90)
    alpha_135_fit_params, alpha_135_fit_errs = find_with_least_squares(corr.beta, corr.alpha_135)
    fit_params = [alpha_0_fit_params, alpha_45_fit_params, alpha_90_fit_params, alpha_135_fit_params]
    fit_errs = [alpha_0_fit_errs, alpha_45_fit_errs, alpha_90_fit_errs, alpha_135_fit_errs]

    p = plot.Plot(
        file_name, "Polarizer 2 angle (beta) [deg]", "Coincidence count"
    )
    p.plot("alpha=0", corr.beta, corr.alpha_0, style="o")
    p.plot("alpha=45", corr.beta, corr.alpha_45, style="o")
    p.plot("alpha=90", corr.beta, corr.alpha_90, style="o")
    p.plot("alpha=135", corr.beta, corr.alpha_135, style="o")
    fit_beta = np.linspace(corr.beta.min(), corr.beta.max())
    p.plot("fit alpha=0", fit_beta, fit(fit_beta, *alpha_0_fit_params), style="-")
    p.plot("fit alpha=45", fit_beta, fit(fit_beta, *alpha_45_fit_params), style="-")
    p.plot("fit alpha=90", fit_beta, fit(fit_beta, *alpha_90_fit_params), style="-")
    p.plot("fit alpha=135", fit_beta, fit(fit_beta, *alpha_135_fit_params), style="-")
    p.ax.set_xticks(np.linspace(0, np.pi, 5), ["0", "π/4", "π/2", "3π/4", "π"])
    p.save(f"output/correlation - {file_name}.png")

    print("The visibility at different X angles")
    for indx in range(4):
        a, beta_c, v, p = fit_params[indx]
        a_err, beta_c_err, v_err, p_err = fit_errs[indx]
        v_fit_rel_err = v_err/v
        v_estimate = v_estimates[indx]
        v_estimate_err = v_estimate_errs[indx]
        v_estimate_rel_err = v_estimate_err/v_estimate
        print(
            f"X={indx*45}°:\t"
            f"fit: {v:.2%} ± {v_err:.2%} ({v_fit_rel_err:.2%}), "
            f"estimate: {v_estimate:.2%} ± {v_estimate_err:.2%} ({v_estimate_rel_err:.1%}), "
            f"n_sigma: {statistics.nsigma(v, v_err, v_estimate, v_estimate_err):.2}"
        )
    print()
