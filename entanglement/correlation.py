"""This file includes function used in the visibility experiment"""
import os.path

import numpy as np
import scipy as sp

from .typing import NDFloat64Array
from . import parse
from . import plot
from . import statistics
from .util import Sign, phi_name


def visibility_estimate(c_min, c_max):
    """An estimate of the visibility"""
    return (c_max - c_min)/(c_max + c_min)


def visibility_estimate_err(c_min, c_max):
    """The error in the estimate of the visibility based on the gaussian error propagation rule.

    The errors in c_min and c_max are assumed to equal their square roots.
    """
    return 2 * (c_min*c_max)**(1/2) / (c_min + c_max)**(3/2)


def visibility_fit(beta, a, beta_c, v, p):
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


def visibility_from_fit(beta: NDFloat64Array, coincidence: NDFloat64Array):
    res = sp.optimize.curve_fit(
        visibility_fit,
        beta,
        coincidence,
        p0=[coincidence.max(), (beta[np.argmin(coincidence)] - np.pi/4) % np.pi, 1, 1],
        sigma=coincidence**(1/2),
        absolute_sigma=True,
        bounds=([-np.inf, -np.inf, 0, -np.inf], [np.inf, np.inf, 1, np.inf]),
    )
    popt = res[0]
    pcov = res[1]
    return popt, pcov.diagonal()**(1/2)


def find_v(corr: parse.Correlation, file_name: str, sign: Sign):
    v_estimates = []
    for x in [corr.alpha_0, corr.alpha_45, corr.alpha_90, corr.alpha_135]:
        v_estimate = visibility_estimate(x.min(), x.max())
        v_estimate_err = visibility_estimate_err(x.min(), x.max())
        v_estimates.append((v_estimate, v_estimate_err))

    alpha_0_fit_params, alpha_0_fit_errs = visibility_from_fit(corr.beta, corr.alpha_0)
    alpha_45_fit_params, alpha_45_fit_errs = visibility_from_fit(corr.beta, corr.alpha_45)
    alpha_90_fit_params, alpha_90_fit_errs = visibility_from_fit(corr.beta, corr.alpha_90)
    alpha_135_fit_params, alpha_135_fit_errs = visibility_from_fit(corr.beta, corr.alpha_135)
    fit_params = [alpha_0_fit_params, alpha_45_fit_params, alpha_90_fit_params, alpha_135_fit_params]
    fit_errs = [alpha_0_fit_errs, alpha_45_fit_errs, alpha_90_fit_errs, alpha_135_fit_errs]

    print(f"Visibility fit for {phi_name(sign)}")
    for i in range(4):
        a, beta_c, v_fit, p = fit_params[i]
        a_err, beta_c_err, v_fit_err, p_err = fit_errs[i]
        v_fit_rel_err = v_fit_err / v_fit
        v_estimate, v_estimate_err = v_estimates[i]
        v_estimate_rel_err = v_estimate_err / v_estimate
        print(
            f"X={i * 45}°:\t"
            f"fit: {v_fit:.2%} ± {v_fit_err:.2%} ({v_fit_rel_err:.2%}), "
            f"estimate: {v_estimate:.2%} ± {v_estimate_err:.2%} ({v_estimate_rel_err:.1%}), "
            f"n_sigma: {statistics.nsigma(v_fit, v_fit_err, v_estimate, v_estimate_err):.2}"
        )
    print()

    p = plot.Plot(f"Visibility fit for {phi_name(sign)}", "Polarizer 2 angle (beta) [deg]", "Coincidence count")
    p.plot_err("alpha=0°", corr.beta, corr.alpha_0, corr.alpha_0**(1/2))
    p.plot_err("alpha=45°", corr.beta, corr.alpha_45, corr.alpha_45**(1/2))
    p.plot_err("alpha=90°", corr.beta, corr.alpha_90, corr.alpha_90**(1/2))
    p.plot_err("alpha=135°", corr.beta, corr.alpha_135, corr.alpha_135**(1/2))
    fit_beta = np.linspace(corr.beta.min(), corr.beta.max())
    p.plot("fit alpha=0°", fit_beta, visibility_fit(fit_beta, *alpha_0_fit_params), style="-")
    p.plot("fit alpha=45°", fit_beta, visibility_fit(fit_beta, *alpha_45_fit_params), style="-")
    p.plot("fit alpha=90°", fit_beta, visibility_fit(fit_beta, *alpha_90_fit_params), style="-")
    p.plot("fit alpha=135°", fit_beta, visibility_fit(fit_beta, *alpha_135_fit_params), style="-")
    p.ax.set_xticks(np.linspace(0, np.pi, 5), ["0", "π/4", "π/2", "3π/4", "π"])
    p.save(f"output/visibility fit for {phi_name(sign)}.png")


def entangled_coincidence_phi(beta, alpha, amp, noise, *, sign: Sign):
    """The chance of two entangled photons in the | \\phi_- > Bell state to both pass a pair of polarizers.

    alpha and beta are the angles of the two polarizers relative to the horizontal state.
    """
    return noise + amp * np.cos(-sign * alpha + beta)**2


def fit_entangled_coincidence_phi(beta: NDFloat64Array, coincidence: NDFloat64Array, sign: Sign):
    p0 = [sign * beta[np.argmax(coincidence)] % np.pi, coincidence.max() - coincidence.min(), coincidence.min()]
    res = sp.optimize.curve_fit(
        lambda *args: entangled_coincidence_phi(*args, sign=sign),
        beta,
        coincidence,
        p0=p0,
        sigma=coincidence**(1/2),
        absolute_sigma=True,
        bounds=([0, 0, 0], [np.pi, np.inf, np.inf]),
        full_output=True,
    )
    popt = res[0]
    pcov = res[1]
    infodict = res[2]
    chi_2_red = np.sum(infodict['fvec']**2) / (len(beta) - len(p0))
    return popt, pcov.diagonal()**(1/2), chi_2_red


def fit_entangled_coincidence_phi_at_alpha(alpha: float, beta: NDFloat64Array, coincidence: NDFloat64Array, sign: Sign):
    p0 = [coincidence.max() - coincidence.min(), coincidence.min()]
    res = sp.optimize.curve_fit(
        lambda b, *args: entangled_coincidence_phi(b, alpha, *args, sign=sign),
        beta,
        coincidence,
        p0=p0,
        sigma=coincidence**(1/2),
        absolute_sigma=True,
        bounds=([0, 0], [np.inf, np.inf]),
        full_output=True,
    )
    popt = res[0]
    pcov = res[1]
    infodict = res[2]
    chi_2_red = np.sum(infodict['fvec']**2) / (len(beta) - len(p0))
    return popt, pcov.diagonal()**(1/2), chi_2_red


def check_compatibility_with_entanglement(corr: parse.Correlation, file_name: str, sign: Sign):
    p = plot.Plot(f"cos squared fit for {phi_name(sign)} with free alpha", "Polarizer 2 angle (beta) [deg]", "Coincidence count")
    fit_beta = np.linspace(corr.beta.min(), corr.beta.max())

    print(f"cos squared fit for {phi_name(sign)} with free alpha")
    fit_params = []
    fit_errs = []
    for i, correlation in enumerate([corr.alpha_0, corr.alpha_45, corr.alpha_90, corr.alpha_135]):
        params, errs, chi_2_red = fit_entangled_coincidence_phi(corr.beta, correlation, sign)
        fit_params.append(params)
        fit_errs.append(errs)

        alpha, amp, noise = params
        alpha_err, amp_err, noise_err = errs

        alpha_deg = alpha*180/np.pi
        p.plot(f"alpha={alpha_deg:.4}°", corr.beta, correlation)
        p.plot(f"fit alpha={alpha_deg:.4}°", fit_beta, entangled_coincidence_phi(fit_beta, alpha, amp, noise, sign=sign), "-")

        alpha_deg_err = alpha_err * 180 / np.pi
        # noise_ratio = noise/amp
        # noise_ratio_err = noise_ratio * ((noise_err/noise)**2 + (amp_err/amp)**2)**(1/2)
        print(
            f"X={i * 45}°:\t"
            f"alpha: {alpha_deg:.5}° ± {alpha_deg_err:.2}° ({alpha_deg_err/alpha_deg:.2%}), "
            f"offset: {(alpha_deg - i*45 + 180/2)%180 - 180/2:.3}°, "  # How many degrees off is the curve's phase?
            # f"amp: {amp:.5} ± {amp_err:.2} ({amp_err/amp:.2%}), "
            # f"noise: {noise:.5} ± {noise_err:.2} ({noise_err/noise:.2%}), "
            # f"noise_ratio: {noise_ratio:.2%} ± {noise_ratio_err:.2%} ({noise_ratio_err/noise_ratio:.2%}), "  # related to visibility
            f"chi2_red: {chi_2_red:.3}, "
        )
    p.ax.set_xticks(np.linspace(0, np.pi, 5), ["0", "π/4", "π/2", "3π/4", "π"])
    p.save(f"output/cos squared fit for {phi_name(sign)} with free alpha.png")

    print()


def check_compatibility_with_entanglement_at_alphas(corr: parse.Correlation, file_name: str, sign: Sign):
    p = plot.Plot(f"cos squared fit for {phi_name(sign)} with preset alpha", "Polarizer 2 angle (beta) [deg]", "Coincidence count")
    fit_beta = np.linspace(corr.beta.min(), corr.beta.max())

    print(f"cos squared fit for {phi_name(sign)} with preset alpha")
    fit_params = []
    fit_errs = []
    for i, correlation in enumerate([corr.alpha_0, corr.alpha_45, corr.alpha_90, corr.alpha_135]):
        expected_alpha = i*np.pi/4
        params, errs, chi_2_red = fit_entangled_coincidence_phi_at_alpha(expected_alpha, corr.beta, correlation, sign)
        fit_params.append(params)
        fit_errs.append(errs)

        amp, noise = params
        # amp_err, noise_err = errs

        p.plot(f"alpha={i * 45}°", corr.beta, correlation)
        p.plot(f"fit alpha={i * 45}°", fit_beta, entangled_coincidence_phi(fit_beta, expected_alpha, amp, noise, sign=sign), "-")

        # noise_ratio = noise/amp
        # noise_ratio_err = noise_ratio * ((noise_err/noise)**2 + (amp_err/amp)**2)**(1/2)
        print(
            f"X={i * 45}°:\t"
            # f"amp: {amp:.5} ± {amp_err:.2} ({amp_err/amp:.2%}), "
            # f"noise: {noise:.5} ± {noise_err:.2} ({noise_err/noise:.2%}), "
            # f"noise_ratio: {noise_ratio:.2%} ± {noise_ratio_err:.2%} ({noise_ratio_err/noise_ratio:.2%}), "  # related to visibility
            f"chi2_red: {chi_2_red:.4}, "
        )
    p.ax.set_xticks(np.linspace(0, np.pi, 5), ["0", "π/4", "π/2", "3π/4", "π"])
    p.save(f"output/cos squared fit for {phi_name(sign)} with preset alpha.png")

    print()


def non_entangled_coincidence(beta, alpha, amp, noise):
    """The chance of two NON-entangled photons in a \\phi Bell state to both pass a pair of polarizers.

    alpha and beta are the angles of the two polarizers relative to the horizontal state.
    The probability is the same for both \\phi_- and \\phi_+.
    """
    return noise + amp * (np.cos(alpha)**2 * np.cos(beta)**2 + np.cos(alpha + np.pi/2)**2 * np.cos(beta + np.pi/2)**2)/2


def fit_non_entangled_coincidence_phi_minus(beta: NDFloat64Array, coincidence: NDFloat64Array):
    res = sp.optimize.curve_fit(
        non_entangled_coincidence,
        beta,
        coincidence,
        p0=[beta[np.argmax(coincidence)], coincidence.max() - coincidence.min(), coincidence.min()],
        sigma=coincidence ** (1 / 2),
        absolute_sigma=True,
        # bounds=([-np.inf, 0, 0], [np.inf, np.inf, np.inf]),
        full_output=True,
    )
    popt = res[0]
    pcov = res[1]
    infodict = res[2]
    print(f"non entangled coincidence chi squared: {np.sum(infodict['fvec']**2) / (len(beta)-3)}")
    return popt, pcov.diagonal() ** (1 / 2)


def check_compatibility_with_non_entanglement(corr: parse.Correlation, file_name: str):
    pass


def run_visibility(sign: Sign, path: str):
    corr = parse.Correlation.from_file(path)
    file_name, _ = os.path.splitext(os.path.basename(path))
    print(file_name)

    find_v(corr, file_name, sign)
    check_compatibility_with_entanglement(corr, file_name, sign)
    check_compatibility_with_entanglement_at_alphas(corr, file_name, sign)
    check_compatibility_with_non_entanglement(corr, file_name)
