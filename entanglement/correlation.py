"""This file includes function used in the visibility experiment"""
import os.path

import matplotlib
import numpy as np
import scipy as sp

from .typing import NDFloat64Array
from . import parse
from . import plot
from . import statistics
from .util import Sign, phi_name, phi_name_latex, rad_to_deg, mod_dist


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
    curve_params = []
    corr_counts = [corr.alpha_0, corr.alpha_45, corr.alpha_90, corr.alpha_135]
    for corr_count in corr_counts:
        v_estimate = visibility_estimate(corr_count.min(), corr_count.max())
        v_estimate_err = visibility_estimate_err(corr_count.min(), corr_count.max())
        v_estimates.append((v_estimate, v_estimate_err))

        curve_params.append(visibility_from_fit(corr.beta, corr_count))

    print(f"Visibility fit for {phi_name(sign)}")
    for i, (fit_params, fit_errs) in enumerate(curve_params):
        a, beta_c, v_fit, p = fit_params
        a_err, beta_c_err, v_fit_err, p_err = fit_errs
        v_fit_rel_err = v_fit_err / v_fit
        v_estimate, v_estimate_err = v_estimates[i]
        v_estimate_rel_err = v_estimate_err / v_estimate
        print(
            f"X={i * 45}°:\t"
            f"fit: {v_fit:.2%} ± {v_fit_err:.2%} ({v_fit_rel_err:.2%}), "
            f"estimate: {v_estimate:.2%} ± {v_estimate_err:.2%} ({v_estimate_rel_err:.1%}), "
            f"n_sigma: {statistics.nsigma(v_fit, v_fit_err, v_estimate, v_estimate_err):.2}, "
            f"periodicity: {1/p:.5g} ± {p_err/p**2:.2} ({p_err/p:.2%}) n_sigma={np.abs(0.5-p)/p_err:.2}, "
        )
    print()

    colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    p = plot.Plot(f"Visibility fit for {phi_name_latex(sign)}", r"$\beta\ [ \degree ]$", "Coincidence count")
    for i, corr_count in enumerate(corr_counts):
        p.plot_err(fr"$\alpha = {45*i} \degree$", rad_to_deg(corr.beta), corr_count, corr_count ** (1 / 2), color=colors[i])
    fit_beta = np.linspace(corr.beta.min(), corr.beta.max())
    for i, (fit_params, _fit_errs) in enumerate(curve_params):
        fit = visibility_fit(fit_beta, *fit_params)
        p.plot(fr"fit $\alpha = {45*i} \degree$", rad_to_deg(fit_beta), fit, style="-", color=colors[i])
        p.plot(None, [rad_to_deg(fit_beta)[fit.argmax()]]*2, [0, fit.max()], style="--", color="gray")

    ticks = np.linspace(0, 180, 5)
    p.ax.set_xticks(ticks, (str(int(t)) for t in ticks))
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
    p = plot.Plot(f"cos squared fit for {phi_name_latex(sign)} with free alpha", r"$\beta\ [ \degree ]$", "Coincidence count")
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
        p.plot(fr"$\alpha={alpha_deg:.3g}\degree$", corr.beta, correlation)
        p.plot(fr"fit $\alpha={alpha_deg:.3g}\degree$", fit_beta, entangled_coincidence_phi(fit_beta, alpha, amp, noise, sign=sign), "-")

        alpha_deg_err = alpha_err * 180 / np.pi
        alpha_deg_offset = mod_dist(alpha_deg, i*45, 180)
        # noise_ratio = noise/amp
        # noise_ratio_err = noise_ratio * ((noise_err/noise)**2 + (amp_err/amp)**2)**(1/2)
        print(
            f"X={i * 45}°:\t"
            f"alpha: {alpha_deg:.5}° ± {alpha_deg_err:.2}° ({alpha_deg_err/alpha_deg:.2%}), "
            # f"offset: {alpha_deg_offset:.3}°, "  # How many degrees off is the curve's phase?
            f"offset: {alpha_deg_offset:.4} ± {alpha_deg_err:.2} ({alpha_deg_err/alpha_deg_offset:.2%}), "  # How many degrees off is the curve's phase?
            f"n_sigma: {np.abs(alpha_deg_offset)/alpha_deg_err:.2g}, "
            # f"amp: {amp:.5} ± {amp_err:.2} ({amp_err/amp:.2%}), "
            # f"noise: {noise:.5} ± {noise_err:.2} ({noise_err/noise:.2%}), "
            # f"noise_ratio: {noise_ratio:.2%} ± {noise_ratio_err:.2%} ({noise_ratio_err/noise_ratio:.2%}), "  # related to visibility
            f"chi2_red: {chi_2_red:.3}, "
        )
    p.ax.set_xticks(np.linspace(0, np.pi, 5), ["0", "45", "90", "135", "180"])
    p.save(f"output/cos squared fit for {phi_name(sign)} with free alpha.png")

    print()


def check_compatibility_with_entanglement_at_alphas(corr: parse.Correlation, file_name: str, sign: Sign):
    p = plot.Plot(f"cos squared fit for {phi_name_latex(sign)} with preset alpha", r"$\beta\ [ \degree ]$", "Coincidence count")
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

        p.plot(fr"$\alpha = {i * 45}\degree$", corr.beta, correlation)
        p.plot(fr"fit $\alpha={i * 45}\degree$", fit_beta, entangled_coincidence_phi(fit_beta, expected_alpha, amp, noise, sign=sign), "-")

        # noise_ratio = noise/amp
        # noise_ratio_err = noise_ratio * ((noise_err/noise)**2 + (amp_err/amp)**2)**(1/2)
        print(
            f"X={i * 45}°:\t"
            # f"amp: {amp:.5} ± {amp_err:.2} ({amp_err/amp:.2%}), "
            # f"noise: {noise:.5} ± {noise_err:.2} ({noise_err/noise:.2%}), "
            # f"noise_ratio: {noise_ratio:.2%} ± {noise_ratio_err:.2%} ({noise_ratio_err/noise_ratio:.2%}), "  # related to visibility
            f"chi2_red: {chi_2_red:.4}, "
        )
    p.ax.set_xticks(np.linspace(0, np.pi, 5), ["0", "45", "90", "135", "180"])
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
