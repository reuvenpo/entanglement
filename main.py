import os.path

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from entanglement import visibility, parse, plot


def nsigma(x1, x1_err, x2, x2_err):
    return np.abs(x1 - x2)/(x1_err**2 + x2_err**2)**(1/2)


def run_visibility(path):
    corr = parse.Correlation.from_file(path)
    file_name, _ = os.path.splitext(os.path.basename(path))
    print(file_name)

    v_estimates = []
    v_estimate_errs = []
    for x in [corr.alpha_0, corr.alpha_45, corr.alpha_90, corr.alpha_135]:
        v_estimate, v_estimate_err = visibility.find_with_estimate(x)
        v_estimates.append(v_estimate)
        v_estimate_errs.append(v_estimate_err)

    alpha_0_fit_params, alpha_0_fit_errs = visibility.find_with_least_squares(corr.beta, corr.alpha_0)
    alpha_45_fit_params, alpha_45_fit_errs = visibility.find_with_least_squares(corr.beta, corr.alpha_45)
    alpha_90_fit_params, alpha_90_fit_errs = visibility.find_with_least_squares(corr.beta, corr.alpha_90)
    alpha_135_fit_params, alpha_135_fit_errs = visibility.find_with_least_squares(corr.beta, corr.alpha_135)
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
    p.plot("fit alpha=0", fit_beta, visibility.fit(fit_beta, *alpha_0_fit_params), style="-")
    p.plot("fit alpha=45", fit_beta, visibility.fit(fit_beta, *alpha_45_fit_params), style="-")
    p.plot("fit alpha=90", fit_beta, visibility.fit(fit_beta, *alpha_90_fit_params), style="-")
    p.plot("fit alpha=135", fit_beta, visibility.fit(fit_beta, *alpha_135_fit_params), style="-")
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
            f"n_sigma: {nsigma(v, v_err, v_estimate, v_estimate_err):.2}"
        )
    print()


def main():
    # This one looks like phi-
    run_visibility("./data/exp-1/correlation/corr_meas_01 - waveplate white part toward source - integ time 2.4s.txt")
    # This one looks like phi-
    run_visibility("./data/exp-1/correlation/corr_meas_01 - waveplate white part toward source - integ time 3.6s.txt")
    # This one looks like phi+
    run_visibility("./data/exp-1/correlation/corr_meas_01 - waveplate white part away from source - integ time 2.4s.txt")


if __name__ == '__main__':
    main()
