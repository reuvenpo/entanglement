import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from entanglement import visibility, parse, plot


def nsigma(x1, x1_err, x2, x2_err):
    return np.abs(x1 - x2)/(x1_err**2 + x2_err**2)**(1/2)


def run_visibility():
    corr = parse.Correlation.from_file(
        "./data/exp-1/correlation/corr_meas_01 - waveplate white part toward source - integ time 3.6s.txt"
    )
    # print(corr)

    v_estimates = []
    v_estimate_errs = []
    for x in [corr.x_0, corr.x_45, corr.x_90, corr.x_135]:
        v_estimate, v_estimate_err = visibility.find_with_estimate(x)
        v_estimates.append(v_estimate)
        v_estimate_errs.append(v_estimate_err)

    x_0_fit_params, x_0_fit_errs = visibility.find_with_least_squares(corr.y, corr.x_0)
    x_45_fit_params, x_45_fit_errs = visibility.find_with_least_squares(corr.y, corr.x_45)
    x_90_fit_params, x_90_fit_errs = visibility.find_with_least_squares(corr.y, corr.x_90)
    x_135_fit_params, x_135_fit_errs = visibility.find_with_least_squares(corr.y, corr.x_135)
    fit_params = [x_0_fit_params, x_45_fit_params, x_90_fit_params, x_135_fit_params]
    fit_errs = [x_0_fit_errs, x_45_fit_errs, x_90_fit_errs, x_135_fit_errs]

    p = plot.Plot(
        "Coincidences over 3.6s integration time", "Polarizer 2 angle [deg]", "Coincidence count"
    )
    p.plot("X=0", corr.y, corr.x_0, style="o")
    p.plot("X=45", corr.y, corr.x_45, style="o")
    p.plot("X=90", corr.y, corr.x_90, style="o")
    p.plot("X=135", corr.y, corr.x_135, style="o")
    fit_x = np.linspace(corr.y.min(), corr.y.max())
    p.plot("fit X=0", fit_x, visibility.fit(fit_x, *x_0_fit_params), style="-")
    p.plot("fit X=45", fit_x, visibility.fit(fit_x, *x_45_fit_params), style="-")
    p.plot("fit X=90", fit_x, visibility.fit(fit_x, *x_90_fit_params), style="-")
    p.plot("fit X=135", fit_x, visibility.fit(fit_x, *x_135_fit_params), style="-")
    p.ax.set_xticks(np.linspace(0, np.pi, 5), ["0", "π/4", "π/2", "3π/4", "π"])
    p.save("output/correlation.png")

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
            f"fit: {v:.1%}, fit rel err: {v_fit_rel_err:.1%} "
            f"estimate: {v_estimates[indx]:.1%}, estimate rel err:{v_estimate_rel_err:.1%}, "
            f"n_sigma: {nsigma(v, v_err, v_estimate, v_estimate_err):.2}"
        )


def main():
    run_visibility()


if __name__ == '__main__':
    main()
