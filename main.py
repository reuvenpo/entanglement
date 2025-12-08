import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from entanglement import visibility, parse, plot


def run_visibility():
    corr = parse.Correlation.from_file(
        "./data/exp-1/correlation/corr_meas_01 - waveplate white part toward source - integ time 3.6s.txt"
    )
    # print(corr)

    x_estimates = []
    x_estimate_errs = []
    for x in [corr.x_0, corr.x_45, corr.x_90, corr.x_135]:
        x_estimate, x_estimate_err = visibility.find_with_estimate(x)
        x_estimates.append(x_estimate)
        x_estimate_errs.append(x_estimate_err)

    x_0_fit_params = visibility.find_with_least_squares(corr.y, corr.x_0)
    x_45_fit_params = visibility.find_with_least_squares(corr.y, corr.x_45)
    x_90_fit_params = visibility.find_with_least_squares(corr.y, corr.x_90)
    x_135_fit_params = visibility.find_with_least_squares(corr.y, corr.x_135)

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
    p.save("output/correlation.png")

    print("The visibility at different X angles")
    for indx, params in enumerate([x_0_fit_params, x_45_fit_params, x_90_fit_params, x_135_fit_params]):
        a, beta_c, v, p = params
        x_estimate_rel_err = x_estimate_errs[indx]/x_estimates[indx]
        print(
            f"X={indx*45}°:\t"
            f"fit: {v:.1%},\t"
            f"estimate: {x_estimates[indx]:.1%},\testimate err:{x_estimate_rel_err:.1%},\t"
            f"n_sigma: {np.abs(v-x_estimates[indx])/x_estimate_errs[indx]:.2}"
        )


def main():
    run_visibility()


if __name__ == '__main__':
    main()
