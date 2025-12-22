import numpy as np
import scipy.optimize as opt

from entanglement.michelson_sim import bandwidth
from entanglement.plot import Plot
from entanglement.parse import Linear_Scan

wavelength = 810e-9
k = 1 / wavelength


def run_HOM(input_path: str, output_path: str):
    HOM_data = Linear_Scan.from_file(input_path)

    # Display data
    p = Plot("HOM 01 channel coincidence", "scanner [mm]", "coincidence counts")
    p.plot("HOM data", HOM_data.positions, HOM_data.ch01)

    w0 = 3e8 / wavelength
    bandwidth = w0*0.1

    popt = opt.curve_fit(
        HOM_fit,
        HOM_data.positions,
        HOM_data.ch01,
        [1000, 1e3, 0.5, 0.5, 0.0284375],
    )
    parameters = popt[0]
    Count, Delta_omega, R, T, correction = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
    print(Count)
    print(Delta_omega)
    print(R)
    print(T)
    print(R / T)
    print(correction)
    p.plot_err_with_fit(
        "HOM Fit",
        HOM_data.positions,
        HOM_data.ch01,
        sigma=np.zeros(HOM_data.positions.size),
        fit=lambda d: HOM_fit(d, Count, Delta_omega, R, T, correction),
    )
    p.save(output_path)


def HOM_fit(d, C, Delta_omega, R, T, correction):
    """Fitting function, converting d into the path phase difference"""
    exp = np.exp(-(Delta_omega * (d-correction)) ** 2)
    return C * (R ** 2 + T ** 2) * (1 - (2 * R * T) * exp / (R ** 2 + T ** 2))
