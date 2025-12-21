import numpy as np
from numpy.typing import ndarray, float64, Any, dtype
from plot import Plot

# Constants

C = 3e8  # speed of light

# Spatial Coordinates - Wedge width
d = np.linespace(4e-4, 8e-4, 5000)

# Spectrum Parameters
# 810 nm
LAMBDA_0 = 810e-9
bandwidth = 5e-9  #
n_lambda = 11  # Number of waves, keep odd values to get symmetrical dist around LAMBDA_0

lambdas = np.linespace(
    LAMBDA_0 - 3 * bandwidth,
    LAMBDA_0 + 3 * bandwidth,
    n_lambda
)

# Add weights to the different lambdas,
# assume Gaussian dist around central wavelength decay
# can also use Lorentzian for spectroscopy
# (keep as free parameters for scipy fit)

spectral_weights = np.exp(
    -0.5 * ((lambdas - LAMBDA_0) / bandwidth) ** 2
)
# Normalize
spectral_weights /= spectral_weights.sum()

# TO DO - ADD TIME COHERENCE
t = np.array([0.0])

# Beams and Polarization

E_in = np.array([1.0, 0.0])


# Using polarization matrix instead of Malus's Law to pass field as Malus's law discusses the intensity
def polarizer_matrix(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([[c ** 2, c * s],
                     [c * s, s ** 2]])


# All polarizers are the same as the incoming wave - no polarizers
# To add - dynamic polarizers
P1 = polarizer_matrix(0)
P2 = polarizer_matrix(0)
P_out = polarizer_matrix(0)


# Phase Delay
def phi(d, k=12345.67901235):
    """
    Phase delay due to wedge/refraction.
    Assumed identical for all wavelengths.
    Using n~1.51
    Set k for 810nm -> k=1/810nm
    """
    Delta = d / 56.1
    phase = 2 * Delta * k
    return phase


# Electric Fields
# Sum the electric fields

def electric_field_with_time_coherence():
    E_total_time = np.zeros((t.size, 2, d.size))

    for ti, time in enumerate(t):
        E_total = calc_electric_field(time)

        E_total_time[ti] = E_total
    return E_total_time


def calc_electric_field(time: tuple[int, Any]) -> ndarray[tuple[int, int], dtype[float64]]:
    E_total = np.zeros((2, d.size))

    for lam, w in zip(lambdas, spectral_weights):
        k = 2 * np.pi / lam
        omega = 2 * np.pi * C / lam

        phase = k * d - omega * time
        added_phase = phi(d)

        # Note - @ symbol is matrix multiplication operator vs * which is elementwise

        # Arm 1 - no extra phase
        E1 = P1 @ E_in
        E1 = E1[:, None] * w * np.cos(phase)
        # Arm 2 - extra phase
        E2 = P2 @ E_in
        E2 = E2[:, None] * w * np.cos(phase + added_phase)
        # Arm 3 - out
        E_out = P_out @ (E1 + E2)

        E_total += E_out
    return E_total


def intensity_average(E_total_time: ndarray):
    """
    Expects E_total_time to be a 3d numpy array.
    E_total_time = [time,2,d]
    where 2 is for polarization dimensions and d is wedge width array
    """
    return np.mean(
        np.sum(E_total_time ** 2, axis=1),
        axis=0
    )


def plot(d, E_tot):
    plt = Plot(
        f"Michelson-Sim - {n_lambda} waves around {LAMBDA_0}",
        "wedge width [m]",
        "Intensity average"
    )
    plt.plot("", d, E_tot)
    plt.save(f"../output/Michelson-Sim - {n_lambda} waves around {LAMBDA_0}.png")
