import numpy as np
# from numpy.typing import ndarray, float64, Any, dtype
from .plot import Plot

# Constants
C = 3e8  # speed of light

# Field amplitudes consistent with V
A1 = 0.5
A2 = 0.5

# Spatial Coordinates - Wedge width
d = np.linspace(4e-4, 6e-4, 500)

# Spectrum Parameters
# 810 nm
LAMBDA_0 = 810e-9
# LAMBDA_1 = 610e-9
bandwidth = 5e-9  #
n_lambda = 1  # Number of waves, keep odd values to get symmetrical dist around LAMBDA_0

lambdas0 = np.linspace(
    LAMBDA_0 - 3 * bandwidth,
    LAMBDA_0 + 3 * bandwidth,
    n_lambda
)

# lambdas1 = np.linspace(
#     LAMBDA_1 - 3 * bandwidth,
#     LAMBDA_1 + 3 * bandwidth,
#     n_lambda
# )

# lambdas = np.append(lambdas0, lambdas1)
lambdas = lambdas0

# Add weights to the different lambdas,
# assume Gaussian dist around central wavelength decay
# can also use Lorentzian for spectroscopy
# (keep as free parameters for scipy fit)

spectral_weights_0 = np.exp(
    -0.5 * ((lambdas0 - LAMBDA_0) / bandwidth) ** 2
)

# spectral_weights_1 = np.exp(
#     -0.5 * ((lambdas1 - LAMBDA_1) / bandwidth) ** 2
# )
#
# spectral_weights = np.append(spectral_weights_0, spectral_weights_1)

spectral_weights = spectral_weights_0

# Normalize
spectral_weights /= spectral_weights.sum()

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


def calc_electric_field():
    E_total = np.zeros((2, d.size))
    I = np.zeros((d.size))
    for lam, w in zip(lambdas, spectral_weights):
        k = 2 * np.pi / lam

        # Note - @ symbol is matrix multiplication operator vs * which is elementwise

        # Arm 1 - no extra phase
        E1 = A1 * (P1 @ E_in)
        E1 = E1[:, None]
        # Arm 2 - extra phase
        E2 = A2 * (P2 @ E_in)
        E2 = E2[:, None] * np.cos(phi(d, k))
        # Arm 3 - out
        E_out = P_out @ (E1 + E2)

        I_lambda = np.sum(E_out ** 2, axis=0)

        I += w*I_lambda
    return I


def intensity_average(E_total_time):
    """
    Expects E_total_time to be a 3d numpy array.
    E_total_time = [2,d]
    where 2 is for polarization dimensions and d is wedge width array
    """
    return np.sum(E_total_time ** 2, axis=0)


# Using intensity as probability indicator
# Detector model
def model_counts(intensity):
    T = 1  # Integration time [s]
    A = 1e4  # Amplitude base (laser-crystal efficiency)
    photon_number = A * T * intensity
    counts = np.random.poisson(photon_number)
    return counts


def plot(d, I):
    plt = Plot(
        f"Michelson-Sim - {n_lambda} waves around {LAMBDA_0}",
        "wedge width [m]",
        "Counts - Poisson distribution of (int time * Intensity)",
    )
    plt.plot("", d, I)
    plt.save(f"./output/Michelson-Sim - {n_lambda} waves around {LAMBDA_0}.png")
