from dataclasses import dataclass

import numpy as np

from .typing import NDFloat64Array, NDStrArray


@dataclass
class Correlation:
    """Parses `corr_meas` files from quCR"""
    y: NDFloat64Array
    """Polarizer Y angle [deg]"""
    x_0: NDFloat64Array
    """Coincidence counts for polarizer X at 0°"""
    x_45: NDFloat64Array
    """Coincidence counts for polarizer X at 45°"""
    x_90: NDFloat64Array
    """Coincidence counts for polarizer X at 90°"""
    x_135: NDFloat64Array
    """Coincidence counts for polarizer X at 135°"""

    @classmethod
    def from_file(cls, path: str) -> "Correlation":
        df = np.genfromtxt(path, skip_header=4, dtype=np.float64).transpose()
        y = df[0]
        x_0 = df[1]
        x_45 = df[2]
        x_90 = df[3]
        x_135 = df[4]

        return cls(y, x_0, x_45, x_90, x_135)


# TODO: add parsing of the information in the comments in the file header
@dataclass
class Visibility:
    """Parses `Visibility` files from quCR"""
    measurement: NDStrArray
    """The measurement operator used (Dirac notation)"""
    single_1: NDFloat64Array
    """Single photon counts in detector 1"""
    single_2: NDFloat64Array
    """Single photon counts in detector 2"""
    coincidences: NDFloat64Array
    """Coincidence counts between the detectors"""
    randoms: NDFloat64Array
    """???"""

    @classmethod
    def from_file(cls, path: str) -> "Visibility":
        df = np.genfromtxt(path, skip_header=4, dtype=np.str_).transpose()
        measurement = df[0]

        df = np.genfromtxt(path, skip_header=4, dtype=np.float64).transpose()
        single_1 = df[1]
        single_2 = df[2]
        coincidences = df[3]
        randoms = df[4]

        return cls(measurement, single_1, single_2, coincidences, randoms)
