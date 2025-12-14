from dataclasses import dataclass

import numpy as np

from .typing import NDFloat64Array, NDStrArray


@dataclass
class Correlation:
    """Parses `corr_meas` files from quCR"""
    beta: NDFloat64Array
    """Polarizer Y angle [rad]"""
    alpha_0: NDFloat64Array
    """Coincidence counts for polarizer X at 0°"""
    alpha_45: NDFloat64Array
    """Coincidence counts for polarizer X at 45°"""
    alpha_90: NDFloat64Array
    """Coincidence counts for polarizer X at 90°"""
    alpha_135: NDFloat64Array
    """Coincidence counts for polarizer X at 135°"""

    @classmethod
    def from_file(cls, path: str) -> "Correlation":
        df = np.genfromtxt(path, skip_header=4, dtype=np.float64).transpose()
        y = df[0] * np.pi / 180
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


@dataclass
class Counts:
    """Parses `quCNT` files from quCR"""
    time: NDFloat64Array
    """Time in seconds"""
    count: NDFloat64Array
    """Single count rate"""

    @classmethod
    def from_file(cls, path: str) -> "Counts":
        df = np.genfromtxt(path, dtype=np.float64).transpose()
        time = df[0]
        counts = df[1]

        return cls(time, counts)


@dataclass
class CHSHCount:
    rate_1: float
    rate_2: float
    coincidences: int
    corrected: int

    # This property exists so we can easily switch between `coincidences` and `corrected` in testing
    @property
    def c(self):
        return self.corrected


CHSHCounts = dict[tuple[float, float], CHSHCount]


@dataclass
class CHSH:
    counts: CHSHCounts

    @classmethod
    def from_file(cls, path: str):
        # collect all the data from the file:
        data: list[dict] = []
        with open(path, "r") as file:
            for line in file:
                if line.startswith("="):
                    continue
                comment_index = line.find("#")
                if comment_index != -1:
                    line = line[:comment_index]
                line = line.strip()
                if line == "":
                    continue
                data_line = {}
                fields = line.split(",")
                for field in fields:
                    key, value = field.split("=")
                    key = key.strip()
                    value = value.strip()
                    data_line[key] = value
                data.append(data_line)

        # organize the data in a convenient way
        counts = {
            (
                float(data_line["X"][:-len(" deg")]),
                float(data_line["Y"][:-len(" deg")]),
            ): CHSHCount(
                float(data_line["rate1"]),
                float(data_line["rate2"]),
                int(data_line["coincidences"]),
                int(data_line["corrected"]),
            )
            for data_line in data
        }
        return cls(counts)
