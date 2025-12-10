import os.path

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from entanglement.visibility import run_visibility


def main():
    # This one looks like phi-
    run_visibility("./data/exp-1/correlation/corr_meas_01 - waveplate white part toward source - integ time 2.4s.txt")
    # This one looks like phi-
    run_visibility("./data/exp-1/correlation/corr_meas_01 - waveplate white part toward source - integ time 3.6s.txt")
    # This one looks like phi+
    run_visibility("./data/exp-1/correlation/corr_meas_01 - waveplate white part away from source - integ time 2.4s.txt")


if __name__ == '__main__':
    main()
