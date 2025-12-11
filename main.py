import os.path

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from entanglement.correlation import run_visibility, Sign
from entanglement.michelson import run_michelson


def main():
    # This one looks like phi-
    run_visibility(Sign.Minus, "./data/exp-1/correlation/corr_meas_01 - waveplate white part toward source - integ time 2.4s.txt")
    # This one looks like phi-
    # run_visibility(Sign.Minus, "./data/exp-1/correlation/corr_meas_01 - waveplate white part toward source - integ time 3.6s.txt")
    # This one looks like phi+
    run_visibility(Sign.Plus, "./data/exp-1/correlation/corr_meas_01 - waveplate white part away from source - integ time 2.4s.txt")
    run_michelson("data/michelson/quCNT", 9, 83, 5.73, 0.01)


if __name__ == '__main__':
    main()
