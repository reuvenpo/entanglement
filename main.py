import os.path

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from entanglement.util import Sign
from entanglement.correlation import run_visibility
from entanglement.michelson import run_michelson
from entanglement.bell import run_bell
from entanglement.HOM import run_HOM
import entanglement.michelson_sim as sim

def main():
    # This one looks like phi-
    # run_visibility(Sign.Minus, "./data/exp-1/correlation/corr_meas_01 - waveplate white part toward source - integ time 2.4s.txt")
    # # This one looks like phi-
    # # run_visibility(Sign.Minus, "./data/exp-1/correlation/corr_meas_01 - waveplate white part toward source - integ time 3.6s.txt")
    # # This one looks like phi+
    # run_visibility(Sign.Plus, "./data/exp-1/correlation/corr_meas_01 - waveplate white part away from source - integ time 2.4s.txt")
    # run_michelson("data/michelson/quCNT", 9, 83, 5.73, 0.01)
    # run_bell(Sign.Plus, "data/bell/CHSHmeasurement_01.txt")
    # run_bell(Sign.Minus, "data/bell/CHSHmeasurement_02.txt")

    # Michelson Sim
    # I = sim.calc_electric_field()
    # # I = sim.intensity_average(E)
    # counts = sim.model_counts(I)
    # sim.plot(sim.d, I)

    run_HOM("data/HOM/HOM_data.txt", "output/HOM.png")

if __name__ == '__main__':
    main()
