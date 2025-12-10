import numpy as np


def nsigma(x1, x1_err, x2, x2_err):
    return np.abs(x1 - x2)/(x1_err**2 + x2_err**2)**(1/2)
