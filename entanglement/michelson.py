import os.path

import numpy as np

from . import parse
from . import plot


def run_michelson(folder: str, i_start: int, i_end: int, d0: float, dx: float):
    x = []
    y = []
    y_err = []
    for i in range(i_start, i_end):
        file_name = os.path.join(folder, f"quCNT_{i:02,}.txt")
        counts = parse.Counts.from_file(file_name)
        count = np.average(counts.count)
        count_err = np.std(counts.count) #/ counts.count.size ** (1/2)

        x.append(d0 + (i - i_start)*dx)
        y.append(count)
        y_err.append(count_err)

    p = plot.Plot("Interference Pattern in Michelson Interferometer", "wedge position [mm]", "single photon counts")
    p.plot_err("counts", x, y, y_err, "o", "-")
    p.save("output/michelson interference.png")
