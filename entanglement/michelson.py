import os.path

import numpy as np
import scipy.fft

from . import parse
from . import plot


WEDGE_ALPHA = np.arctan2(0.44, 25)


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

    N = len(x)
    x = np.array(x)
    p = plot.Plot("Interference Pattern in Michelson Interferometer", "wedge position [mm]", "single photon counts")
    p.plot_err("counts", x, y, y_err, "o", "-")
    p.save("output/michelson interference.png")

    fft = scipy.fft.fft(y)
    fft_freqs = scipy.fftpack.fftfreq(len(y), np.abs(dx))
    frac = 4
    fft[N//2-N//frac:N//2+N//frac] = 0
    # print(fft)
    # fft[0] = 0

    plot_fft_freqs = fft_freqs#[:N//2]
    plot_fft = 2/len(y) * fft#[:N//2]
    # print(np.abs(plot_fft))
    # print(plot_fft_freqs)

    p = plot.Plot("fourier transform", "frequencies [1/mm]", "factor")
    p.plot("imag", plot_fft_freqs, plot_fft.imag, "-")
    p.plot("real", plot_fft_freqs, plot_fft.real, "-")
    p.plot("abs", plot_fft_freqs, np.abs(plot_fft), "-")
    p.save("output/michelson frequencies.png")

    N = 1000
    rec_x = np.linspace(x[0], x[-1], N, dtype=np.float64)
    # rec_y = np.zeros_like(rec_x)
    # for i in range(len(fft_freqs)):
    #     rec_y += fft[i]
    # rec_y = fft * np.sin(np.abs(fft) * rec_x + np.angle(fft))
    # M = len(fft)
    # rec_y = np.sum([fft[i] * (np.cos(rec_x*i*2*np.pi/M) + 1j * np.sin(rec_x*i*2*np.pi/M)) for i in range(len(fft))], axis=0)
    rec_y = scipy.fft.ifft(fft, len(rec_x))
    p = plot.Plot("pattern recreation", "wedge position [mm]", "single photon counts")
    p.plot("counts", rec_x, np.abs(rec_y), "-")
    # p.plot_err("counts", x, y, y_err, "o")
    p.save("output/michelson interference recreation.png")

