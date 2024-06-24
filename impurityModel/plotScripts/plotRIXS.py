"""
Python plot script of RIXS spectra
"""

import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Figure out which files to read from.
    if len(sys.argv) == 1:
        filename = "RIXS.bin"
        if not os.path.isfile(filename):
            sys.exit("Data file does not exist: " + filename)
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        if not os.path.isfile(filename):
            sys.exit("Data file does not exist: " + filename)
    else:
        sys.exit("Number of input arguments wrong")

    x = np.fromfile(filename, dtype=np.float32)
    ncols = int(x[0])
    x = x.reshape((int(len(x) / (ncols + 1)), ncols + 1))
    wLoss = x[1:, 0]
    wIn = x[0, 1:]
    rixs = x[1:, 1:]

    # Plot design parameter
    plotCutOff = 1e-6
    tmp = np.copy(rixs)
    mask = tmp < plotCutOff
    tmp[mask] = np.nan

    dx = wIn[1] - wIn[0]
    dy = wLoss[1] - wLoss[0]
    left = wIn[0] - dx / 2
    right = wIn[-1] + dx / 2
    bottom = wLoss[0] - dy / 2
    top = wLoss[-1] + dy / 2

    plt.figure()
    cs = plt.imshow(tmp, origin="lower", extent=(left, right, bottom, top), aspect="auto", norm="log")
    cbar = plt.colorbar(cs)
    cbar.ax.set_ylabel("RIXS intensity")
    plt.xlabel(r"$\omega_{in}$   (eV)")
    plt.ylabel(r"$\omega_{loss}$   (eV)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
