#!/usr/bin/env python3

# Python plot script of RIXS spectra
import matplotlib.pylab as plt
import numpy as np
import os.path
import sys


def main():
    # Figure out which files to read from.
    if len(sys.argv) == 1:
        filename = 'RIXS.bin'
        if not os.path.isfile(filename):
            sys.exit('Data file does not exist: ' + filename)
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        if not os.path.isfile(filename):
            sys.exit('Data file does not exist: ' + filename)
    else:
        sys.exit('Number of input arguments wrong')
    
    x = np.fromfile(filename,dtype=np.float32)
    ncols = int(x[0])
    x = x.reshape((int(len(x)/(ncols+1)),ncols+1))
    wLoss = x[1:,0]
    wIn = x[0,1:]
    rixs = x[1:,1:]
    
    # Plot design parameter
    plotCutOff = 0.001
    tmp = np.copy(rixs)
    mask = tmp < plotCutOff
    tmp[mask] = np.nan
    
    fig = plt.figure()
    # Choose a nice colormap, e.g. 'viridis' or 'Blues'
    cs = plt.contourf(wIn, wLoss, np.log10(tmp), cmap=plt.get_cmap('viridis'))
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('log RIXS intensity')
    #plt.grid(c='k', ls='-', alpha=0.3)
    plt.xlabel(r'$\omega_{in}$')
    plt.ylabel(r'$\omega_{loss}$')
    plt.tight_layout()
    #plt.savefig('RIXSmap.pdf')
    plt.show()


if __name__ == '__main__':
    main()

