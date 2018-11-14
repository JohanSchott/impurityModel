#!/usr/bin/env python2

# Python plot script of RIXS spectra
import matplotlib.pylab as plt
import numpy as np
import os.path
import sys

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
x = x.reshape((len(x)/(ncols+1),ncols+1))
wLoss = x[1:,0]
wIn = x[0,1:]
rixs = x[1:,1:]

# Plot design parameter
plotCutOff = 0.001
tmp = np.copy(rixs)
mask = tmp < plotCutOff
tmp[mask] = np.nan

fig = plt.figure()
cs = plt.contourf(wIn,wLoss,tmp,cmap=plt.get_cmap('Blues'))
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel('RIXS intensity')
plt.grid(c='k', ls='-', alpha=0.3)
plt.xlabel(r'$\omega_{in}$')
plt.ylabel(r'$\omega_{loss}$')
#plt.savefig('RIXSmap.pdf')
plt.show()

