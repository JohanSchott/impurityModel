#!/usr/bin/env python3

# Python plot script of RIXS spectra
import matplotlib.pylab as plt
import numpy as np
import os.path
import sys
from math import pi

# Figure out which files to read from.
if len(sys.argv) == 1:
    filename = 'spectra.h5'
    if os.path.isfile(filename):
        readH5 = True
    else:
        readH5 = False
elif len(sys.argv) == 2:
    filename = sys.argv[1]
    if not os.path.isfile(filename):
        sys.exit('Data file does not exist: ' + filename)
    if filename[-3:] == '.h5':
        sys.exit('A filename of format .h5 expected as input argument')
    readH5 = True
else:
    sys.exit('Number of input arguments wrong')

# Read input data
if readH5:
    print('Read data from file: ',filename)
    import h5py
    h5f = h5py.File(filename,'r')
    print(list(h5f.keys()))
    # Load energy meshes
    if 'w' in h5f:
        w = np.array(h5f['w'])
    if 'wIn' in h5f:
        wIn = np.array(h5f['wIn'])
    if 'wLoss' in h5f:
        wLoss = np.array(h5f['wLoss'])
    # Load momentum vector information
    if 'qsNIXS' in h5f:
        qs = np.array(h5f['qsNIXS'])
    # Load radial mesh information
    if 'r' in h5f and 'RiNIXS' in h5f and 'RjNIXS' in h5f:
        r = np.array(h5f['r'])
        Ri = np.array(h5f['RiNIXS'])
        Rj = np.array(h5f['RjNIXS'])
    # Load thermally averaged spectra
    if 'PSthermal' in h5f:
        ps = np.array(h5f['PSthermal'])
        print(np.shape(ps))
    if 'XPSthermal' in h5f:
        xps = np.array(h5f['XPSthermal'])
        print(np.shape(xps))
    if 'XASthermal' in h5f:
        xas = np.array(h5f['XASthermal'])
        print(np.shape(xas))
    if 'RIXSthermal' in h5f:
        rixs = np.array(h5f['RIXSthermal'])
        print(np.shape(rixs))
    if 'NIXSthermal' in h5f:
        nixs = np.array(h5f['NIXSthermal'])
        print(np.shape(nixs))
    h5f.close()
else:
    print('Read data from several .npz files.')
    # Load energy meshes, and momentum and radial information
    if os.path.isfile('spectraInfo.npz'):
        data = np.load('spectraInfo.npz')
        w = data['w']
        wIn = data['wIn']
        wLoss = data['wLoss']
        qs = data['qsNIXS']
        r = data['r']
        Ri = data['RiNIXS']
        Rj = data['RjNIXS']
    # Load thermally averaged spectra
    if os.path.isfile('spectraPS.npz'):
        ps = np.load('spectraPS.npz')['PSthermal']
        print(np.shape(ps))
    if os.path.isfile('spectraXPS.npz'):
        xps = np.load('spectraXPS.npz')['XPSthermal']
        print(np.shape(xps))
    if os.path.isfile('spectraXAS.npz'):
        xas = np.load('spectraXAS.npz')['XASthermal']
        print(np.shape(xas))
    if os.path.isfile('spectraRIXS.npz'):
        rixs = np.load('spectraRIXS.npz')['RIXSthermal']
        print(np.shape(rixs))
    if os.path.isfile('spectraNIXS.npz'):
        nixs = np.load('spectraNIXS.npz')['NIXSthermal']
        print(np.shape(nixs))

print('Plot spectra...')

if 'ps' in locals():
    print('Photo-emission spectroscopy (PS) spectrum')
    fig = plt.figure()
    # Sum over spin-orbitals
    plt.plot(w,np.sum(ps,axis=0),'-k',label='photo-emission')
    plt.legend()
    plt.xlabel(r'$\omega$')
    plt.ylabel('Intensity')
    #plt.xlim([-8,18])
    #plt.ylim([0,0.25])
    plt.tight_layout()
    plt.show()

if 'xps' in locals():
    print('X-ray photo-emission spectroscopy (XPS) spectrum')
    fig = plt.figure()
    # Sum over spin-orbitals
    plt.plot(w,np.sum(xps,axis=0),'-k',label='XPS')
    plt.legend()
    plt.xlabel(r'$\omega$')
    plt.ylabel('Intensity')
    #plt.xlim([-8,18])
    #plt.ylim([0,0.25])
    plt.tight_layout()
    plt.show()

if 'xas' in locals():
    print('XAS spectrum')
    fig = plt.figure()
    # Sum over polarizations
    plt.plot(w,np.sum(xas,axis=0),'-k',label='XAS')
    if 'rixs' in locals():
        scaleFY = 1./(pi*np.shape(rixs)[0])
        print('Fluorescence yield spectrum')
        plt.plot(wIn,(wLoss[1]-wLoss[0])*np.sum(rixs,axis=(0,1,3))*scaleFY,
                 '-r',label='FY')
        mask = wLoss < 0.2
        y = np.sum(rixs[:,:,:,mask],axis=(0,1,3))
        plt.plot(wIn,(wLoss[1]-wLoss[0])*y*scaleFY,'-b',label='quasi-elastic FY')
    plt.legend()
    plt.xlabel(r'$\omega_{in}$')
    plt.ylabel('Intensity')
    #plt.xlim([-8,18])
    #plt.ylim([0,0.25])
    plt.tight_layout()
    plt.show()

if 'nixs' in locals():
    print('NIXS spectrum')
    fig = plt.figure()
    if 'qs' in locals():
        labels = ['|q|={:3.1f}'.format(np.linalg.norm(q)) + r' A$^{-1}$' for q in qs]
    else:
        labels = [str(i) for i in range(len(nixs))]
    for i in range(len(nixs)):
        plt.plot(wLoss,nixs[i,:],label=labels[i])
    plt.legend()
    plt.xlabel(r'$\omega_{loss}$')
    plt.ylabel('Intensity')
    plt.tight_layout()
    plt.show()

if 'rixs' in locals():
    print('Energy loss spectra')
    fig,axes = plt.subplots(nrows=2,sharex=True)
    # L3-edge energies.
    # Adjust these energies to the current material.
    es = np.arange(-7,2,1)
    plotOffset = 1.5
    print('Chosen L3 energies: ',es)
    print('Chosen plotOffset: ',plotOffset)
    for n,e in enumerate(es[-1::-1]):
        i = np.argmin(np.abs(wIn-e))
        axes[0].plot(wLoss,plotOffset*(len(es)-1-n) + np.sum(rixs,axis=(0,1))[i,:],
                     label=r'$\omega_{in}$' + '={:3.1f}'.format(e))
    # L2-edge energies.
    # Adjust these energies to the current material.
    es = np.arange(11,16,1)
    plotOffset = 1
    print('Chosen L2 energies: ',es)
    print('Chosen plotOffset: ',plotOffset)
    for n,e in enumerate(es[-1::-1]):
        i = np.argmin(np.abs(wIn-e))
        axes[1].plot(wLoss,plotOffset*(len(es)-1-n) + np.sum(rixs,axis=(0,1))[i,:],
                     label=r'$\omega_{in}$' + '={:3.1f}'.format(e))
    axes[1].set_xlabel(r'$E_{loss}$  (eV)')
    axes[0].set_title(r'$L_3$')
    axes[1].set_title(r'$L_2$')
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.show()


if 'rixs' in locals():
    print('RIXS map')
    print('Plot log10 of RIXS intensity for better visibility.')
    print('In-coming photon mesh resolution: {:5.3f} eV'.format(wIn[1]-wIn[0]))
    print('Energy loss mesh resolution: {:5.3f} eV'.format(wLoss[1]-wLoss[0]))
    plotCutOff = 0.001

    # Sum over in and out-going polarizations
    fig = plt.figure()
    tmp = np.sum(rixs,axis=(0,1)).T
    mask = tmp < plotCutOff
    tmp[mask] = np.nan
    # Choose a nice colormap, e.g. 'viridis' or 'Blues'
    cs = plt.contourf(wIn,wLoss,np.log10(tmp),cmap=plt.get_cmap('viridis'))
    #cs2 = plt.contour(cs, levels=cs.levels[::2], cmap=plt.get_cmap('viridis'))
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('log RIXS intensity')
    # Add the contour line levels to the colorbar
    #cbar.add_lines(cs2)
    #for e in wIn:
    #    plt.plot([e,e],[wLoss[0],wLoss[-1]],'-k',lw=0.5)
    plt.grid(c='k', ls='-', alpha=0.3)
    plt.xlabel(r'$\omega_{in}$   (eV)')
    plt.ylabel(r'$\omega_{loss}$   (eV)')
    plt.tight_layout()
    plt.show()

    # All polarization combinations In:x,y,z , Out:x,y,z
    fig,axes = plt.subplots(nrows=np.shape(rixs)[0],ncols=np.shape(rixs)[1],
                            sharex=True,sharey=True)
    if np.shape(rixs)[:2] == (1,1):
        tmp = np.copy(rixs[0,0,:,:].T)
        mask = tmp < plotCutOff
        tmp[mask] = np.nan
        # Choose a nice colormap, e.g. 'viridis' or 'Blues'
        cs = axes.contourf(wIn,wLoss,np.log10(tmp),cmap=plt.get_cmap('viridis'))
        #cs2 = plt.contour(cs, levels=cs.levels[::2], cmap=plt.get_cmap('viridis'))
        # Make a colorbar for the ContourSet returned by the contourf call.
        cbar = fig.colorbar(cs, ax=axes)
        cbar.ax.set_ylabel('log RIXS intensity')
        # Add the contour line levels to the colorbar
        #cbar.add_lines(cs2)
        #for e in wIn:
        #    plt.plot([e,e],[wLoss[0],wLoss[-1]],'-k',lw=0.5)
        plt.grid(c='k', ls='-', alpha=0.3)
        axes.set_xlabel(r'$\omega_{in}$   (eV)')
        axes.set_ylabel(r'$\omega_{loss}$   (eV)')
    else:
        for i in range(np.shape(axes)[0]):
            for j in range(np.shape(axes)[1]):
                tmp = np.copy(rixs[i,j,:,:].T)
                mask = tmp < plotCutOff
                tmp[mask] = np.nan
                # Choose a nice colormap, e.g. 'viridis' or 'Blues'
                cs = axes[i,j].contourf(wIn,wLoss,np.log10(tmp),cmap=plt.get_cmap('viridis'))
                #cs2 = plt.contour(cs, levels=cs.levels[::2], cmap=plt.get_cmap('viridis'))
                # Make a colorbar for the ContourSet returned by the contourf call.
                cbar = fig.colorbar(cs, ax=axes[i,j])
                cbar.ax.set_ylabel('log RIXS intensity')
                # Add the contour line levels to the colorbar
                #cbar.add_lines(cs2)
                #for e in wIn:
                #    plt.plot([e,e],[wLoss[0],wLoss[-1]],'-k',lw=0.5)
                #plt.grid(c='k', ls='-', alpha=0.3)
                axes[i,j].set_title('(' + str(i) + str(j) + ')')
        for ax in axes[2,:]:
            ax.set_xlabel(r'$\omega_{in}$   (eV)')
        for ax in axes[:,0]:
            ax.set_ylabel(r'$\omega_{loss}$   (eV)')
    plt.tight_layout()
    plt.show()

