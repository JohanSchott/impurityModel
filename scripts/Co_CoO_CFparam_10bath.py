#!/usr/bin/env python3

# Script for solving many-body impurity problem.

import numpy as np
import scipy.sparse.linalg
from collections import OrderedDict
import sys,os
from mpi4py import MPI
import time

from impurityModel import spectra
from impurityModel import finite
from impurityModel.finite import c2i
from impurityModel.average import k_B, thermal_average


def main():

    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.rank
    ranks = comm.size

    if rank == 0: t0 = time.time()

    # -----------------------
    # System specific information
    l1, l2 = 1, 2 # Angular momentum
    # Number of bath states.
    nBaths = OrderedDict()
    nBaths[l1] = 0
    nBaths[l2] = 10
    # Number of valence bath states.
    valBaths = OrderedDict()
    valBaths[l1] = 0
    valBaths[l2] = 10
    # -----------------------
    # Basis occupation information.
    # Angular momentum : initial impurity occupation
    n0imp = OrderedDict()
    n0imp[l1] = 6 # 0 = empty, 2*(2*l1+1) = Full occupation
    n0imp[l2] = 7 # 8 for Ni+2
    # Angular momentum : max devation from initial impurity occupation
    dnTol = OrderedDict()
    dnTol[l1] = 0
    dnTol[l2] = 2
    # Angular momentum : max number of electrons to leave
    # valence bath orbitals
    dnValBaths = OrderedDict()
    dnValBaths[l1] = 0
    dnValBaths[l2] = 2
    # Angular momentum : max number of electrons to enter
    # conduction bath orbitals
    dnConBaths = OrderedDict()
    dnConBaths[l1] = 0
    dnConBaths[l2] = 0
    # -----------------------
    # Hamiltonian parameters
    # Slater-Condon parameters
    Fdd = [7.0, 0, 9.6, 0, 6.4]
    Fpp = [0, 0, 0]
    Fpd = [8.0, 0, 6.4]
    Gpd = [0, 4.6, 0, 2.6]
    # SOC values
    xi_2p = 9.859
    xi_3d = 0.079
    # Double counting parameter
    chargeTransferCorrection = 1.5
    # Onsite 3d energy parameters
    eImp3d = -0.802
    deltaO = 0.612
    # Magnetic field
    hField = [0, 0, 0.0001] # 0.120*np.array([1,1,2])/np.sqrt(6) # [0,0,0.00001]
    # Bath energies and hoppings for 3d orbitals
    eValEg = -4.5
    eValT2g = -6.5
    eConEg = 3
    eConT2g = 2
    vValEg = 1.919
    vValT2g = 1.412
    vConEg = 0.6
    vConT2g = 0.4
    # -----------------------
    # Maximum number of eigenstates to consider
    nPsiMax = 13
    # -----------------------
    # Printing parameters
    nPrintSlaterWeights = 3
    tolPrintOccupation = 0.5
    # To print spectra to HDF5 format (or to .npz format).
    printH5 = True  # True or False
    # -----------------------
    # Spectra parameters
    # Temperature (Kelvin)
    T = 300
    # How much above lowest eigenenergy to consider
    energy_cut = 10*k_B*T
    # energy-mesh
    w = np.linspace(-25,25,3000)
    # Smearing, half with half maximum (HWHM). Due to short core-hole lifetime
    delta = 0.2
    # Occupation restrictions, used when spectra are generated
    l = 2
    restrictions = {}
    # Restriction on impurity orbitals
    indices = frozenset(c2i(nBaths,(l,s,m)) for s in range(2) for m in range(-l,l+1))
    restrictions[indices] = (n0imp[l] - 1, n0imp[l] + 3)
    # Restriction on valence bath orbitals
    indices = []
    for b in range(valBaths[l]):
        indices.append(c2i(nBaths, (l, b)))
    restrictions[frozenset(indices)] = (valBaths[l] - 2, valBaths[l])
    # Restriction on conduction bath orbitals
    indices = []
    for b in range(valBaths[l], nBaths[l]):
        indices.append(c2i(nBaths, (l, b)))
    restrictions[frozenset(indices)] = (0, 0)
    # XAS polarization vectors.
    epsilons = [[1,0,0],[0,1,0],[0,0,1]] # [[0,0,1]]
    # RIXS parameters
    # Polarization vectors, of in and outgoing photon.
    epsilonsRIXSin = [[1,0,0],[0,1,0],[0,0,1]]  # [[0,0,1]]
    epsilonsRIXSout = [[1,0,0],[0,1,0],[0,0,1]] # [[0,0,1]]
    wIn = np.linspace(-10,15,50)
    wLoss = np.linspace(-2,12,4000)
    # Smearing, half with half maximum (HWHM).
    # Due to finite lifetime of excited states
    deltaRIXS = 0.050
    # NIXS parameters
    qsNIXS = [2*np.array([1,1,1])/np.sqrt(3),7*np.array([1,1,1])/np.sqrt(3)]
    # Smearing, half with half maximum (HWHM). Due to finite lifetime of excited states
    deltaNIXS = 0.100
    # Angular momentum of final and initial orbitals in the NIXS excitation process.
    liNIXS,ljNIXS = 2,2
    # File name of file containing radial mesh and radial part of final
    # and initial orbitals in the NIXS excitation process.
    radialFileName = os.path.dirname(sys.argv[0])[:-7] + 'radialOrbitals/Co3d.dat'
    data = np.loadtxt(radialFileName)
    radialMesh = data[:,0]
    RiNIXS = data[:,1]
    RjNIXS = np.copy(RiNIXS)
    # -----------------------

    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1) + nBath for ang, nBath in nBaths.items())
    if rank == 0: print("#spin-orbitals:",n_spin_orbitals)

    # Hamiltonian
    if rank == 0: print('Construct the Hamiltonian operator...')
    hOp = get_hamiltonian_operator(nBaths, valBaths, [Fdd, Fpp, Fpd, Gpd],
                                   [xi_2p, xi_3d],
                                   [n0imp, chargeTransferCorrection],
                                   [eImp3d, deltaO], hField,
                                   [vValEg, vValT2g, vConEg, vConT2g],
                                   [eValEg, eValT2g, eConEg, eConT2g])
    # Measure how many physical processes the Hamiltonian contains.
    if rank == 0: print('{:d} processes in the Hamiltonian.'.format(len(hOp)))
    # Many body basis for the ground state
    if rank == 0: print('Create basis...')
    basis = finite.get_basis(nBaths, valBaths, dnValBaths, dnConBaths,
                             dnTol, n0imp)
    if rank == 0: print('#basis states = {:d}'.format(len(basis)))
    # Diagonalization of restricted active space Hamiltonian
    es, psis = finite.eigensystem(n_spin_orbitals, hOp, basis, nPsiMax)

    if rank == 0:
        print("time(ground_state) = {:.2f} seconds \n".format(time.time()-t0))
        t0 = time.time()

    # Calculate static expectation values
    finite.printThermalExpValues(nBaths, es, psis)
    finite.printExpValues(nBaths, es, psis)

    # Print Slater determinants and weights
    if rank == 0:
        print('Slater determinants/product states and correspoinding weights')
        weights = []
        for i, psi in enumerate(psis):
            print('Eigenstate {:d}.'.format(i))
            print('Consists of {:d} product states.'.format(len(psi)))
            ws = np.array([ abs(a)**2 for a in psi.values() ])
            s = np.array([ ps for ps in psi.keys() ])
            j = np.argsort(ws)
            ws = ws[j[-1::-1]]
            s = s[j[-1::-1]]
            weights.append(ws)
            if nPrintSlaterWeights > 0:
                print('Highest (product state) weights:')
                print(ws[:nPrintSlaterWeights])
                print('Corresponding product states:')
                print(s[:nPrintSlaterWeights])
                print('')

    # Calculate density matrix
    if rank == 0:
        print('Density matrix (in cubic harmonics basis):')
        for i, psi in enumerate(psis):
            print('Eigenstate {:d}'.format(i))
            n = finite.getDensityMatrixCubic(nBaths, psi)
            print('#density matrix elements: {:d}'.format(len(n)))
            for e, ne in n.items():
                if abs(ne) > tolPrintOccupation:
                    if e[0] == e[1]:
                        print('Diagonal: (i,s) =',e[0],', occupation = {:7.2f}'.format(ne))
                    else:
                        print('Off-diagonal: (i,si), (j,sj) =',e,', {:7.2f}'.format(ne))
            print('')

    # Save some information to disk
    if rank == 0:
        # Most of the input parameters. Dictonaries can be stored in this file format.
        np.savez_compressed('data', l1=l1, l2=l2, nBaths=nBaths,
                            valBaths=valBaths,
                            n0imp=n0imp, dnTol=dnTol,
                            dnValBaths=dnValBaths, dnConBaths=dnConBaths,
                            Fdd=Fdd, Fpp=Fpp, Fpd=Fpd, Gpd=Gpd,
                            xi_2p=xi_2p, xi_3d=xi_3d,
                            chargeTransferCorrection=chargeTransferCorrection,
                            eImp3d=eImp3d, deltaO=deltaO,
                            hField=hField,
                            eBath=[eValEg, eValT2g, eConEg, eConT2g],
                            vBath=[vValEg, vValT2g, vConEg, vConT2g],
                            nPsiMax=nPsiMax,
                            T=T, energy_cut=energy_cut, delta=delta,
                            restrictions=restrictions,
                            epsilons=epsilons,
                            epsilonsRIXSin=epsilonsRIXSin,
                            epsilonsRIXSout=epsilonsRIXSout,
                            deltaRIXS=deltaRIXS,
                            deltaNIXS=deltaNIXS,
                            n_spin_orbitals=n_spin_orbitals,
                            hOp=hOp)
        # Save some of the arrays.
        if printH5:
            import h5py
            # This file format does not support dictonaries.
            h5f = h5py.File('spectra.h5','w')
            h5f.create_dataset('E',data=es)
            h5f.create_dataset('w',data=w)
            h5f.create_dataset('wIn',data=wIn)
            h5f.create_dataset('wLoss',data=wLoss)
            h5f.create_dataset('qsNIXS',data=qsNIXS)
            h5f.create_dataset('r',data=radialMesh)
            h5f.create_dataset('RiNIXS',data=RiNIXS)
            h5f.create_dataset('RjNIXS',data=RjNIXS)
        else:
            np.savez_compressed('spectraInfo', E=es, w=w, wIn=wIn, wLoss=wLoss,
                                qsNIXS=qsNIXS, r=radialMesh, RiNIXS=RiNIXS,
                                RjNIXS=RjNIXS)

    if rank == 0:
        print("time(expectation values) = {:.2f} seconds \n".format(time.time()-t0))
        t0 = time.time()

    # Consider from now on only eigenstates with low energy
    es = tuple( e for e in es if e - es[0] < energy_cut )
    psis = tuple( psis[i] for i in range(len(es)) )
    if rank == 0: print("Consider {:d} eigenstates for the spectra \n".format(len(es)))

    if rank == 0: print('Create 3d inverse photoemission and photoemission spectra...')
    # Transition operators
    tOpsIPS = spectra.getInversePhotoEmissionOperators(nBaths, l=2)
    tOpsPS = spectra.getPhotoEmissionOperators(nBaths, l=2)
    if rank == 0: print("Inverse photoemission Green's function..")
    gsIPS = spectra.getSpectra(n_spin_orbitals, hOp, tOpsIPS, psis, es, w,
                               delta, restrictions)
    if rank == 0: print("Photoemission Green's function..")
    gsPS = spectra.getSpectra(n_spin_orbitals, hOp, tOpsPS, psis, es, -w,
                              -delta, restrictions)
    gsPS *= -1
    gs = gsPS + gsIPS
    if rank == 0:
        print('#eigenstates = {:d}'.format(np.shape(gs)[0]))
        print('#spin orbitals = {:d}'.format(np.shape(gs)[1]))
        print('#mesh points = {:d}'.format(np.shape(gs)[2]))
    # Thermal average
    a = thermal_average(es[:np.shape(gs)[0]],-gs.imag,T=T)
    if rank == 0:
        if printH5:
            h5f.create_dataset('PS',data=-gs.imag)
            h5f.create_dataset('PSthermal',data=a)
        else:
            np.savez_compressed('spectraPS',PS=-gs.imag,PSthermal=a)
    # Sum over transition operators
    aSum = np.sum(a,axis=0)
    # Save spectra to disk
    if rank == 0:
        tmp = [w,aSum]
        # Each transition operator seperatly
        for i in range(np.shape(a)[0]): tmp.append(a[i,:])
        print("Save spectra to disk...\n")
        np.savetxt('PS.dat',np.array(tmp).T,fmt='%8.4f',
                   header='E  sum  T1  T2  T3 ...')
    if rank == 0:
        print("time(PS) = {:.2f} seconds \n".format(time.time()-t0))
        t0 = time.time()

    if rank == 0: print('Create core 2p x-ray photoemission spectra (XPS) ...')
    # Transition operators
    tOpsPS = spectra.getPhotoEmissionOperators(nBaths,l=1)
    # Photoemission Green's function
    gs = spectra.getSpectra(n_spin_orbitals, hOp, tOpsPS, psis, es, -w,
                            -delta, restrictions)
    gs *= -1
    if rank == 0:
        print('#eigenstates = {:d}'.format(np.shape(gs)[0]))
        print('#spin orbitals = {:d}'.format(np.shape(gs)[1]))
        print('#mesh points = {:d}'.format(np.shape(gs)[2]))
    # Thermal average
    a = thermal_average(es[:np.shape(gs)[0]],-gs.imag,T=T)
    if rank == 0:
        if printH5:
            h5f.create_dataset('XPS',data=-gs.imag)
            h5f.create_dataset('XPSthermal',data=a)
        else:
            np.savez_compressed('spectraXPS',XPS=-gs.imag,XPSthermal=a)
    # Sum over transition operators
    aSum = np.sum(a,axis=0)
    # Save spectra to disk
    if rank == 0:
        tmp = [w,aSum]
        # Each transition operator seperatly
        for i in range(np.shape(a)[0]): tmp.append(a[i,:])
        print("Save spectra to disk...\n")
        np.savetxt('XPS.dat',np.array(tmp).T,fmt='%8.4f',
                   header='E  sum  T1  T2  T3 ...')
    if rank == 0:
        print("time(XPS) = {:.2f} seconds \n".format(time.time()-t0))
        t0 = time.time()

    if rank == 0: print('Create NIXS spectra...')
    # Transition operator: exp(iq*r)
    tOps = spectra.getNIXSOperators(nBaths,qsNIXS,liNIXS,ljNIXS,
                                    RiNIXS,RjNIXS,radialMesh)
    # Green's function
    gs = spectra.getSpectra(n_spin_orbitals, hOp, tOps, psis, es, wLoss,
                            deltaNIXS, restrictions)
    if rank == 0:
        print('#eigenstates = {:d}'.format(np.shape(gs)[0]))
        print('#q-points = {:d}'.format(np.shape(gs)[1]))
        print('#mesh points = {:d}'.format(np.shape(gs)[2]))
    # Thermal average
    a = thermal_average(es[:np.shape(gs)[0]],-gs.imag,T=T)
    if rank == 0:
        if printH5:
            h5f.create_dataset('NIXS',data=-gs.imag)
            h5f.create_dataset('NIXSthermal',data=a)
        else:
            np.savez_compressed('spectraNIXS',NIXS=-gs.imag,NIXSthermal=a)
    # Sum over q-points
    aSum = np.sum(a,axis=0)
    # Save spectra to disk
    if rank == 0:
        tmp = [wLoss,aSum]
        # Each q-point seperatly
        for i in range(np.shape(a)[0]): tmp.append(a[i,:])
        print("Save spectra to disk...\n")
        np.savetxt('NIXS.dat',np.array(tmp).T,fmt='%8.4f',
                   header='E  sum  T1  T2  T3 ...')

    if rank == 0:
        print("time(NIXS) = {:.2f} seconds \n".format(time.time()-t0))
        t0 = time.time()


    if rank == 0: print('Create XAS spectra...')
    # Dipole transition operators
    tOps = spectra.getDipoleOperators(nBaths,epsilons)
    # Green's function
    gs = spectra.getSpectra(n_spin_orbitals, hOp, tOps, psis, es, w,
                            delta, restrictions)
    if rank == 0:
        print('#eigenstates = {:d}'.format(np.shape(gs)[0]))
        print('#polarizations = {:d}'.format(np.shape(gs)[1]))
        print('#mesh points = {:d}'.format(np.shape(gs)[2]))
    # Thermal average
    a = thermal_average(es[:np.shape(gs)[0]],-gs.imag,T=T)
    if rank == 0:
        if printH5:
            h5f.create_dataset('XAS',data=-gs.imag)
            h5f.create_dataset('XASthermal',data=a)
        else:
            np.savez_compressed('spectraXAS',XAS=-gs.imag,XASthermal=a)
    # Sum over transition operators
    aSum = np.sum(a,axis=0)
    # Save spectra to disk
    if rank == 0:
        tmp = [w,aSum]
        # Each transition operator seperatly
        for i in range(np.shape(a)[0]): tmp.append(a[i,:])
        print("Save spectra to disk...\n")
        np.savetxt('XAS.dat',np.array(tmp).T,fmt='%8.4f',
                   header='E  sum  T1  T2  T3 ...')
    if rank == 0:
        print("time(XAS) = {:.2f} seconds \n".format(time.time()-t0))
        t0 = time.time()

    if rank == 0: print('Create RIXS spectra...')
    # Dipole 2p -> 3d transition operators
    tOpsIn = spectra.getDipoleOperators(nBaths, epsilonsRIXSin)
    # Dipole 3d -> 2p transition operators
    tOpsOut = spectra.getDaggeredDipoleOperators(nBaths, epsilonsRIXSout)
    # Green's function
    gs = spectra.getRIXSmap(n_spin_orbitals, hOp, tOpsIn, tOpsOut, psis, es,
                            wIn, wLoss, delta, deltaRIXS, restrictions)
    if rank == 0:
        print('#eigenstates = {:d}'.format(np.shape(gs)[0]))
        print('#in-polarizations = {:d}'.format(np.shape(gs)[1]))
        print('#out-polarizations = {:d}'.format(np.shape(gs)[2]))
        print('#mesh points of input energy = {:d}'.format(np.shape(gs)[3]))
        print('#mesh points of energy loss = {:d}'.format(np.shape(gs)[4]))
    # Thermal average
    a = thermal_average(es[:np.shape(gs)[0]],-gs.imag,T=T)
    if rank == 0:
        if printH5:
            h5f.create_dataset('RIXS',data=-gs.imag)
            h5f.create_dataset('RIXSthermal',data=a)
        else:
            np.savez_compressed('spectraRIXS',RIXS=-gs.imag,RIXSthermal=a)
    # Sum over transition operators
    aSum = np.sum(a,axis=(0,1))
    # Save spectra to disk
    if rank == 0:
        print("Save spectra to disk...\n")
        # I[wLoss,wIn], with wLoss on first column and wIn on first row.
        tmp = np.zeros((len(wLoss)+1,len(wIn)+1),dtype=np.float32)
        tmp[0,0] = len(wIn)
        tmp[0,1:] = wIn
        tmp[1:,0] = wLoss
        tmp[1:,1:] = aSum.T
        tmp.tofile('RIXS.bin')
    if rank == 0:
        print("time(RIXS) = {:.2f} seconds \n".format(time.time()-t0))
        t0 = time.time()

    if rank == 0 and printH5: h5f.close()
    print('Script finished for rank:',rank)


def get_hamiltonian_operator(nBaths, valBaths, slaterCondon, SOCs,
                             DCinfo, impurityInfo, hField,
                             vHoppings, eBaths, bath_state_basis = 'spherical'):
    """
    Return the Hamiltonian, in operator form.

    The impurity orbitals are spherical harmonics orbitals.
    The bath states can be expressed in different basis.

    Parameters
    ----------
    nBaths : dict
        Number of bath states for each angular momentum.
    nBaths : dict
        Number of valence bath states for each angular momentum.
    slaterCondon : list
        List of Slater-Condon parameters.
    SOCs : list
        List of SOC parameters.
    DCinfo : list
        Contains information needed for the double counting energy.
    impurityInfo : list
        Contains information of 3d single particle energies.
    hField : list
        External magnetic field.
        Elements hx,hy,hz
    vHoppings : list
        Contains information about hybridization hoppings.
    eBaths : list
        Contains information about bath energies.
    bath_state_basis : str
        'spherical' or 'cubic'.
        Which basis to use for the bath states.

    """
    # Divide up input parameters to more concrete variables
    Fdd, Fpp, Fpd, Gpd = slaterCondon
    xi_2p, xi_3d = SOCs
    n0imp, chargeTransferCorrection = DCinfo
    eImp3d, deltaO = impurityInfo
    hx, hy, hz = hField
    vValEg, vValT2g, vConEg, vConT2g = vHoppings
    eValEg, eValT2g, eConEg, eConT2g = eBaths

    # Calculate the U operator, in spherical harmonics basis.
    uOperator = finite.get2p3dSlaterCondonUop(Fdd=Fdd, Fpp=Fpp,
                                              Fpd=Fpd, Gpd=Gpd)
    # Add SOC, in spherical harmonics basis.
    SOC2pOperator = finite.getSOCop(xi_2p, l=1)
    SOC3dOperator = finite.getSOCop(xi_3d, l=2)

    # Double counting (DC) correction values.
    # MLFT DC
    dc = finite.dc_MLFT(n3d_i=n0imp[2], c=chargeTransferCorrection, Fdd=Fdd,
                        n2p_i=n0imp[1], Fpd=Fpd, Gpd=Gpd)
    eDCOperator = {}
    for il,l in enumerate([2,1]):
        for s in range(2):
            for m in range(-l,l+1):
                eDCOperator[(((l, s, m), 'c'), ((l, s, m), 'a'))] = -dc[il]

    # Calculate impurity 3d Hamiltonian.
    # First formulate in cubic harmonics basis and then rotate to
    # the spherical harmonics basis.
    l = 2
    eImpEg = eImp3d + 3/5*deltaO
    eImpT2g = eImp3d - 2/5*deltaO
    hImp3d = np.zeros((2*l+1, 2*l+1))
    np.fill_diagonal(hImp3d, (eImpEg, eImpEg, eImpT2g, eImpT2g, eImpT2g))
    # Convert to spherical harmonics basis
    u = finite.get_spherical_2_cubic_matrix(spinpol=False, l=l)
    hImp3d = np.dot(u, np.dot(hImp3d, np.conj(u.T)))
    # Convert from matrix to operator form
    # Also add spin
    hImp3dOperator = {}
    for i,mi in enumerate(range(-l, l+1)):
        for j,mj in enumerate(range(-l, l+1)):
            if hImp3d[i,j] != 0:
                for s in range(2):
                    hImp3dOperator[(((l, s, mi), 'c'), ((l, s, mj), 'a'))] = hImp3d[i,j]

    # Magnetic field
    l = 2
    hHfieldOperator = {}
    for m in range(-l, l+1):
        hHfieldOperator[(((l, 1, m), 'c'), ((l, 0, m), 'a'))] = hx*1/2.
        hHfieldOperator[(((l, 0, m), 'c'), ((l, 1, m), 'a'))] = hx*1/2.
        hHfieldOperator[(((l, 1, m), 'c'), ((l, 0, m), 'a'))] = -hy*1/2.*1j
        hHfieldOperator[(((l, 0, m), 'c'), ((l, 1, m), 'a'))] = hy*1/2.*1j
        for s in range(2):
            hHfieldOperator[(((l, s, m), 'c'), ((l, s, m), 'a'))] = hz*1/2 if s==1 else -hz*1/2

    # Bath (3d) on-site energies and hoppings.
    # Calculate hopping terms between bath and impurity.
    # First formulate the terms in the cubic harmonics basis.
    l = 2
    assert valBaths[l] == 10
    assert nBaths[l]-valBaths[l] == 0 or nBaths[l]-valBaths[l] == 10
    vVal3d = np.zeros((2*l+1, 2*l+1))
    vCon3d = np.zeros((2*l+1, 2*l+1))
    eBathVal3d = np.zeros((2*l+1, 2*l+1))
    eBathCon3d = np.zeros((2*l+1, 2*l+1))
    np.fill_diagonal(vVal3d, (vValEg, vValEg, vValT2g, vValT2g, vValT2g))
    np.fill_diagonal(vCon3d, (vConEg, vConEg, vConT2g, vConT2g, vConT2g))
    np.fill_diagonal(eBathVal3d, (eValEg, eValEg, eValT2g, eValT2g, eValT2g))
    np.fill_diagonal(eBathCon3d, (eConEg, eConEg, eConT2g, eConT2g, eConT2g))
    # For the bath states, we can rotate to any basis.
    # Which bath state basis to use is determined selected here.
    if bath_state_basis == 'spherical':
        # One example is to use spherical harmonics basis for the bath states.
        # This implies the following rotation matrix:
        u_bath = u
    elif bath_state_basis == 'cubic':
        # One example is to keep the cubic harmonics basis for the bath states.
        # This implies the following rotation matrix:
        u_bath = np.eye(np.shape(u)[0])
    else:
        sys.exit('Design of this basis is not (yet) implemented.')
    # Rotate the bath energies and the hopping parameters
    vVal3d = np.dot(u_bath, np.dot(vVal3d, np.conj(u.T)))
    vCon3d = np.dot(u_bath, np.dot(vCon3d, np.conj(u.T)))
    eBathVal3d = np.dot(u_bath, np.dot(eBathVal3d, np.conj(u_bath.T)))
    eBathCon3d = np.dot(u_bath, np.dot(eBathCon3d, np.conj(u_bath.T)))
    # Convert from matrix to operator form.
    # Also introduce spin.
    hHoppOperator = {}
    eBath3dOperator = {}
    # Loop over spin
    for s in range(2):
        # Loop over impurity orbitals
        for i, mi in enumerate(range(-l, l+1)):
            # Bath state index for valence bath states.
            bi_val = s*(2*l+1) + i
            # Bath state index for conduction bath states.
            bi_con = 2*(2*l+1) + bi_val
            # Loop over impurity orbitals
            for j, mj in enumerate(range(-l, l+1)):
                # Bath state index for valence bath states.
                bj_val = s*(2*l+1) + j
                # Bath state index for conduction bath states.
                bj_con = 2*(2*l+1) + bj_val
                # Hamiltonian values related to valence bath states.
                vHopp = vVal3d[i,j]
                eBath = eBathVal3d[i,j]
                if vHopp != 0:
                    hHoppOperator[(((l, bi_val), 'c'), ((l, s, mj), 'a'))] = vHopp
                    hHoppOperator[(((l, s, mj), 'c'), ((l, bi_val), 'a'))] = vHopp.conjugate()
                if eBath != 0:
                    eBath3dOperator[(((l, bi_val), 'c'), ((l, bj_val),'a'))] = eBath
                # Only add the processes related to the conduction bath states if they are
                # in the basis.
                if nBaths[l]-valBaths[l] == 10:
                    # Hamiltonian values related to conduction bath states.
                    vHopp = vCon3d[i,j]
                    eBath = eBathCon3d[i,j]
                    if vHopp != 0:
                        hHoppOperator[(((l, bi_con), 'c'), ((l, s, mj), 'a'))] = vHopp
                        hHoppOperator[(((l, s, mj), 'c'), ((l, bi_con), 'a'))] = vHopp.conjugate()
                    if eBath != 0:
                        eBath3dOperator[(((l, bi_con), 'c'), ((l, bj_con),'a'))] = eBath

    # Add Hamiltonian terms to one operator.
    hOperator = finite.addOps([uOperator,
                               hImp3dOperator,
                               hHfieldOperator,
                               SOC2pOperator,
                               SOC3dOperator,
                               eDCOperator,
                               hHoppOperator,
                               eBath3dOperator])
    # Convert spin-orbital and bath state indices to a single index notation.
    hOp = {}
    for process,value in hOperator.items():
        hOp[tuple((c2i(nBaths, spinOrb), action) for spinOrb, action in process)] = value
    return hOp


if __name__== "__main__":
    main()
