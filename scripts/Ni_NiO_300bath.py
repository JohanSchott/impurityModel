#!/usr/bin/env python3

# Script for solving many-body impurity problem.

import numpy as np
import scipy.sparse.linalg
from collections import OrderedDict
import sys,os
from mpi4py import MPI
import pickle
import time
# Local local stuff
from impurityModel.ed import spectra
from impurityModel.ed import finite
from impurityModel.ed.finite import c2i
from impurityModel.ed.average import k_B, thermal_average


def main():

    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.rank
    ranks = comm.size

    if rank == 0: t0 = time.time()

    # -----------------------
    # Read the non-relativistic non-interacting Hamiltonian operator from file.
    h0FileName = os.path.dirname(sys.argv[0])[:-7] + 'h0/h0_NiO_300bath.pickle'
    with open(h0FileName, 'rb') as handle:
        h0_operator = pickle.loads(handle.read())
    # System specific information
    l1, l2 = 1, 2 # Angular momentum
    # Number of bath states.
    nBaths = OrderedDict()
    nBaths[l1] = 0
    nBaths[l2] = 300
    # Number of valence bath states.
    valBaths = OrderedDict()
    valBaths[l1] = 0
    valBaths[l2] = 300
    # -----------------------
    # Basis occupation information.
    # Angular momentum : initial impurity occupation
    n0imp = OrderedDict()
    n0imp[l1] = 6 # 0 = empty, 2*(2*l1+1) = Full occupation
    n0imp[l2] = 8 # 8 for Ni+2
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
    Fdd = [7.5, 0, 9.9, 0, 6.6]
    Fpp = [0, 0, 0]
    Fpd = [8.9, 0, 6.8]
    Gpd = [0, 5, 0, 2.8]
    # SOC values
    xi_2p = 11.629
    xi_3d = 0.096
    # Double counting parameter
    chargeTransferCorrection = 1.5 # 3.5 gives good position of PS peaks
    # Magnetic field
    hField = [0, 0, 0.0001]
    # -----------------------
    # Maximum number of eigenstates to consider
    nPsiMax = 5
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
    wIn = np.linspace(-10,20,50)
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
    radialFileName = os.path.dirname(sys.argv[0])[:-7] + 'radialOrbitals/Ni3d.dat'
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
                                   hField, h0_operator)
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
                            hField=hField,
                            h0_operator=h0_operator,
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
                             DCinfo, hField, h0_operator):
    """
    Return the Hamiltonian, in operator form.

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
    hField : list
        External magnetic field.
        Elements hx,hy,hz
    h0_operator : dict
        The non-relativistic non-interacting Hamiltonian in operator form.
        tuple : complex,
        where each tuple describes a process of several steps.
        Each step is described by a tuple of the form:
        (spin_orb, 'c') or (spin_orb, 'a'),
        where spin_orb is a tuple of the form (l, s, m) or (l, b) or ((l_a, l_b), b).

    Returns
    -------
    hOp : dict
        The Hamiltonian in operator form.
        tuple : complex,
        where each tuple describes a process of several steps.
        Each step is described by a tuple of the form: (i,'c') or (i,'a'),
        where i is a spin-orbital index.

    """
    # Divide up input parameters to more concrete variables
    Fdd, Fpp, Fpd, Gpd = slaterCondon
    xi_2p, xi_3d = SOCs
    n0imp, chargeTransferCorrection = DCinfo
    hx, hy, hz = hField

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
    for il, l in enumerate([2,1]):
        for s in range(2):
            for m in range(-l, l+1):
                eDCOperator[(((l, s, m), 'c'), ((l, s, m), 'a'))] = -dc[il]

    # Magnetic field
    hHfieldOperator = {}
    l = 2
    for m in range(-l, l+1):
        hHfieldOperator[(((l, 1, m), 'c'), ((l, 0, m), 'a'))] = hx*1/2.
        hHfieldOperator[(((l, 0, m), 'c'), ((l, 1, m), 'a'))] = hx*1/2.
        hHfieldOperator[(((l, 1, m), 'c'), ((l, 0, m), 'a'))] = -hy*1/2.*1j
        hHfieldOperator[(((l, 0, m), 'c'), ((l, 1, m), 'a'))] = hy*1/2.*1j
        for s in range(2):
            hHfieldOperator[(((l, s, m), 'c'), ((l, s, m), 'a'))] = hz*1/2 if s==1 else -hz*1/2

    # Add Hamiltonian terms to one operator.
    hOperator = finite.addOps([uOperator,
                               hHfieldOperator,
                               SOC2pOperator,
                               SOC3dOperator,
                               eDCOperator,
                               h0_operator])
    # Convert spin-orbital and bath state indices to a single index notation.
    hOp = {}
    for process,value in hOperator.items():
        hOp[tuple((c2i(nBaths, spinOrb), action) for spinOrb, action in process)] = value
    return hOp


if __name__== "__main__":
    main()
