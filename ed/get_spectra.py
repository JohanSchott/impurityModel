
"""
Script for calculating various spectra.

"""

import numpy as np
import scipy.sparse.linalg
from collections import OrderedDict
import sys,os
from mpi4py import MPI
import pickle
import time
import argparse
import h5py
# Local local stuff
from impurityModel.ed import spectra
from impurityModel.ed import finite
from impurityModel.ed.finite import c2i
from impurityModel.ed.average import k_B, thermal_average


def main(h0_filename,
         radial_filename,
         ls, nBaths, nValBaths,
         n0imps, dnTols, dnValBaths, dnConBaths,
         Fdd, Fpp, Fpd, Gpd,
         xi_2p, xi_3d, chargeTransferCorrection,
         hField, nPsiMax,
         nPrintSlaterWeights, tolPrintOccupation,
         T, energy_cut,
         delta, deltaRIXS, deltaNIXS):
    """
    First find the lowest eigenstates and then use them to calculate various spectra.

    Parameters
    ----------
    h0_filename : str
        Filename of the non-relativistic non-interacting Hamiltonian operator.
    radial_filename : str
        File name of file containing radial mesh and radial part of final
        and initial orbitals in the NIXS excitation process.
    ls : tuple
        Angular momenta of correlated orbitals.
    nBaths : tuple
        Number of bath states,
        for each angular momentum.
    nValBaths : tuple
        Number of valence bath states,
        for each angular momentum.
    n0imps : tuple
        Initial impurity occupation.
    dnTols : tuple
        Max devation from initial impurity occupation,
        for each angular momentum.
    dnValBaths : tuple
        Max number of electrons to leave valence bath orbitals,
        for each angular momentum.
    dnConBaths : tuple
        Max number of electrons to enter conduction bath orbitals,
        for each angular momentum.
    Fdd : tuple
        Slater-Condon parameters Fdd. This assumes d-orbitals.
    Fpp : tuple
        Slater-Condon parameters Fpp. This assumes p-orbitals.
    Fpd : tuple
        Slater-Condon parameters Fpd. This assumes p- and d-orbitals.
    Gpd : tuple
        Slater-Condon parameters Gpd. This assumes p- and d-orbitals.
    xi_2p : float
        SOC value for p-orbitals. This assumes p-orbitals.
    xi_3d : float
        SOC value for d-orbitals. This assumes d-orbitals.
    chargeTransferCorrection : float
        Double counting parameter
    hField : tuple
        Magnetic field.
    nPsiMax : int
        Maximum number of eigenstates to consider.
    nPrintSlaterWeights : int
        Printing parameter.
    tolPrintOccupation : float
        Printing parameter.
    T : float
        Temperature (Kelvin)
    energy_cut : float
        How many k_B*T above lowest eigenenergy to consider.
    delta : float
        Smearing, half width half maximum (HWHM). Due to short core-hole lifetime.
    deltaRIXS : float
        Smearing, half width half maximum (HWHM).
        Due to finite lifetime of excited states.
    deltaNIXS : float
        Smearing, half width half maximum (HWHM).
        Due to finite lifetime of excited states.


    """

    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0: t0 = time.time()

    # -- System information --
    nBaths = OrderedDict(zip(ls, nBaths))
    nValBaths = OrderedDict(zip(ls, nValBaths))

    # -- Basis occupation information --
    n0imps = OrderedDict(zip(ls, n0imps))
    dnTols = OrderedDict(zip(ls, dnTols))
    dnValBaths = OrderedDict(zip(ls, dnValBaths))
    dnConBaths = OrderedDict(zip(ls, dnConBaths))

    # -- Spectra information --
    # Energy cut in eV.
    energy_cut *= k_B*T
    # XAS parameters
    # Energy-mesh
    w = np.linspace(-25, 25, 3000)
    # Each element is a XAS polarization vector.
    epsilons = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # [[0,0,1]]
    # RIXS parameters
    # Polarization vectors, of in and outgoing photon.
    epsilonsRIXSin = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # [[0,0,1]]
    epsilonsRIXSout = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # [[0,0,1]]
    wIn = np.linspace(-10, 20, 50)
    wLoss = np.linspace(-2, 12, 4000)
    # NIXS parameters
    qsNIXS = [2 * np.array([1, 1, 1]) / np.sqrt(3), 7 * np.array([1, 1, 1]) / np.sqrt(3)]
    # Angular momentum of final and initial orbitals in the NIXS excitation process.
    liNIXS,ljNIXS = 2, 2

    # -- Occupation restrictions for excited states --
    l = 2
    restrictions = {}
    # Restriction on impurity orbitals
    indices = frozenset(c2i(nBaths, (l, s, m)) for s in range(2) for m in range(-l, l + 1))
    restrictions[indices] = (n0imps[l] - 1, n0imps[l] + dnTols[l] + 1)
    # Restriction on valence bath orbitals
    indices = []
    for b in range(nValBaths[l]):
        indices.append(c2i(nBaths, (l, b)))
    restrictions[frozenset(indices)] = (nValBaths[l] - dnValBaths[l], nValBaths[l])
    # Restriction on conduction bath orbitals
    indices = []
    for b in range(nValBaths[l], nBaths[l]):
        indices.append(c2i(nBaths, (l, b)))
    restrictions[frozenset(indices)] = (0, dnConBaths[l])

    # Read the radial part of correlated orbitals
    radialMesh, RiNIXS = np.loadtxt(radial_filename).T
    RjNIXS = np.copy(RiNIXS)

    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    if rank == 0: print("#spin-orbitals:", n_spin_orbitals)

    # Hamiltonian
    if rank == 0: print('Construct the Hamiltonian operator...')
    hOp = get_hamiltonian_operator(nBaths, nValBaths, [Fdd, Fpp, Fpd, Gpd],
                                   [xi_2p, xi_3d],
                                   [n0imps, chargeTransferCorrection],
                                   hField,
                                   h0_filename)
    # Measure how many physical processes the Hamiltonian contains.
    if rank == 0: print('{:d} processes in the Hamiltonian.'.format(len(hOp)))
    # Many body basis for the ground state
    if rank == 0: print('Create basis...')
    basis = finite.get_basis(nBaths, nValBaths, dnValBaths, dnConBaths,
                             dnTols, n0imps)
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
        np.savez_compressed('data', ls=ls, nBaths=nBaths,
                            nValBaths=nValBaths,
                            n0imps=n0imps, dnTols=dnTols,
                            dnValBaths=dnValBaths, dnConBaths=dnConBaths,
                            Fdd=Fdd, Fpp=Fpp, Fpd=Fpd, Gpd=Gpd,
                            xi_2p=xi_2p, xi_3d=xi_3d,
                            chargeTransferCorrection=chargeTransferCorrection,
                            hField=hField,
                            h0_filename=h0_filename,
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
        # HDF5-format does not directly support dictonaries.
        h5f = h5py.File('spectra.h5','w')
        h5f.create_dataset('E',data=es)
        h5f.create_dataset('w',data=w)
        h5f.create_dataset('wIn',data=wIn)
        h5f.create_dataset('wLoss',data=wLoss)
        h5f.create_dataset('qsNIXS',data=qsNIXS)
        h5f.create_dataset('r',data=radialMesh)
        h5f.create_dataset('RiNIXS',data=RiNIXS)
        h5f.create_dataset('RjNIXS',data=RjNIXS)

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
    a = thermal_average(es[:np.shape(gs)[0]], -gs.imag, T=T)
    if rank == 0:
        h5f.create_dataset('PS', data=-gs.imag)
        h5f.create_dataset('PSthermal', data=a)
    # Sum over transition operators
    aSum = np.sum(a, axis=0)
    # Save spectra to disk
    if rank == 0:
        tmp = [w, aSum]
        # Each transition operator seperatly
        for i in range(np.shape(a)[0]): tmp.append(a[i,:])
        print("Save spectra to disk...\n")
        np.savetxt('PS.dat', np.array(tmp).T, fmt='%8.4f',
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
    a = thermal_average(es[:np.shape(gs)[0]], -gs.imag, T=T)
    if rank == 0:
        h5f.create_dataset('XPS', data=-gs.imag)
        h5f.create_dataset('XPSthermal', data=a)
    # Sum over transition operators
    aSum = np.sum(a, axis=0)
    # Save spectra to disk
    if rank == 0:
        tmp = [w, aSum]
        # Each transition operator seperatly
        for i in range(np.shape(a)[0]): tmp.append(a[i,:])
        print("Save spectra to disk...\n")
        np.savetxt('XPS.dat', np.array(tmp).T, fmt='%8.4f',
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
    a = thermal_average(es[:np.shape(gs)[0]], -gs.imag, T=T)
    if rank == 0:
        h5f.create_dataset('NIXS', data=-gs.imag)
        h5f.create_dataset('NIXSthermal', data=a)
    # Sum over q-points
    aSum = np.sum(a, axis=0)
    # Save spectra to disk
    if rank == 0:
        tmp = [wLoss, aSum]
        # Each q-point seperatly
        for i in range(np.shape(a)[0]): tmp.append(a[i,:])
        print("Save spectra to disk...\n")
        np.savetxt('NIXS.dat', np.array(tmp).T, fmt='%8.4f',
                   header='E  sum  T1  T2  T3 ...')

    if rank == 0:
        print("time(NIXS) = {:.2f} seconds \n".format(time.time()-t0))
        t0 = time.time()


    if rank == 0: print('Create XAS spectra...')
    # Dipole transition operators
    tOps = spectra.getDipoleOperators(nBaths, epsilons)
    # Green's function
    gs = spectra.getSpectra(n_spin_orbitals, hOp, tOps, psis, es, w,
                            delta, restrictions)
    if rank == 0:
        print('#eigenstates = {:d}'.format(np.shape(gs)[0]))
        print('#polarizations = {:d}'.format(np.shape(gs)[1]))
        print('#mesh points = {:d}'.format(np.shape(gs)[2]))
    # Thermal average
    a = thermal_average(es[:np.shape(gs)[0]], -gs.imag, T=T)
    if rank == 0:
        h5f.create_dataset('XAS', data=-gs.imag)
        h5f.create_dataset('XASthermal', data=a)
    # Sum over transition operators
    aSum = np.sum(a, axis=0)
    # Save spectra to disk
    if rank == 0:
        tmp = [w, aSum]
        # Each transition operator seperatly
        for i in range(np.shape(a)[0]): tmp.append(a[i,:])
        print("Save spectra to disk...\n")
        np.savetxt('XAS.dat', np.array(tmp).T, fmt='%8.4f',
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
    a = thermal_average(es[:np.shape(gs)[0]], -gs.imag, T=T)
    if rank == 0:
        h5f.create_dataset('RIXS', data=-gs.imag)
        h5f.create_dataset('RIXSthermal', data=a)
    # Sum over transition operators
    aSum = np.sum(a, axis=(0,1))
    # Save spectra to disk
    if rank == 0:
        print("Save spectra to disk...\n")
        # I[wLoss,wIn], with wLoss on first column and wIn on first row.
        tmp = np.zeros((len(wLoss) + 1, len(wIn) + 1), dtype=np.float32)
        tmp[0,0] = len(wIn)
        tmp[0,1:] = wIn
        tmp[1:,0] = wLoss
        tmp[1:,1:] = aSum.T
        tmp.tofile('RIXS.bin')
    if rank == 0:
        print("time(RIXS) = {:.2f} seconds \n".format(time.time()-t0))
        t0 = time.time()

    if rank == 0: h5f.close()
    print('Script finished for rank:', rank)


def get_hamiltonian_operator(nBaths, nValBaths, slaterCondon, SOCs,
                             DCinfo, hField,
                             h0_filename):
    """
    Return the Hamiltonian, in operator form.

    Parameters
    ----------
    nBaths : dict
        Number of bath states for each angular momentum.
    nValBaths : dict
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
    h0_filename : str
        Filename of non-interacting, non-relativistic operator.

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
    n0imps, chargeTransferCorrection = DCinfo
    hx, hy, hz = hField

    # Calculate the U operator, in spherical harmonics basis.
    uOperator = finite.get2p3dSlaterCondonUop(Fdd=Fdd, Fpp=Fpp,
                                              Fpd=Fpd, Gpd=Gpd)
    # Add SOC, in spherical harmonics basis.
    SOC2pOperator = finite.getSOCop(xi_2p, l=1)
    SOC3dOperator = finite.getSOCop(xi_3d, l=2)

    # Double counting (DC) correction values.
    # MLFT DC
    dc = finite.dc_MLFT(n3d_i=n0imps[2], c=chargeTransferCorrection, Fdd=Fdd,
                        n2p_i=n0imps[1], Fpd=Fpd, Gpd=Gpd)
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

    # Read the non-relativistic non-interacting Hamiltonian operator from file.
    h0_operator = get_h0_operator(h0_filename, nBaths)

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


def get_h0_operator(h0_filename, nBaths):
    """
    Return h0 operator.

    Parameters
    ----------
    h0_filename : str
        Filename of non-interacting, non-relativistic operator.
    nBaths : dict
        Number of bath states for each angular momentum.

    Returns
    -------
    h0_operator : dict
        The non-relativistic non-interacting Hamiltonian in operator form.
        Hamiltonian describes 3d orbitals and bath orbitals.
        tuple : complex,
        where each tuple describes a process of two steps (annihilation and then creation).
        Each step is described by a tuple of the form:
        (spin_orb, 'c') or (spin_orb, 'a'),
        where spin_orb is a tuple of the form (l, s, m) or (l, b) or ((l_a, l_b), b).

    """
    with open(h0_filename, 'rb') as handle:
        h0_operator = pickle.loads(handle.read())
    # Sanity check
    for process in h0_operator.keys():
        for event in process:
            if len(event[0]) == 2:
                assert nBaths[event[0][0]] > event[0][1]
    return h0_operator


if __name__== "__main__":
    # Parse input parameters
    parser = argparse.ArgumentParser(description='Spectroscopy simulations')
    parser.add_argument('h0_filename', type=str,
                        help='Filename of non-interacting Hamiltonian.')
    parser.add_argument('radial_filename', type=str,
                        help='Filename of radial part of correlated orbitals.')
    parser.add_argument('--ls', type=int, nargs='+', default=[1, 2],
                        help='Angular momenta of correlated orbitals.')
    parser.add_argument('--nBaths', type=int, nargs='+', default=[0, 10],
                        help='Number of bath states, for each angular momentum.')
    parser.add_argument('--nValBaths', type=int, nargs='+', default=[0, 10],
                        help='Number of valence bath states, for each angular momentum.')
    parser.add_argument('--n0imps', type=int, nargs='+', default=[6, 8],
                        help='Initial impurity occupation, for each angular momentum.')
    parser.add_argument('--dnTols', type=int, nargs='+', default=[0, 2],
                        help=('Max devation from initial impurity occupation, '
                              'for each angular momentum.'))
    parser.add_argument('--dnValBaths', type=int, nargs='+', default=[0, 2],
                        help=('Max number of electrons to leave valence bath orbitals, '
                              'for each angular momentum.'))
    parser.add_argument('--dnConBaths', type=int, nargs='+', default=[0, 0],
                        help=('Max number of electrons to enter conduction bath orbitals, '
                              'for each angular momentum.'))
    parser.add_argument('--Fdd', type=float, nargs='+', default=[7.5, 0, 9.9, 0, 6.6],
                        help='Slater-Condon parameters Fdd. d-orbitals are assumed.')
    parser.add_argument('--Fpp', type=float, nargs='+', default=[0., 0., 0.],
                        help='Slater-Condon parameters Fpp. p-orbitals are assumed.')
    parser.add_argument('--Fpd', type=float, nargs='+', default=[8.9, 0, 6.8],
                        help='Slater-Condon parameters Fpd. p- and d-orbitals are assumed.')
    parser.add_argument('--Gpd', type=float, nargs='+', default=[0., 5., 0, 2.8],
                        help='Slater-Condon parameters Gpd. p- and d-orbitals are assumed.')
    parser.add_argument('--xi_2p', type=float, default=11.629,
                        help='SOC value for p-orbitals. p-orbitals are assumed.')
    parser.add_argument('--xi_3d', type=float, default=0.096,
                        help='SOC value for d-orbitals. d-orbitals are assumed.')
    parser.add_argument('--chargeTransferCorrection', type=float, default=1.5,
                        help='Double counting parameter.')
    parser.add_argument('--hField', type=float, nargs='+', default=[0, 0, 0.0001],
                        help='Magnetic field. (h_x, h_y, h_z)')
    parser.add_argument('--nPsiMax', type=int, default=5,
                        help='Maximum number of eigenstates to consider.')
    parser.add_argument('--nPrintSlaterWeights', type=int, default=3,
                        help='Printing parameter.')
    parser.add_argument('--tolPrintOccupation', type=float, default=0.5,
                        help='Printing parameter.')
    parser.add_argument('--T', type=float, default=300,
                        help='Temperature (Kelvin).')
    parser.add_argument('--energy_cut', type=float, default=10,
                        help='How many k_B*T above lowest eigenenergy to consider.')
    parser.add_argument('--delta', type=float, default=0.2,
                        help=('Smearing, half width half maximum (HWHM). '
                              'Due to short core-hole lifetime.'))
    parser.add_argument('--deltaRIXS', type=float, default=0.050,
                        help=('Smearing, half width half maximum (HWHM). '
                              'Due to finite lifetime of excited states.'))
    parser.add_argument('--deltaNIXS', type=float, default=0.100,
                        help=('Smearing, half width half maximum (HWHM). '
                              'Due to finite lifetime of excited states.'))

    args = parser.parse_args()

    # Sanity checks
    assert len(args.ls) == len(args.nBaths)
    assert len(args.ls) == len(args.nValBaths)
    for nBath, nValBath in zip(args.nBaths, args.nValBaths):
        assert nBath >= nValBath
    for ang, n0imp in zip(args.ls, args.n0imps):
        assert n0imp <= 2 * (2 * ang + 1)  # Full occupation
        assert n0imp >= 0
    assert len(args.Fdd) == 5
    assert len(args.Fpp) == 3
    assert len(args.Fpd) == 3
    assert len(args.Gpd) == 4
    assert len(args.hField) == 3

    main(h0_filename=args.h0_filename,
         radial_filename=args.radial_filename,
         ls=tuple(args.ls), nBaths=tuple(args.nBaths),
         nValBaths=tuple(args.nValBaths), n0imps=tuple(args.n0imps),
         dnTols=tuple(args.dnTols), dnValBaths=tuple(args.dnValBaths),
         dnConBaths=tuple(args.dnConBaths),
         Fdd=tuple(args.Fdd), Fpp=tuple(args.Fpp),
         Fpd=tuple(args.Fpd), Gpd=tuple(args.Gpd),
         xi_2p=args.xi_2p, xi_3d=args.xi_3d,
         chargeTransferCorrection=args.chargeTransferCorrection,
         hField=tuple(args.hField), nPsiMax=args.nPsiMax,
         nPrintSlaterWeights=args.nPrintSlaterWeights,
         tolPrintOccupation=args.tolPrintOccupation,
         T=args.T, energy_cut=args.energy_cut,
         delta=args.delta, deltaRIXS=args.deltaRIXS, deltaNIXS=args.deltaNIXS)

