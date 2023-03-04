"""
This module contains functions for calculating various spectra.
"""

import time
from math import sqrt

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from mpi4py import MPI
from scipy.special import sph_harm, spherical_jn

from impurityModel.ed.average import thermal_average

# Local imports
from impurityModel.ed.finite import (
    add,
    applyOp,
    c2i,
    daggerOp,
    expand_basis_and_hamiltonian,
    gauntC,
    get_job_tasks,
    get_tridiagonal_krylov_vectors,
    inner,
    norm2,
)

# MPI variables
comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size


def simulate_spectra(
    es,
    psis,
    hOp,
    T,
    w,
    delta,
    epsilons,
    wLoss,
    deltaNIXS,
    qsNIXS,
    liNIXS,
    ljNIXS,
    RiNIXS,
    RjNIXS,
    radialMesh,
    wIn,
    deltaRIXS,
    epsilonsRIXSin,
    epsilonsRIXSout,
    restrictions,
    h5f,
    nBaths,
):
    """
    Simulate various spectra.

    Parameters
    ----------
    es : tuple
        Eigen-energy (in eV).
    psis : tuple
        Many-body eigen-states.
    hOp : dict
        The Hamiltonian in operator form.
        tuple : complex,
        where each tuple describes a process of several steps.
        Each step is described by a tuple of the form: (i,'c') or (i,'a'),
        where i is a spin-orbital index.
    T : float
        Temperature (in Kelvin).
    w : ndarray
        Real-energy mesh (in eV).
    delta : float
        Distance above the real axis (in eV).
        Gives smearing to spectra.
    epsilons : list
        Each element is a XAS polarization vector.
    wLoss : ndarray
        Real-energy mesh (in eV).
        Incoming minus outgoing photon energy.
    deltaNIXS : float
        Distance above the real axis (in eV).
        Gives smearing to NIXS spectra.
    qsNIXS : list
        Various momenta used in NIXS.
    liNIXS : int
        Angular momentum of final orbitals in the NIXS excitation process.
    ljNIXS : int
        Angular momentum of initial orbitals in the NIXS excitation process.
    RiNIXS : ndarray
        Radial part of final correlated orbitals.
    RjNIXS : ndarray
        Radial part of initial correlated orbitals.
    radialMesh : ndarray
        Radial mesh, using in NIXS.
    wIn : ndarray
        Incoming photon energies in RIXS.
    deltaRIXS : float
        Distance above the real axis (in eV).
        Gives smearing to RIXS spectra.
    epsilonsRIXSin : list
        Polarization vectors of in-going photon.
    epsilonsRIXSout : list
        Polarization vectors of out-going photon.
    restrictions : dict
        Restriction the occupation of generated
        product states.
    h5f : h5py file-handle
        Will be used to write data to disk.
    nBaths : OrderedDict
        Angular momentum : number of bath states.

    """
    if rank == 0:
        t0 = time.time()

    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())

    if rank == 0:
        print("Create 3d inverse photoemission and photoemission spectra...")
    # Transition operators
    tOpsIPS = getInversePhotoEmissionOperators(nBaths, l=2)
    tOpsPS = getPhotoEmissionOperators(nBaths, l=2)
    if rank == 0:
        print("Inverse photoemission Green's function..")
    gsIPS = getSpectra(n_spin_orbitals, hOp, tOpsIPS, psis, es, w, delta, restrictions)
    if rank == 0:
        print("Photoemission Green's function..")
    gsPS = getSpectra(n_spin_orbitals, hOp, tOpsPS, psis, es, -w, -delta, restrictions)
    gsPS *= -1
    gs = gsPS + gsIPS
    if rank == 0:
        print("#eigenstates = {:d}".format(np.shape(gs)[0]))
        print("#spin orbitals = {:d}".format(np.shape(gs)[1]))
        print("#mesh points = {:d}".format(np.shape(gs)[2]))
    # Thermal average
    a = thermal_average(es[: np.shape(gs)[0]], -gs.imag, T=T)
    if rank == 0:
        h5f.create_dataset("PS", data=-gs.imag)
        h5f.create_dataset("PSthermal", data=a)
    # Sum over transition operators
    aSum = np.sum(a, axis=0)
    # Save spectra to disk
    if rank == 0:
        tmp = [w, aSum]
        # Each transition operator seperatly
        for i in range(np.shape(a)[0]):
            tmp.append(a[i, :])
        print("Save spectra to disk...\n")
        np.savetxt("PS.dat", np.array(tmp).T, fmt="%8.4f", header="E  sum  T1  T2  T3 ...")
    if rank == 0:
        print("time(PS) = {:.2f} seconds \n".format(time.time() - t0))
        t0 = time.time()

    if rank == 0:
        print("Create core 2p x-ray photoemission spectra (XPS) ...")
    # Transition operators
    tOpsPS = getPhotoEmissionOperators(nBaths, l=1)
    # Photoemission Green's function
    gs = getSpectra(n_spin_orbitals, hOp, tOpsPS, psis, es, -w, -delta, restrictions)
    gs *= -1
    if rank == 0:
        print("#eigenstates = {:d}".format(np.shape(gs)[0]))
        print("#spin orbitals = {:d}".format(np.shape(gs)[1]))
        print("#mesh points = {:d}".format(np.shape(gs)[2]))
    # Thermal average
    a = thermal_average(es[: np.shape(gs)[0]], -gs.imag, T=T)
    if rank == 0:
        h5f.create_dataset("XPS", data=-gs.imag)
        h5f.create_dataset("XPSthermal", data=a)
    # Sum over transition operators
    aSum = np.sum(a, axis=0)
    # Save spectra to disk
    if rank == 0:
        tmp = [w, aSum]
        # Each transition operator seperatly
        for i in range(np.shape(a)[0]):
            tmp.append(a[i, :])
        print("Save spectra to disk...\n")
        np.savetxt("XPS.dat", np.array(tmp).T, fmt="%8.4f", header="E  sum  T1  T2  T3 ...")
    if rank == 0:
        print("time(XPS) = {:.2f} seconds \n".format(time.time() - t0))
        t0 = time.time()

    if rank == 0:
        print("Create NIXS spectra...")
    # Transition operator: exp(iq*r)
    tOps = getNIXSOperators(nBaths, qsNIXS, liNIXS, ljNIXS, RiNIXS, RjNIXS, radialMesh)
    # Green's function
    gs = getSpectra(n_spin_orbitals, hOp, tOps, psis, es, wLoss, deltaNIXS, restrictions)
    if rank == 0:
        print("#eigenstates = {:d}".format(np.shape(gs)[0]))
        print("#q-points = {:d}".format(np.shape(gs)[1]))
        print("#mesh points = {:d}".format(np.shape(gs)[2]))
    # Thermal average
    a = thermal_average(es[: np.shape(gs)[0]], -gs.imag, T=T)
    if rank == 0:
        h5f.create_dataset("NIXS", data=-gs.imag)
        h5f.create_dataset("NIXSthermal", data=a)
    # Sum over q-points
    aSum = np.sum(a, axis=0)
    # Save spectra to disk
    if rank == 0:
        tmp = [wLoss, aSum]
        # Each q-point seperatly
        for i in range(np.shape(a)[0]):
            tmp.append(a[i, :])
        print("Save spectra to disk...\n")
        np.savetxt("NIXS.dat", np.array(tmp).T, fmt="%8.4f", header="E  sum  T1  T2  T3 ...")

    if rank == 0:
        print("time(NIXS) = {:.2f} seconds \n".format(time.time() - t0))
        t0 = time.time()

    if rank == 0:
        print("Create XAS spectra...")
    # Dipole transition operators
    tOps = getDipoleOperators(nBaths, epsilons)
    # Green's function
    gs = getSpectra(n_spin_orbitals, hOp, tOps, psis, es, w, delta, restrictions)
    if rank == 0:
        print("#eigenstates = {:d}".format(np.shape(gs)[0]))
        print("#polarizations = {:d}".format(np.shape(gs)[1]))
        print("#mesh points = {:d}".format(np.shape(gs)[2]))
    # Thermal average
    a = thermal_average(es[: np.shape(gs)[0]], -gs.imag, T=T)
    if rank == 0:
        h5f.create_dataset("XAS", data=-gs.imag)
        h5f.create_dataset("XASthermal", data=a)
    # Sum over transition operators
    aSum = np.sum(a, axis=0)
    # Save spectra to disk
    if rank == 0:
        tmp = [w, aSum]
        # Each transition operator seperatly
        for i in range(np.shape(a)[0]):
            tmp.append(a[i, :])
        print("Save spectra to disk...\n")
        np.savetxt("XAS.dat", np.array(tmp).T, fmt="%8.4f", header="E  sum  T1  T2  T3 ...")
    if rank == 0:
        print("time(XAS) = {:.2f} seconds \n".format(time.time() - t0))
        t0 = time.time()

    if rank == 0:
        print("Create RIXS spectra...")
    # Dipole 2p -> 3d transition operators
    tOpsIn = getDipoleOperators(nBaths, epsilonsRIXSin)
    # Dipole 3d -> 2p transition operators
    tOpsOut = getDaggeredDipoleOperators(nBaths, epsilonsRIXSout)
    # Green's function
    gs = getRIXSmap(
        n_spin_orbitals,
        hOp,
        tOpsIn,
        tOpsOut,
        psis,
        es,
        wIn,
        wLoss,
        delta,
        deltaRIXS,
        restrictions,
    )
    if rank == 0:
        print("#eigenstates = {:d}".format(np.shape(gs)[0]))
        print("#in-polarizations = {:d}".format(np.shape(gs)[1]))
        print("#out-polarizations = {:d}".format(np.shape(gs)[2]))
        print("#mesh points of input energy = {:d}".format(np.shape(gs)[3]))
        print("#mesh points of energy loss = {:d}".format(np.shape(gs)[4]))
    # Thermal average
    a = thermal_average(es[: np.shape(gs)[0]], -gs.imag, T=T)
    if rank == 0:
        h5f.create_dataset("RIXS", data=-gs.imag)
        h5f.create_dataset("RIXSthermal", data=a)
    # Sum over transition operators
    aSum = np.sum(a, axis=(0, 1))
    # Save spectra to disk
    if rank == 0:
        print("Save spectra to disk...\n")
        # I[wLoss,wIn], with wLoss on first column and wIn on first row.
        tmp = np.zeros((len(wLoss) + 1, len(wIn) + 1), dtype=np.float32)
        tmp[0, 0] = len(wIn)
        tmp[0, 1:] = wIn
        tmp[1:, 0] = wLoss
        tmp[1:, 1:] = aSum.T
        tmp.tofile("RIXS.bin")
    if rank == 0:
        print("time(RIXS) = {:.2f} seconds \n".format(time.time() - t0))
        t0 = time.time()

    if rank == 0:
        h5f.close()


def getDipoleOperators(nBaths, ns):
    r"""
    Return dipole transition operators.

    Transitions between states of different angular momentum,
    defined by the keys in the nBaths dictionary.

    Parameters
    ----------
    nBaths : dict
        angular momentum: number of bath states.
    ns : list
        Each element contains a polarization vector n = [nx,ny,nz]

    """
    tOps = []
    for n in ns:
        tOps.append(getDipoleOperator(nBaths, n))
    return tOps


def getDaggeredDipoleOperators(nBaths, ns):
    """
    Return daggered dipole transition operators.

    Parameters
    ----------
    nBaths : dict
        angular momentum: number of bath states.
    ns : list
        Each element contains a polarization vector n = [nx,ny,nz]

    """
    tDaggerOps = []
    for n in ns:
        tDaggerOps.append(daggerOp(getDipoleOperator(nBaths, n)))
    return tDaggerOps


def getDipoleOperator(nBaths, n):
    r"""
    Return dipole transition operator :math:`\hat{T}`.

    Transition between states of different angular momentum,
    defined by the keys in the nBaths dictionary.

    Parameters
    ----------
    nBaths : Ordered dict
        int : int,
        where the keys are angular momenta and values are number of bath states.
    n : list
        polarization vector n = [nx,ny,nz]

    """
    tOp = {}
    nDict = {
        -1: (n[0] + 1j * n[1]) / sqrt(2),
        0: n[2],
        1: (-n[0] + 1j * n[1]) / sqrt(2),
    }
    # Angular momentum
    l1, l2 = nBaths.keys()
    for m in range(-l2, l2 + 1):
        for mp in range(-l1, l1 + 1):
            for s in range(2):
                if abs(m - mp) <= 1:
                    # See Robert Eder's lecture notes:
                    # "Multiplets in Transition Metal Ions"
                    # in Julich school.
                    # tij = d*n*c1(l=2,m;l=1,mp),
                    # d - radial integral
                    # n - polarization vector
                    # c - Gaunt coefficient
                    tij = gauntC(k=1, l=l2, m=m, lp=l1, mp=mp, prec=16)
                    tij *= nDict[m - mp]
                    if tij != 0:
                        i = c2i(nBaths, (l2, s, m))
                        j = c2i(nBaths, (l1, s, mp))
                        tOp[((i, "c"), (j, "a"))] = tij
    return tOp


def getNIXSOperators(nBaths, qs, li, lj, Ri, Rj, r, kmin=1):
    r"""
    Return non-resonant inelastic x-ray scattering transition operators.

    :math:`\hat{T} = \sum_{i,j,\sigma} T_{i,j}
    \hat{c}_{i\sigma}^\dagger \hat{c}_{j\sigma}`,

    where
    :math:`T_{i,j} = \langle i | e^{i\mathbf{q}\cdot \mathbf{r}} | j \rangle`.
    The plane-wave is expanded in spherical harmonics.
    See PRL 99 257401 (2007) for more information.

    Parameters
    ----------
    nBaths : Ordered dict
        angular momentum: number of bath states.
    qs : list
        Each element contain a photon scattering vector q = [qx,qy,qz].
    li : int
        Angular momentum of the orbitals to excite into.
    lj : int
        Angular momentum of the orbitals to excite from.
    Ri : list
        Radial part of the orbital to excite into.
        Normalized such that the integral of Ri^2(r) * r^2
        should be equal to one.
    Rj : list
        Radial part of the orbital to excite from.
        Normalized such that the integral of Ri^2(r) * r^2
        should be equal to one.
    r : list
        Radial mesh points.
    kmin : int
        The lowest integer in the plane-wave expansion.
        By default kmin = 1, which means that the monopole contribution
        is not included.
        To include also the monopole scattering, set kmin = 0.

    """
    if rank == 0:
        if kmin == 0:
            print("Monopole contribution included in the expansion")
        elif kmin > 0:
            print("Monopole contribution not included in the expansion")
    tOps = []
    for q in qs:
        if rank == 0:
            print("q =", q)
        tOps.append(getNIXSOperator(nBaths, q, li, lj, Ri, Rj, r, kmin))
    return tOps


def getNIXSOperator(nBaths, q, li, lj, Ri, Rj, r, kmin=1):
    r"""
    Return non-resonant inelastic x-ray scattering transition
    operator :math:`\hat{T}`.

    :math:`\hat{T} = \sum_{i,j,\sigma} T_{i,j}
    \hat{c}_{i\sigma}^\dagger \hat{c}_{j\sigma}`,

    where
    :math:`T_{i,j} = \langle i | e^{i\mathbf{q}\cdot \mathbf{r}} | j \rangle`.
    The plane-wave is expanded in spherical harmonics.
    See PRL 99 257401 (2007) for more information.

    Parameters
    ----------
    nBaths : Ordered dict
        angular momentum: number of bath states.
    q : list
        Photon scattering vector q = [qx,qy,qz]
        The change in photon momentum.
    li : int
        Angular momentum of the orbitals to excite into.
    lj : int
        Angular momentum of the orbitals to excite from.
    Ri : list
        Radial part of the orbital to excite into.
        Normalized such that the integral of Ri^2(r) * r^2
        should be equal to one.
    Rj : list
        Radial part of the orbital to excite from.
        Normalized such that the integral of Ri^2(r) * r^2
        should be equal to one.
    r : list
        Radial mesh points.
    kmin : int
        The lowest integer in the plane-wave expansion.
        By default kmin = 1, which means that the monopole contribution
        is not included.
        To include also the monopole scattering, set kmin = 0.

    """
    # Convert scattering list to numpy array
    q = np.array(q)
    qNorm = np.linalg.norm(q)
    # Polar (colatitudinal) coordinate
    theta = np.arccos(q[2] / qNorm)
    # Azimuthal (longitudinal) coordinate
    phi = np.arccos(q[0] / (qNorm * np.sin(theta)))
    tOp = {}
    for k in range(kmin, abs(li + lj) + 1):
        if (li + lj + k) % 2 == 0:
            Rintegral = np.trapz(np.conj(Ri) * spherical_jn(k, qNorm * r) * Rj * r**2, r)
            if rank == 0:
                print("Rintegral(k=", k, ") =", Rintegral)
            for mi in range(-li, li + 1):
                for mj in range(-lj, lj + 1):
                    m = mi - mj
                    if abs(m) <= k:
                        tij = Rintegral
                        tij *= 1j ** (k) * sqrt(2 * k + 1)
                        tij *= np.conj(sph_harm(m, k, phi, theta))
                        tij *= gauntC(k, li, mi, lj, mj, prec=16)
                        if tij != 0:
                            for s in range(2):
                                i = c2i(nBaths, (li, s, mi))
                                j = c2i(nBaths, (lj, s, mj))
                                process = ((i, "c"), (j, "a"))
                                if process in tOp:
                                    tOp[((i, "c"), (j, "a"))] += tij
                                else:
                                    tOp[((i, "c"), (j, "a"))] = tij
    return tOp


def getInversePhotoEmissionOperators(nBaths, l=2):
    r"""
    Return inverse photo emission operators :math:`\{ c_i^\dagger \}`.

    Parameters
    ----------
    nBaths : OrderedDict
        Angular momentum: number of bath states.
    l : int
        Angular momentum.

    """
    # Transition operators
    tOpsIPS = []
    for s in range(2):
        for m in range(-l, l + 1):
            tOpsIPS.append({((c2i(nBaths, (l, s, m)), "c"),): 1})
    return tOpsIPS


def getPhotoEmissionOperators(nBaths, l=2):
    r"""
    Return photo emission operators :math:`\{ c_i \}`.

    Parameters
    ----------
    nBaths : OrderedDict
        Angular momentum: number of bath states.
    l : int
        Angular momentum.

    """
    # Transition operators
    tOpsPS = []
    for s in range(2):
        for m in range(-l, l + 1):
            tOpsPS.append({((c2i(nBaths, (l, s, m)), "a"),): 1})
    return tOpsPS


def getGreen(
    n_spin_orbitals,
    e,
    psi,
    hOp,
    omega,
    delta,
    krylovSize,
    slaterWeightMin,
    restrictions=None,
    h_dict=None,
    mode="sparse",
    parallelization_mode="serial",
):
    r"""
    return Green's function
    :math:`\langle psi|((omega+1j*delta+e)\hat{1} - hOp)^{-1} |psi \rangle`.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    e : float
        Total energy
    psi : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.
    hOp : dict
        Operator
    omega : list
        Real axis energy mesh
    delta : float
        Deviation from real axis.
        Broadening/resolution parameter.
    krylovSize : int
        Size of the Krylov space
    slaterWeightMin : float
        Restrict the number of product states by
        looking at `|amplitudes|^2`.
    restrictions : dict
        Restriction the occupation of generated
        product states.
    h_dict : dict
        Stores the result of the (Hamiltonian) operator hOp acting
        on individual product states. Information is stored according to:
        `|product state> : H|product state>`, where
        each product state is represented by an integer, and the result is
        a dictionary (of the format int : complex).
        If present, it may also be updated by this function.
    mode : str
        'dict', 'dense', 'sparse'
        Determines which algorithm to use.
        Option 'sparse' should be best.
    parallelization_mode : str
        Parallelization mode. Either: "serial" or "H_build".

    """
    # Allocation of output vector.
    g = np.zeros(len(omega), dtype=np.complex)
    # In the exceptional case of an empty state psi, return zero.
    if len(psi) == 0:
        return g
    # Initialization
    if h_dict is None:
        h_dict = {}
    if mode == "dict":
        assert parallelization_mode == "serial"
        v = list(np.zeros(krylovSize))
        w = list(np.zeros(krylovSize))
        wp = list(np.zeros(krylovSize))
        v[0] = psi
        # print('len(h_dict) = ',len(h_dict),', len(v[0]) = ',len(v[0]))
        wp[0] = applyOp(n_spin_orbitals, hOp, v[0], slaterWeightMin, restrictions, h_dict)
        # print('#len(h_dict) = ',len(h_dict),', len(wp[0]) = ',len(wp[0]))
        alpha = np.zeros(krylovSize, dtype=np.float)
        beta = np.zeros(krylovSize - 1, dtype=np.float)
        alpha[0] = inner(wp[0], v[0]).real
        w[0] = add(wp[0], v[0], -alpha[0])
        # Approximate position of spectrum.
        # print('alpha[0]-E_i = {:5.1f}'.format(alpha[0]-e))
        # Construct Krylov states,
        # and elements alpha and beta.
        for j in range(1, krylovSize):
            beta[j - 1] = sqrt(norm2(w[j - 1]))
            # print('beta[',j-1,'] = ',beta[j-1])
            if beta[j - 1] != 0:
                v[j] = {s: 1.0 / beta[j - 1] * a for s, a in w[j - 1].items()}
            else:
                # Pick normalized state v[j],
                # orthogonal to v[0],v[1],v[2],...,v[j-1]
                print("Warning: beta==0, implementation missing!")
            # print('len(v[',j,'] =',len(v[j]))
            wp[j] = applyOp(n_spin_orbitals, hOp, v[j], slaterWeightMin, restrictions, h_dict)
            alpha[j] = inner(wp[j], v[j]).real
            w[j] = add(add(wp[j], v[j], -alpha[j]), v[j - 1], -beta[j - 1])
            # print('len(h_dict) = ',len(h_dict),', len(w[j]) = ',len(w[j]))
    elif mode == "sparse" or mode == "dense":
        # If we use a parallelized mode, we want to work with
        # only the MPI local part of the Hamiltonian matrix h.
        h_local = parallelization_mode == "H_build"
        # Obtain Hamiltonian in matrix format.
        # Possibly also add new product state keys to h_dict.
        # If h_local equals to True, the returning sparse matrix
        # Hamiltonian will not contain all column in each MPI rank.
        # Instead all matrix columns are distributed over all the MPI ranks.
        h, basis_index = expand_basis_and_hamiltonian(
            n_spin_orbitals,
            h_dict,
            hOp,
            psi.keys(),
            restrictions,
            parallelization_mode,
            h_local,
        )
        # Number of basis states
        n = len(basis_index)
        # Express psi as a vector
        psi0 = np.zeros(n, dtype=np.complex)
        for ps, amp in psi.items():
            psi0[basis_index[ps]] = amp
        # Unnecessary (and impossible) to find more than n Krylov basis vectors.
        krylovSize = min(krylovSize, n)
        # Get tridiagonal elements of the Krylov Hamiltonian matrix.
        alpha, beta = get_tridiagonal_krylov_vectors(h, psi0, krylovSize, h_local, mode)
    else:
        raise Exception("Value of variable 'mode' is incorrect.")
    # Construct Green's function from continued fraction.
    omegaP = omega + 1j * delta + e
    for i in range(krylovSize - 1, -1, -1):
        g = 1.0 / (omegaP - alpha[i]) if i == krylovSize - 1 else 1.0 / (omegaP - alpha[i] - beta[i] ** 2 * g)
    return g


def getSpectra(
    n_spin_orbitals,
    hOp,
    tOps,
    psis,
    es,
    w,
    delta,
    restrictions=None,
    krylovSize=150,
    slaterWeightMin=1e-7,
    parallelization_mode="H_build",
):
    r"""
    Return Green's function for states with low enough energy.

    For states :math:`|psi \rangle`, calculate:

    :math:`g(w+1j*delta) =
    = \langle psi| tOp^\dagger ((w+1j*delta+e)*\hat{1} - hOp)^{-1} tOp
    |psi \rangle`,

    where :math:`e = \langle psi| hOp |psi \rangle`

    Lanczos algorithm is used.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    hOp : dict
        Operator
    tOps : list
        List of dict operators
    psis : list
        List of Multi state dictionaries
    es : list
        Total energies
    w : list
        Real axis energy mesh
    delta : float
        Deviation from real axis.
        Broadening/resolution parameter.
    restrictions : dict
        Restriction the occupation of generated
        product states.
    krylovSize : int
        Size of the Krylov space
    slaterWeightMin : float
        Restrict the number of product states by
        looking at `|amplitudes|^2`.
    parallelization_mode : str
            "eigen_states" or "H_build".

    """
    n = len(es)
    # Green's functions
    gs = np.zeros((n, len(tOps), len(w)), dtype=np.complex)
    # Hamiltonian dict of the form  |PS> : {H|PS>}
    # New elements are added each time getGreen is called.
    # Also acts as an input to getGreen and speed things up dramatically.
    h = {}
    if parallelization_mode == "eigen_states":
        g = {}
        # Loop over eigen states, unique for each MPI rank
        for i in get_job_tasks(rank, ranks, range(n)):
            psi = psis[i]
            e = es[i]
            # Initialize Green's functions
            g[i] = np.zeros((len(tOps), len(w)), dtype=np.complex)
            # Loop over transition operators
            for t, tOp in enumerate(tOps):
                psiR = applyOp(n_spin_orbitals, tOp, psi, slaterWeightMin, restrictions)
                normalization = sqrt(norm2(psiR))
                for state in psiR.keys():
                    psiR[state] /= normalization
                g[i][t, :] = normalization**2 * getGreen(
                    n_spin_orbitals,
                    e,
                    psiR,
                    hOp,
                    w,
                    delta,
                    krylovSize,
                    slaterWeightMin,
                    restrictions,
                    h,
                    parallelization_mode="serial",
                )
        # Distribute the Green's functions among the ranks
        for r in range(ranks):
            gTmp = comm.bcast(g, root=r)
            for i, gValue in gTmp.items():
                gs[i, :, :] = gValue
    elif parallelization_mode == "H_build":
        # Loop over transition operators
        for t, tOp in enumerate(tOps):
            t_big = {}
            # Loop over eigen states
            for i in range(n):
                psi = psis[i]
                e = es[i]
                psiR = applyOp(n_spin_orbitals, tOp, psi, slaterWeightMin, restrictions, t_big)
                # if rank == 0: print("len(t_big) = {:d}".format(len(t_big)))
                normalization = sqrt(norm2(psiR))
                for state in psiR.keys():
                    psiR[state] /= normalization
                gs[i, t, :] = normalization**2 * getGreen(
                    n_spin_orbitals,
                    e,
                    psiR,
                    hOp,
                    w,
                    delta,
                    krylovSize,
                    slaterWeightMin,
                    restrictions,
                    h,
                    parallelization_mode=parallelization_mode,
                )
    else:
        raise Exception("Incorrect value of variable parallelization_mode.")
    return gs


def getRIXSmap(
    n_spin_orbitals,
    hOp,
    tOpsIn,
    tOpsOut,
    psis,
    es,
    wIns,
    wLoss,
    delta1,
    delta2,
    restrictions=None,
    krylovSize=150,
    slaterWeightMin=1e-7,
    h_dict_ground=None,
    parallelization_mode="H_build_wIn",
):
    r"""
    Return RIXS Green's function for states.

    For states :math:`|psi \rangle`, calculate:

    :math:`g(w+1j*delta)
    = \langle psi| ROp^\dagger ((wLoss+1j*delta2+e)*\hat{1} - hOp)^{-1} ROp
    |psi \rangle`,

    where :math:`e = \langle psi| hOp |psi \rangle`, and

    :math:`Rop = tOpOut ((wIns+1j*delta1+e)*\hat{1} - hOp)^{-1} tOpIn`.

    Calculations are performed according to:

    1) Calculate state `|psi1> = tOpIn |psi>`.
    2) Calculate state `|psi2> = ((wIns+1j*delta1+e)*\hat{1} - hOp)^{-1}|psi1>`
        This is done by introducing operator:
        `A = (wIns+1j*delta1+e)*\hat{1} - hOp`.
        By applying A from the left on `|psi2> = A^{-1}|psi1>` gives
        the inverse problem: `A|psi2> = |psi1>`.
        This equation can be solved by guessing `|psi2>` and iteratively
        improving it.
    3) Calculate state `|psi3> = tOpOut |psi2>`
    4) Calculate `normalization = sqrt(<psi3|psi3>)`
    5) Normalize psi3 according to: `psi3 /= normalization`
    6) Now the Green's function is given by:
        :math:`g(wLoss+1j*delta2) = normalization^2
        * \langle psi3| ((wLoss+1j*delta2+e)*\hat{1} - hOp)^{-1} |psi3 \rangle`,
        which can efficiently be evaluation using Lanczos.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    hOp : dict
        Operator
    tOpsIn : list
        List of dict operators, describing core-hole excitation.
    tOpsOut : list
        List of dict operators, describing filling of the core-hole.
    psis : list
        List of Multi state dictionaries
    es : list
        Total energies
    wIns : list
        Real axis energy mesh for incoming photon energy
    wLoss : list
        Real axis energy mesh for photon energy loss, i.e.
        wLoss = wIns - wOut
    delta1 : float
        Deviation from real axis.
        Broadening/resolution parameter.
    delta2 : float
        Deviation from real axis.
        Broadening/resolution parameter.
    restrictions : dict
        Restriction the occupation of generated
        product states.
    krylovSize : int
        Size of the Krylov space
    slaterWeightMin : float
        Restrict the number of product states by
        looking at `|amplitudes|^2`.
    h_dict_ground : dict
        Stores the result of the (Hamiltonian) operator hOp acting
        on individual product states. Information is stored according to:
        `|product state> : H|product state>`, where
        each product state is represented by an integer, and the result is
        a dictionary (of the format int : complex).
        Only product states without a core hole are stored in this variable.
        If present, it may also be updated by this function.
    parallelization_mode : str
        "serial", "H_build", "wIn" or "H_build_wIn"

    """
    if h_dict_ground is None:
        h_dict_ground = {}
    nE = len(es)
    # Green's functions
    gs = np.zeros((nE, len(tOpsIn), len(tOpsOut), len(wIns), len(wLoss)), dtype=np.complex)
    # Hamiltonian dict of the form  |PS> : {H|PS>}
    # For product states with a core hole.
    h_dict_excited = {}
    tOut_big = [{}] * len(tOpsOut)
    if parallelization_mode == "serial" or parallelization_mode == "H_build":
        # Loop over in-coming transition operators
        for tIn, tOpIn in enumerate(tOpsIn):
            tIn_big = {}
            # Loop over eigen states
            for iE in range(nE):
                psi = psis[iE]
                e = es[iE]
                # Core-hole state
                psi1 = applyOp(n_spin_orbitals, tOpIn, psi, slaterWeightMin, restrictions, tIn_big)
                # Hamiltonian acting on relevant product states. |PS> : {H|PS>}
                n_tmp = len(h_dict_excited)
                if rank == 0:
                    print("Construct H for core-hole excited system.")
                h, basis_index = expand_basis_and_hamiltonian(
                    n_spin_orbitals,
                    h_dict_excited,
                    hOp,
                    psi1.keys(),
                    restrictions,
                    parallelization_mode,
                    return_h_local=False,
                )
                if rank == 0:
                    print(
                        "#elements added to local h_dict_excited: ",
                        len(h_dict_excited) - n_tmp,
                    )
                n = len(basis_index)
                # Express psi1 as a vector
                y = np.zeros(n, dtype=np.complex)
                for ps, amp in psi1.items():
                    y[basis_index[ps]] = amp
                # If one would like to store psi1 as a sparse vector
                # y = scipy.sparse.csr_matrix(y)

                # Fast look-up of product states
                basis_state = {index: ps for ps, index in basis_index.items()}
                if rank == 0:
                    print("Loop over in-coming photon energies...")
                for iwIn, wIn in enumerate(wIns):
                    # A = (wIn+1j*delta1+e)*\hat{1} - hOp.
                    a = scipy.sparse.csr_matrix(
                        ([wIn + 1j * delta1 + e] * n, (range(n), range(n))),
                        shape=(n, n),
                    )
                    a -= h
                    # Find x by solving: a*x = y
                    # Biconjugate gradient stabilized method.
                    # Pure conjugate gradient does not apply since
                    # it requires a Hermitian matrix.
                    x, info = scipy.sparse.linalg.bicgstab(a, y)
                    if info > 0:
                        print("Rank ", rank, ": Convergence to tolerance not achieved")
                        print("#iterations = ", info)
                    elif info < 0:
                        print(
                            "Rank ",
                            rank,
                            "illegal input or breakdown" + " in conjugate gradient",
                        )
                    # Convert multi state from vector to dict format
                    psi2 = {}
                    for i, amp in enumerate(x):
                        if amp != 0:
                            psi2[basis_state[i]] = amp

                    # Loop over out-going transition operators
                    for tOut, tOpOut in enumerate(tOpsOut):
                        # Calculate state |psi3> = tOpOut |psi2>
                        # This state has no core-hole.
                        psi3 = applyOp(
                            n_spin_orbitals,
                            tOpOut,
                            psi2,
                            slaterWeightMin,
                            restrictions,
                            tOut_big[tOut],
                        )
                        # Normalization factor
                        normalization = sqrt(norm2(psi3))
                        for state in psi3.keys():
                            psi3[state] /= normalization
                        # Remove product states with small weight
                        for state, amp in list(psi3.items()):
                            if abs(amp) ** 2 < slaterWeightMin:
                                psi3.pop(state)
                        # Calculate Green's function
                        gs[iE, tIn, tOut, iwIn, :] = normalization**2 * getGreen(
                            n_spin_orbitals,
                            e,
                            psi3,
                            hOp,
                            wLoss,
                            delta2,
                            krylovSize,
                            slaterWeightMin,
                            restrictions,
                            h_dict_ground,
                            parallelization_mode=parallelization_mode,
                        )
    elif parallelization_mode == "wIn" or parallelization_mode == "H_build_wIn":
        # Loop over in-coming transition operators
        for tIn, tOpIn in enumerate(tOpsIn):
            tIn_big = {}
            # Loop over eigen states
            for iE in range(nE):
                psi = psis[iE]
                e = es[iE]
                # Core-hole state
                psi1 = applyOp(n_spin_orbitals, tOpIn, psi, slaterWeightMin, restrictions, tIn_big)
                # Hamiltonian acting on relevant product states. |PS> : {H|PS>}
                n_tmp = len(h_dict_excited)
                if rank == 0:
                    print("Construct H for core-hole excited system.")
                if parallelization_mode == "wIn":
                    h, basis_index = expand_basis_and_hamiltonian(
                        n_spin_orbitals,
                        h_dict_excited,
                        hOp,
                        psi1.keys(),
                        restrictions,
                        parallelization_mode="serial",
                        return_h_local=False,
                    )
                elif parallelization_mode == "H_build_wIn":
                    h, basis_index = expand_basis_and_hamiltonian(
                        n_spin_orbitals,
                        h_dict_excited,
                        hOp,
                        psi1.keys(),
                        restrictions,
                        parallelization_mode="H_build",
                        return_h_local=False,
                    )
                if rank == 0:
                    print(
                        "#elements added to local h_dict_excited: ",
                        len(h_dict_excited) - n_tmp,
                    )
                n = len(basis_index)
                # Express psi1 as a vector
                y = np.zeros(n, dtype=np.complex)
                for ps, amp in psi1.items():
                    y[basis_index[ps]] = amp
                # If one would like to store psi1 as a sparse vector
                # y = scipy.sparse.csr_matrix(y)

                # Fast look-up of product states
                basis_state = {index: ps for ps, index in basis_index.items()}
                # Rank dependent variable
                g = {}
                if rank == 0:
                    print("Loop over in-coming photon energies...")
                # Loop over in-coming photon energies, unique for each MPI rank
                for iwIn in get_job_tasks(rank, ranks, range(len(wIns))):
                    wIn = wIns[iwIn]
                    # Initialize Green's functions
                    g[iwIn] = np.zeros((len(tOpsOut), len(wLoss)), dtype=np.complex)
                    # A = (wIn+1j*delta1+e)*\hat{1} - hOp.
                    a = scipy.sparse.csr_matrix(
                        ([wIn + 1j * delta1 + e] * n, (range(n), range(n))),
                        shape=(n, n),
                    )
                    a -= h
                    # Find x by solving: a*x = y
                    # Biconjugate gradient stabilized method.
                    # Pure conjugate gradient does not apply since
                    # it requires a Hermitian matrix.
                    x, info = scipy.sparse.linalg.bicgstab(a, y)
                    if info > 0:
                        print("convergence to tolerance not achieved")
                        print("#iterations = ", info)
                    elif info < 0:
                        print("illegal input or breakdown " + "in conjugate gradient")
                    # Convert multi state from vector to dict format
                    psi2 = {}
                    for i, amp in enumerate(x):
                        if amp != 0:
                            psi2[basis_state[i]] = amp
                    # Loop over out-going transition operators
                    for tOut, tOpOut in enumerate(tOpsOut):
                        # Calculate state |psi3> = tOpOut |psi2>
                        # This state has no core-hole.
                        psi3 = applyOp(
                            n_spin_orbitals,
                            tOpOut,
                            psi2,
                            slaterWeightMin,
                            restrictions,
                            tOut_big[tOut],
                        )
                        # Normalization factor
                        normalization = sqrt(norm2(psi3))
                        for state in psi3.keys():
                            psi3[state] /= normalization
                        # Remove product states with small weight
                        for state, amp in list(psi3.items()):
                            if abs(amp) ** 2 < slaterWeightMin:
                                psi3.pop(state)
                        # Calculate Green's function
                        g[iwIn][tOut, :] = normalization**2 * getGreen(
                            n_spin_orbitals,
                            e,
                            psi3,
                            hOp,
                            wLoss,
                            delta2,
                            krylovSize,
                            slaterWeightMin,
                            restrictions,
                            h_dict_ground,
                            parallelization_mode="serial",
                        )
                # Distribute the Green's functions among the ranks
                for r in range(ranks):
                    gTmp = comm.bcast(g, root=r)
                    for iwIn, gValue in gTmp.items():
                        gs[iE, tIn, :, iwIn, :] = gValue
    return gs
