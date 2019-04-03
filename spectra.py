#!/usr/bin/env python3

"""

spectra
=======

This module contains functions for calculating various spectra.

"""

from math import sqrt
import numpy as np
from mpi4py import MPI
import scipy.sparse
import scipy.sparse.linalg
from scipy.special import spherical_jn, sph_harm
import time

from .finite import gauntC, c2i, get_job_tasks
from .finite import daggerOp, applyOp, inner, add, norm2
from .finite import expand_basis_and_hamiltonian

# MPI variables
comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size


def getDipoleOperators(nBaths,ns):
    r'''
    Return dipole transition operators.

    Transitions between states of different angular momentum,
    defined by the keys in the nBaths dictionary.

    Parameters
    ----------
    nBaths : dict
        angular momentum: number of bath sets
    ns : list
        Each element contains a polarization vector n = [nx,ny,nz]

    '''
    tOps = []
    for n in ns:
        tOps.append(getDipoleOperator(nBaths,n))
    return tOps


def getDaggeredDipoleOperators(nBaths,ns):
    '''
    Return daggered dipole transition operators.

    Parameters
    ----------
    nBaths : dict
        angular momentum: number of bath sets
    ns : list
        Each element contains a polarization vector n = [nx,ny,nz]

    '''
    tDaggerOps = []
    for n in ns:
        tDaggerOps.append(daggerOp(getDipoleOperator(nBaths,n)))
    return tDaggerOps


def getDipoleOperator(nBaths,n):
    r'''
    Return dipole transition operator :math:`\hat{T}`.

    Transition between states of different angular momentum,
    defined by the keys in the nBaths dictionary.

    Parameters
    ----------
    nBaths : Ordered dict
        int : int,
        where the keys are angular momenta and values are number of bath sets.
    n : list
        polarization vector n = [nx,ny,nz]

    '''
    tOp = {}
    nDict = {-1:(n[0]+1j*n[1])/sqrt(2),0:n[2],1:(-n[0]+1j*n[1])/sqrt(2)}
    # Angular momentum
    l1,l2 = nBaths.keys()
    for m in range(-l2,l2+1):
        for mp in range(-l1,l1+1):
            for s in range(2):
                if abs(m-mp) <= 1:
                    # See Robert Eder's lecture notes:
                    # "Multiplets in Transition Metal Ions"
                    # in Julich school.
                    # tij = d*n*c1(l=2,m;l=1,mp),
                    # d - radial integral
                    # n - polarization vector
                    # c - Gaunt coefficient
                    tij = gauntC(k=1,l=l2,m=m,lp=l1,mp=mp,prec=16)
                    tij *= nDict[m-mp]
                    if tij != 0:
                        i = c2i(nBaths,(l2,m,s))
                        j = c2i(nBaths,(l1,mp,s))
                        tOp[((i,'c'),(j,'a'))] = tij
    return tOp


def getNIXSOperators(nBaths,qs,li,lj,Ri,Rj,r,kmin=1):
    r'''
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
        angular momentum: number of bath sets
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

    '''
    if rank == 0:
        if kmin == 0:
            print('Monopole contribution included in the expansion')
        elif kmin > 0:
            print('Monopole contribution not included in the expansion')
    tOps = []
    for q in qs:
        if rank == 0: print('q =',q)
        tOps.append(getNIXSOperator(nBaths,q,li,lj,Ri,Rj,r,kmin))
    return tOps


def getNIXSOperator(nBaths,q,li,lj,Ri,Rj,r,kmin=1):
    r'''
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
        angular momentum: number of bath sets
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

    '''
    # Convert scattering list to numpy array
    q = np.array(q)
    qNorm = np.linalg.norm(q)
    # Polar (colatitudinal) coordinate
    theta = np.arccos(q[2]/qNorm)
    # Azimuthal (longitudinal) coordinate
    phi = np.arccos(q[0]/(qNorm*np.sin(theta)))
    tOp = {}
    for k in range(kmin,abs(li+lj)+1):
        if (li+lj+k) % 2 == 0:
            Rintegral = np.trapz(np.conj(Ri)*spherical_jn(k,qNorm*r)*Rj*r**2,r)
            if rank == 0: print('Rintegral(k=',k,') =',Rintegral)
            for mi in range(-li,li+1):
                for mj in range(-lj,lj+1):
                    m = mi-mj
                    if abs(m) <= k:
                        tij = Rintegral
                        tij *= 1j**(k)*sqrt(2*k+1)
                        tij *= np.conj(sph_harm(m,k,phi,theta))
                        tij *= gauntC(k,li,mi,lj,mj,prec=16)
                        if tij != 0:
                            for s in range(2):
                                i = c2i(nBaths,(li,mi,s))
                                j = c2i(nBaths,(lj,mj,s))
                                process = ((i,'c'),(j,'a'))
                                if process in tOp:
                                    tOp[((i,'c'),(j,'a'))] += tij
                                else:
                                    tOp[((i,'c'),(j,'a'))] = tij
    return tOp


def getInversePhotoEmissionOperators(nBaths,l=2):
    r'''
    Return inverse photo emission operators :math:`\{ c_i^\dagger \}`.

    Parameters
    ----------
    nBaths : dict
        Angular momentum: number of bath sets
    l : int
        Angular momentum.

    '''
    # Transition operators
    tOpsIPS = []
    for m in range(-l,l+1):
        for s in range(2):
            tOpsIPS.append({((c2i(nBaths,(l,m,s)),'c'),):1})
    return tOpsIPS


def getPhotoEmissionOperators(nBaths,l=2):
    r'''
    Return photo emission operators :math:`\{ c_i \}`.

    Parameters
    ----------
    nBaths : dict
        Angular momentum: number of bath sets
    l : int
        Angular momentum.

    '''
    # Transition operators
    tOpsPS = []
    for m in range(-l,l+1):
        for s in range(2):
            tOpsPS.append({((c2i(nBaths,(l,m,s)),'a'),):1})
    return tOpsPS


def getGreen(n_spin_orbitals, e, psi, hOp, omega, delta, krylovSize,
             slaterWeightMin, restrictions=None, h_big=None, mode="sparse",
             parallelization_mode="serial"):
    r'''
    return Green's function
    :math:`\langle psi|((omega+1j*delta+e)\hat{1} - hOp)^{-1} |psi \rangle`.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    e : float
        Total energy
    psi : dict
        Multi state
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
    h_big : dict
        In and output argument.
        If present, the results of the operator hOp acting on each
        product state in the state psi is added and stored in this
        variable. Format: `|PS> : H|PS>`,
        where `|PS>` is a product state and `H|PS>` is stored as a dictionary.
    mode : str
        'dict', 'dense', 'sparse'
        Determines which algorithm to use.
        Option 'sparse' should be best.
    parallelization_mode : str
        Parallelization mode. Either: "serial" or "H_build".

    '''

    # Allocations
    g = np.zeros(len(omega),dtype=np.complex)
    # In the exceptional case of an empty state psi, return zero.
    if len(psi) == 0: return g
    # Initialization
    if h_big is None:
        h_big = {}
    if mode == 'dict':
        assert parallelization_mode == "serial"
        v = list(np.zeros(krylovSize))
        w = list(np.zeros(krylovSize))
        wp = list(np.zeros(krylovSize))
        v[0] = psi
        #print('len(h_big) = ',len(h_big),', len(v[0]) = ',len(v[0]))
        wp[0] = applyOp(n_spin_orbitals, hOp, v[0], slaterWeightMin,
                        restrictions, h_big)
        #print('#len(h_big) = ',len(h_big),', len(wp[0]) = ',len(wp[0]))
        alpha = np.zeros(krylovSize,dtype=np.float)
        beta = np.zeros(krylovSize-1,dtype=np.float)
        alpha[0] = inner(wp[0],v[0]).real
        w[0] = add(wp[0],v[0],-alpha[0])
        #print('len(w[0]) = ',len(w[0]))

        # Approximate position of spectrum
        #print('alpha[0]-E_i = {:5.1f}'.format(alpha[0]-e))

        # Construct Krylov states,
        # and elements alpha and beta
        for j in range(1,krylovSize):
            beta[j-1] = sqrt(norm2(w[j-1]))
            #print('beta[',j-1,'] = ',beta[j-1])
            if beta[j-1] != 0:
                v[j] = {s:1./beta[j-1]*a for s,a in w[j-1].items()}
            else:
                # Pick normalized state v[j],
                # orthogonal to v[0],v[1],v[2],...,v[j-1]
                print('Warning: beta==0, implementation missing!')
            #print('len(v[',j,'] =',len(v[j]))
            wp[j] = applyOp(n_spin_orbitals, hOp, v[j], slaterWeightMin,
                            restrictions,h_big)
            alpha[j] = inner(wp[j],v[j]).real
            w[j] = add(add(wp[j],v[j],-alpha[j]),v[j-1],-beta[j-1])
            #print('len(h_big) = ',len(h_big),', len(w[j]) = ',len(w[j]))
    elif mode == "sparse" or mode == "dense":
        # Obtain Hamiltonian in matrix format.
        # Possibly add new product state keys to h_big.
        h, basis_index = expand_basis_and_hamiltonian(
            n_spin_orbitals, h_big, hOp, psi.keys(), restrictions,
            parallelization_mode)
        if rank == 0: print(("Hamiltonian basis sizes: len(basis_index) = {:d},"
                             + " len(h_big) = {:d}").format(len(basis_index),
                                                            len(h_big)))
        # Number of basis states
        n = len(basis_index)
        # Unnecessary (and impossible) to find more than n Krylov basis vectors.
        krylovSize = min(krylovSize,n)
        if mode == "dense":
            h = h.toarray()
        # Allocate tri-diagonal matrix elements
        alpha = np.zeros(krylovSize,dtype=np.float)
        beta = np.zeros(krylovSize-1,dtype=np.float)
        # Allocate space for Krylov state vectors.
        # Do not save all Krylov vectors to save memory.
        v = np.zeros((2,n), dtype=np.complex)
        w = np.zeros(n, dtype=np.complex)
        wp = np.zeros(n, dtype=np.complex)
        # Express psi as a vector
        for ps, amp in psi.items():
            v[0,basis_index[ps]] = amp
        wp = h.dot(v[0,:])
        alpha[0] = np.dot(np.conj(wp),v[0,:]).real
        w = wp - alpha[0]*v[0,:]
        # Construct Krylov states,
        # and more importantly the vectors alpha and beta
        for j in range(1,krylovSize):
            beta[j-1] = sqrt(np.sum(np.abs(w)**2))
            if beta[j-1] != 0:
                v[1,:] = w/beta[j-1]
            else:
                # Pick normalized state v[j],
                # orthogonal to v[0],v[1],v[2],...,v[j-1]
                raise ValueError('Warning: beta==0, implementation missing!')
            wp = h.dot(v[1,:])
            alpha[j] = np.dot(np.conj(wp),v[1,:]).real
            w = wp - alpha[j]*v[1,:] - beta[j-1]*v[0,:]
            v[0,:] = v[1,:]
    else:
        sys.exit("Value of variable 'mode' is incorrect.")
    # Construct Green's function from
    # continued fraction
    omegaP = omega+1j*delta+e
    for i in range(krylovSize-1,-1,-1):
        if i == krylovSize-1:
            g = 1./(omegaP - alpha[i])
        else:
            g = 1./(omegaP-alpha[i]-beta[i]**2*g)
    return g


def getSpectra(n_spin_orbitals, hOp, tOps, psis, es, w, delta,
               restrictions=None, krylovSize=150, slaterWeightMin=1e-7,
               parallelization_mode="H_build"):
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
    gs = np.zeros((n,len(tOps),len(w)),dtype=np.complex)
    # Hamiltonian dict of the form  |PS> : {H|PS>}
    # New elements are added each time getGreen is called.
    # Also acts as an input to getGreen and speed things up dramatically.
    h = {}
    if parallelization_mode == "eigen_states":
        g = {}
        # Loop over eigen states, unique for each MPI rank
        for i in get_job_tasks(rank, ranks, range(n)):
            psi =  psis[i]
            e = es[i]
            # Initialize Green's functions
            g[i] = np.zeros((len(tOps),len(w)), dtype=np.complex)
            # Loop over transition operators
            for t, tOp in enumerate(tOps):
                psiR = applyOp(n_spin_orbitals, tOp, psi, slaterWeightMin,
                               restrictions)
                normalization = sqrt(norm2(psiR))
                for state in psiR.keys():
                    psiR[state] /= normalization
                g[i][t,:] = normalization**2*getGreen(
                    n_spin_orbitals, e, psiR, hOp, w, delta, krylovSize,
                    slaterWeightMin, restrictions, h,
                    parallelization_mode="serial")
        # Distribute the Green's functions among the ranks
        for r in range(ranks):
            gTmp = comm.bcast(g, root=r)
            for i,gValue in gTmp.items():
                gs[i,:,:] = gValue
    elif parallelization_mode == "H_build":
        # Loop over transition operators
        for t, tOp in enumerate(tOps):
            t_big = {}
            # Loop over eigen states
            for i in range(n):
                psi =  psis[i]
                e = es[i]
                psiR = applyOp(n_spin_orbitals, tOp, psi, slaterWeightMin,
                               restrictions, t_big)
                if rank == 0: print("len(t_big) = {:d}".format(len(t_big)))
                normalization = sqrt(norm2(psiR))
                for state in psiR.keys():
                    psiR[state] /= normalization
                gs[i,t,:] = normalization**2*getGreen(
                    n_spin_orbitals, e, psiR, hOp, w, delta, krylovSize,
                    slaterWeightMin, restrictions, h,
                    parallelization_mode=parallelization_mode)
    else:
        sys.error("Incorrect value of variable parallelization_mode.")
    return gs


def getRIXSmap(n_spin_orbitals, hOp, tOpsIn, tOpsOut, psis, es, wIns, wLoss,
               delta1, delta2, restrictions=None, krylovSize=150,
               slaterWeightMin=1e-7, hGround=None,
               parallelization_mode='H_build_wIn'):
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
    hGround : dict
        Optional
        tuple : dict, where dict is of the format tuple : float or complex
        The Hamiltonian for product states with no core hole.
        If present, it will be updated by this function.
    parallelization_mode : str
        "serial", "H_build", "wIn" or "H_build_wIn"

    """
    if hGround is None:
        hGround = {}
    nE = len(es)
    # Green's functions
    gs = np.zeros((nE,len(tOpsIn),len(tOpsOut),len(wIns),len(wLoss)),
                  dtype=np.complex)
    # Hamiltonian dict of the form  |PS> : {H|PS>}
    # For product states with a core hole.
    hExcited = {}
    tOut_big = [{}]*len(tOpsOut)
    if parallelization_mode == 'serial' or parallelization_mode == "H_build":
        # Loop over in-coming transition operators
        for tIn,tOpIn in enumerate(tOpsIn):
            tIn_big = {}
            # Loop over eigen states
            for iE in range(nE):
                psi =  psis[iE]
                e = es[iE]
                # Core-hole state
                psi1 = applyOp(n_spin_orbitals, tOpIn, psi, slaterWeightMin,
                               restrictions, tIn_big)
                # Hamiltonian acting on relevant product states. |PS> : {H|PS>}
                nTMP = len(hExcited)
                h, basis_index = expand_basis_and_hamiltonian(
                    n_spin_orbitals, hExcited, hOp, psi1.keys(), restrictions,
                    parallelization_mode)
                if rank == 0:
                    print("len(psi1), len(basis_index), len(hExcited), "
                          + "#elements added to hExcited: ",
                          len(psi1), len(basis_index), len(hExcited),
                          len(hExcited)-nTMP)
                n = len(basis_index)
                # Express psi1 as a vector
                y = np.zeros(n,dtype=np.complex)
                for ps,amp in psi1.items():
                    y[basis_index[ps]] = amp
                # If one would like to store psi1 as a sparse vector
                #y = scipy.sparse.csr_matrix(y)

                # Fast look-up of product states
                basis_state = {index : ps for ps, index in basis_index.items()}
                if rank == 0: print('Loop over in-coming photon energies...')
                for iwIn, wIn in enumerate(wIns):
                    # A = (wIn+1j*delta1+e)*\hat{1} - hOp.
                    a = scipy.sparse.csr_matrix(
                        ([wIn+1j*delta1+e]*n,(range(n),range(n))), shape=(n,n))
                    a -= h
                    # Find x by solving: a*x = y
                    # Biconjugate gradient stabilized method.
                    # Pure conjugate gradient does not apply since
                    # it requires a Hermitian matrix.
                    x,info = scipy.sparse.linalg.bicgstab(a,y)
                    if info > 0 :
                        print("Rank ", rank,
                              ": Convergence to tolerance not achieved")
                        print('#iterations = ',info)
                    elif info < 0 :
                        print("Rank ", rank, "illegal input or breakdown"
                              + " in conjugate gradient")
                    # Convert multi state from vector to dict format
                    psi2 = {}
                    for i, amp in enumerate(x):
                        if amp != 0:
                            psi2[basis_state[i]] = amp

                    # Loop over out-going transition operators
                    for tOut,tOpOut in enumerate(tOpsOut):
                        # Calculate state |psi3> = tOpOut |psi2>
                        # This state has no core-hole.
                        psi3 = applyOp(n_spin_orbitals, tOpOut, psi2,
                                       slaterWeightMin, restrictions,
                                       tOut_big[tOut])
                        # Normalization factor
                        normalization = sqrt(norm2(psi3))
                        for state in psi3.keys():
                            psi3[state] /= normalization
                        # Remove product states with small weight
                        for state, amp in list(psi3.items()):
                            if abs(amp)**2 < slaterWeightMin:
                                psi3.pop(state)
                        # Calculate Green's function
                        gs[iE,tIn,tOut,iwIn,:] = normalization**2*getGreen(
                            n_spin_orbitals, e, psi3, hOp, wLoss, delta2,
                            krylovSize, slaterWeightMin, restrictions, hGround,
                            parallelization_mode=parallelization_mode)
    elif parallelization_mode == 'wIn' or parallelization_mode == "H_build_wIn":
        # Loop over in-coming transition operators
        for tIn, tOpIn in enumerate(tOpsIn):
            tIn_big = {}
            # Loop over eigen states
            for iE in range(nE):
                psi =  psis[iE]
                e = es[iE]
                # Core-hole state
                psi1 = applyOp(n_spin_orbitals, tOpIn, psi, slaterWeightMin,
                               restrictions, tIn_big)
                # Hamiltonian acting on relevant product states. |PS> : {H|PS>}
                nTMP = len(hExcited)
                if parallelization_mode == "wIn":
                    h, basis_index = expand_basis_and_hamiltonian(
                        n_spin_orbitals, hExcited, hOp, psi1.keys(),
                        restrictions, parallelization_mode="serial")
                elif parallelization_mode == "H_build_wIn":
                    h, basis_index = expand_basis_and_hamiltonian(
                        n_spin_orbitals, hExcited, hOp, psi1.keys(),
                        restrictions, parallelization_mode="H_build")
                if rank == 0:
                    print("len(psi1), len(basis_index), len(hExcited), "
                          + "#elements added to hExcited: ",
                          len(psi1), len(basis_index), len(hExcited),
                          len(hExcited)-nTMP)
                n = len(basis_index)
                # Express psi1 as a vector
                y = np.zeros(n,dtype=np.complex)
                for ps, amp in psi1.items():
                    y[basis_index[ps]] = amp
                # If one would like to store psi1 as a sparse vector
                #y = scipy.sparse.csr_matrix(y)

                # Fast look-up of product states
                basis_state = {index : ps for ps, index in basis_index.items()}
                # Rank dependent variable
                g = {}
                if rank == 0: print('Loop over in-coming photon energies...')
                # Loop over in-coming photon energies, unique for each MPI rank
                for iwIn in get_job_tasks(rank, ranks, range(len(wIns))):
                    wIn = wIns[iwIn]
                    # Initialize Green's functions
                    g[iwIn] =  np.zeros((len(tOpsOut), len(wLoss)),
                                        dtype=np.complex)
                    # A = (wIn+1j*delta1+e)*\hat{1} - hOp.
                    a = scipy.sparse.csr_matrix(
                        ([wIn+1j*delta1+e]*n,(range(n),range(n))), shape=(n,n))
                    a -= h
                    # Find x by solving: a*x = y
                    # Biconjugate gradient stabilized method.
                    # Pure conjugate gradient does not apply since
                    # it requires a Hermitian matrix.
                    x,info = scipy.sparse.linalg.bicgstab(a,y)
                    if info > 0 :
                        print('convergence to tolerance not achieved')
                        print('#iterations = ',info)
                    elif info < 0 :
                        print("illegal input or breakdown "
                              + "in conjugate gradient")
                    # Convert multi state from vector to dict format
                    psi2 = {}
                    for i, amp in enumerate(x):
                        if amp != 0:
                            psi2[basis_state[i]] = amp
                    # Loop over out-going transition operators
                    for tOut, tOpOut in enumerate(tOpsOut):
                        # Calculate state |psi3> = tOpOut |psi2>
                        # This state has no core-hole.
                        psi3 = applyOp(n_spin_orbitals, tOpOut, psi2,
                                       slaterWeightMin, restrictions,
                                       tOut_big[tOut])
                        # Normalization factor
                        normalization = sqrt(norm2(psi3))
                        for state in psi3.keys():
                            psi3[state] /= normalization
                        # Remove product states with small weight
                        for state,amp in list(psi3.items()):
                            if abs(amp)**2 < slaterWeightMin:
                                psi3.pop(state)

                        t0 = time.time()

                        # Calculate Green's function
                        g[iwIn][tOut,:] = normalization**2*getGreen(
                            n_spin_orbitals, e, psi3, hOp, wLoss, delta2,
                            krylovSize, slaterWeightMin, restrictions, hGround,
                            parallelization_mode="serial")

                        if rank == 0: print("dt = ", time.time() - t0)

                # Distribute the Green's functions among the ranks
                for r in range(ranks):
                    gTmp = comm.bcast(g, root=r)
                    for iwIn, gValue in gTmp.items():
                        gs[iE,tIn,:,iwIn,:] = gValue
    return gs
