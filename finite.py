#!/usr/bin/env python3

"""

finite
======

This module contains functions doing the bulk of the calculations.

"""

import sys
from math import pi,sqrt
import numpy as np
from sympy.physics.wigner import gaunt
import itertools
from bisect import bisect_left
from collections import OrderedDict
import scipy.sparse
from mpi4py import MPI


from . import product_state_representation as psr
from .average import k_B, thermal_average


# MPI variables
comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size


def get_job_tasks(rank, ranks, tasks_tot):
    """
    Return a tuple of job task indices for a particular rank.

    This function distribute the job tasks in tasks_tot
    over all the ranks.

    Note
    ----
    This is a primerly a MPI help function.

    Parameters
    ----------
    rank : int
        Current MPI rank/worker.
    ranks : int
        Number of MPI ranks/workers in total.
    tasks_tot : list
        List of task indices.
        Length is the total number of job tasks.

    """
    n_tot = len(tasks_tot)
    nj = n_tot//ranks
    rest = n_tot%ranks
    #tasks = range(nj*rank, nj*rank + nj)
    tasks = [tasks_tot[i] for i in range(nj*rank, nj*rank + nj)]
    if rank < rest:
        #tasks.append(n_tot - rest + rank)
        tasks.append(tasks_tot[n_tot - rest + rank])
    return tuple(tasks)


def eigensystem(n_spin_orbitals, hOp, basis, nPsiMax, groundDiagMode='Lanczos',
               eigenValueTol=1e-9, slaterWeightMin=1e-7):
    """
    Return eigen-energies and eigenstates.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    hOp : dict
        tuple : float or complex
        The Hamiltonian operator to diagonalize.
        Each keyword contains ordered instructions
        where to add or remove electrons.
        Values indicate the strengths of
        the corresponding processes.
    basis : tuple
        All product states included in the basis.
    nPsiMax : int
        Number of eigenvalues to find.
    groundDiagMode : str
        'Lanczos' or 'full' diagonalization.
    eigenValueTol : float
        The precision of the returned eigenvalues.
    slaterWeightMin : float
        Minimum product state weight for product states to be kept.

    """
    if rank == 0: print('Create Hamiltonian matrix...')
    h = get_hamiltonian_matrix(n_spin_orbitals, hOp, basis)
    if rank == 0: print('<#Hamiltonian elements/column> = {:d}'.format(
        int(len(np.nonzero(h)[0])/len(basis))))
    if rank == 0: print('Diagonalize the Hamiltonian...')
    if groundDiagMode == 'full':
        es, vecs = np.linalg.eigh(h.todense())
        es = es[:nPsiMax]
        vecs = vecs[:,:nPsiMax]
    elif groundDiagMode == 'Lanczos':
        es, vecs = scipy.sparse.linalg.eigsh(h, k=nPsiMax, which='SA',
                                             tol=eigenValueTol)
        # Sort the eigenvalues and eigenvectors in ascending order.
        indices = np.argsort(es)
        es = np.array([es[i] for i in indices])
        vecs = np.array([vecs[:,i] for i in indices]).T
    else:
        print('Wrong diagonalization mode')
    if rank == 0: print("Proceed with {:d} eigenstates.\n".format(len(es)))
    psis = [({basis[i]:vecs[i,vi] for i in range(len(basis))
              if slaterWeightMin <= abs(vecs[i,vi])**2 })
            for vi in range(len(es))]
    return es, psis


def printExpValues(nBaths, es, psis, n=None):
    '''
    print several expectation values, e.g. E, N, L^2.
    '''
    if n == None:
        n = len(es)
    if rank == 0:
        print('E0 = {:7.4f}'.format(es[0]))
        print(('  i  E-E0  N(3d) N(egDn) N(egUp) N(t2gDn) '
               'N(t2gUp) Lz(3d) Sz(3d) L^2(3d) S^2(3d) L^2(3d+B) S^2(3d+B)'))
    if rank == 0 :
        for i,(e,psi) in enumerate(zip(es[:n] - es[0],psis[:n])):
            oc = getEgT2gOccupation(nBaths, psi)
            print(('{:3d} {:6.3f} {:5.2f} {:6.3f} {:7.3f} {:8.3f} {:7.3f}'
                   ' {:7.2f} {:6.2f} {:7.2f} {:7.2f} {:8.2f} {:8.2f}').format(
                i, e, getTraceDensityMatrix(nBaths, psi),
                oc[0], oc[1], oc[2], oc[3],
                getLz3d(nBaths, psi), getSz3d(nBaths, psi),
                getLsqr3d(nBaths, psi), getSsqr3d(nBaths, psi),
                getLsqr3dWithBath(nBaths, psi), getSsqr3dWithBath(nBaths, psi)))
        print("\n")

def printThermalExpValues(nBaths, es, psis, T=300, cutOff=10):
    '''
    print several thermal expectation values, e.g. E, N, L^2.

    cutOff - float. Energies more than cutOff*kB*T above the
            lowest energy is not considered in the average.
    '''
    e = es - es[0]
    # Select relevant energies
    mask = e < cutOff*k_B*T
    e = e[mask]
    psis = np.array(psis)[mask]
    occs = thermal_average(
        e, np.array([getEgT2gOccupation(nBaths, psi) for psi in psis]),
        T=T)
    if rank == 0:
        print('<E-E0> = {:4.3f}'.format(thermal_average(e, e, T=T)))
        print('<N(3d)> = {:4.3f}'.format(thermal_average(
            e, [getTraceDensityMatrix(nBaths, psi) for psi in psis], T=T)))
        print('<N(egDn)> = {:4.3f}'.format(occs[0]))
        print('<N(egUp)> = {:4.3f}'.format(occs[1]))
        print('<N(t2gDn)> = {:4.3f}'.format(occs[2]))
        print('<N(t2gUp)> = {:4.3f}'.format(occs[3]))
        print('<Lz(3d)> = {:4.3f}'.format(thermal_average(
            e,[getLz3d(nBaths,psi) for psi in psis],T=T)))
        print('<Sz(3d)> = {:4.3f}'.format(thermal_average(
            e,[getSz3d(nBaths,psi) for psi in psis],T=T)))
        print('<L^2(3d)> = {:4.3f}'.format(thermal_average(
            e,[getLsqr3d(nBaths,psi) for psi in psis],T=T)))
        print('<S^2(3d)> = {:4.3f}'.format(thermal_average(
            e,[getSsqr3d(nBaths,psi) for psi in psis],T=T)))


def dc_MLFT(n3d_i, c, Fdd, n2p_i=None, Fpd=None, Gpd=None):
    r"""
    Return double counting (DC) in multiplet ligand field theory.

    Parameters
    ----------
    n3d_i : int
        Nominal (integer) 3d occupation.
    c : float
        Many-body correction to the charge transfer energy.
    n2p_i : int
        Nominal (integer) 2p occupation.
    Fdd : list
        Slater integrals {F_{dd}^k}, k \in [0,1,2,3,4]
    Fpd : list
        Slater integrals {F_{pd}^k}, k \in [0,1,2]
    Gpd : list
        Slater integrals {G_{pd}^k}, k \in [0,1,2,3]

    Notes
    -----
    The `c` parameter is related to the charge-transfer
    energy :math:`\Delta_{CT}` by:

    .. math:: \Delta_{CT} = (e_d-e_b) + c.

    """
    if not int(n3d_i) == n3d_i:
        raise ValueError('3d occupation should be an integer')
    if n2p_i != None and int(n2p_i) != n2p_i:
        raise ValueError('2p occupation should be an integer')

    # Average repulsion energy defines Udd and Upd
    Udd = Fdd[0] - 14.0/441*(Fdd[2] + Fdd[4])
    if n2p_i==None and Fpd==None and Gpd==None:
        return Udd*n3d_i - c
    if n2p_i==6 and Fpd!=None and Gpd!=None:
        Upd = Fpd[0] - (1/15.)*Gpd[1] - (3/70.)*Gpd[3]
        return [Udd*n3d_i+Upd*n2p_i-c,Upd*(n3d_i+1)-c]
    else:
        raise ValueError('double counting input wrong.')


def get_spherical_2_cubic_matrix(spinpol=False,l=2):
    r"""
    Return unitary ndarray for transforming from spherical
    to cubic harmonics.

    Parameters
    ----------
    spinpol : boolean
        If transformation involves spin.
    l : integer
        Angular momentum number. p: l=1, d: l=2.

    Returns
    -------
    u : (M,M) ndarray
        The unitary matrix from spherical to cubic harmonics.

    Notes
    -----
    Element :math:`u_{i,j}` represents the contribution of spherical
    harmonics :math:`i` to the cubic harmonic :math:`j`:

    .. math:: \lvert l_j \rangle  = \sum_{i=0}^4 u_{d,(i,j)}
    \lvert Y_{d,i} \rangle.

    """
    if l == 1:
        u = np.zeros((3,3),dtype=np.complex)
        u[0,0] = 1j/np.sqrt(2)
        u[2,0] = 1j/np.sqrt(2)
        u[0,1] = 1/np.sqrt(2)
        u[2,1] = -1/np.sqrt(2)
        u[1,2] = 1
    elif l == 2:
        u = np.zeros((5,5),dtype=np.complex)
        u[2,0] = 1
        u[[0,-1],1] = 1/np.sqrt(2)
        u[1,2] = -1j/np.sqrt(2)
        u[-2,2] = -1j/np.sqrt(2)
        u[1,3] = 1/np.sqrt(2)
        u[-2,3] = -1/np.sqrt(2)
        u[0,4] = 1j/np.sqrt(2)
        u[-1,4] = -1j/np.sqrt(2)
    if spinpol:
        n,m = np.shape(u)
        U = np.zeros((2*n,2*m),dtype=np.complex)
        U[0:n,0:m] = u
        U[n:,m:] = u
        u = U
    return u


def daggerOp(op):
    '''
    return op^dagger
    '''
    opDagger = {}
    for process,value in op.items():
        processNew = []
        for e in process[-1::-1]:
            if e[1] == 'a':
                processNew.append((e[0],'c'))
            elif e[1] == 'c':
                processNew.append((e[0],'a'))
            else:
                print('Operator type unknown')
        processNew = tuple(processNew)
        opDagger[processNew] = value.conjugate()
    return opDagger


def get_basis(nBaths, valBaths, dnValBaths, dnConBaths, dnTol, n0imp):
    """
    Return restricted basis of product states.

    Parameters
    ----------
    nBaths : dict
    valBaths : dict
    dnValBaths : dict
    dnConBaths : dict
    dnTol : dict
    n0imp : dict

    """
    # Sanity check
    for l in nBaths.keys():
        assert valBaths[l] <= nBaths[l]

    # For each partition, create all configurations
    # given the occupation in that partition.
    basisL = OrderedDict()
    for l in nBaths.keys():
        if rank == 0: print('l=',l)
        # Add configurations to this list
        basisL[l] = []
        # Loop over different occupation partitions
        for dnVal in range(dnValBaths[l]+1):
            for dnCon in range(dnConBaths[l]+1):
                deltaNimp = dnVal - dnCon
                if abs(deltaNimp) <= dnTol[l]:
                    nImp = n0imp[l]+deltaNimp
                    nVal = 2*(2*l+1)*valBaths[l]-dnVal
                    nCon = dnCon
                    # Check for over occupation
                    assert nVal <= 2*(2*l+1)*valBaths[l]
                    assert nCon <= 2*(2*l+1)*(nBaths[l]-valBaths[l])
                    assert nImp <= 2*(2*l+1)

                    if rank == 0: print('New partition occupations:')
                    #if rank == 0:
                    #    print('nImp,dnVal,dnCon = {:d},{:d},{:d}'.format(
                    #        nImp,dnVal,dnCon))
                    if rank == 0:
                        print('nImp,nVal,nCon = {:d},{:d},{:d}'.format(
                            nImp, nVal, nCon))
                    # Impurity electrons
                    indices = range(c2i(nBaths,(l,-l,0)),
                                    c2i(nBaths,(l,l,1))+1)
                    basisImp = tuple(itertools.combinations(indices,nImp))
                    # Valence bath electrons
                    if valBaths[l] == 0:
                        # One way of having zero electrons
                        # in zero spin-orbitals
                        basisVal = ((),)
                    else:
                        indices = range(c2i(nBaths,(l,-l,0,0)),
                                        c2i(nBaths,(l,l,1,valBaths[l]-1))+1)
                        basisVal = tuple(itertools.combinations(indices,nVal))
                    # Conduction bath electrons
                    if nBaths[l]-valBaths[l] == 0:
                        # One way of having zero electrons
                        # in zero spin-orbitals
                        basisCon = ((),)
                    else:
                        indices = range(c2i(nBaths,(l,-l,0,valBaths[l])),
                                        c2i(nBaths,(l,l,1,nBaths[l]-1))+1)
                        basisCon = tuple(itertools.combinations(indices,nCon))
                    # Concatenate partitions
                    for bImp in basisImp:
                        for bVal in basisVal:
                            for bCon in basisCon:
                                basisL[l].append(bImp+bVal+bCon)
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    basis = []
    assert len(nBaths) == 2
    # Angular momentum
    l1, l2 = nBaths.keys()
    # Two explicit loops is only valid for two impurity blocks
    for b1 in basisL[l1]:
        for b2 in basisL[l2]:
            basis.append(psr.tuple2int(tuple(sorted(b1+b2)), n_spin_orbitals))
    return tuple(basis)


def inner(a,b):
    r'''
    Return :math:`\langle a | b \rangle`

    Parameters
    ----------
    a : dict
        Multi configurational state
    b : dict
        Multi configurational state

    Acknowledgement: Written entirely by Petter Saterskog
    '''
    acc=0
    for state,amp in b.items():
    	if state in a:
    		acc += np.conj(a[state])*amp
    return acc


def append_to_first(h1, h2):
    r"""
    Update dictionary h1 by appending it with the elements from h2.

    If h1 and h2 contain same key, the value of h1 for this key
    will be overwritten by the value of h2 for this key.

    Parameters
    ----------
    h1 : dict
    h2 : dict

    """
    for s, a in h2.items():
        #assert s not in h1
        h1[s] = a


def addToFirst(psi1, psi2, mul=1):
    r"""
    To state :math:`|\psi_1\rangle`, add  :math:`mul * |\psi_2\rangle`.

    Acknowledgement: Written by Petter Saterskog.

    Parameters
    ----------
    psi1 : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.
    psi2 : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.
    mul : int, float or complex
        Optional

    """
    for s, a in psi2.items():
    	if s in psi1:
    		psi1[s] += a*mul
    	else:
    		psi1[s] = a*mul

def binary_search(a, x):
    '''
    Return index to the leftmost value exactly equal to x.

    If x is not in the list, return -1.

    '''
    i = bisect_left(a, x)
    return i if i != len(a) and a[i] == x else -1


def binary_search_bigger(a, x):
    '''
    Return the index to the leftmost value bigger than x,
    if x is not in the list.

    If x is in the list, return -1.

    '''
    i = bisect_left(a, x)
    return i if i == len(a) or a[i] != x else -1


def remove_t(i, state):
    '''
    Remove electron at orbital i in state.

    Parameters
    ----------
    i : int
        Spin-orbital index
    state : tuple
        Product state.
        Elements are indices of occupied orbitals.

    Returns
    -------
    stateNew : tuple
        Product state
    amp : int
        Amplitude

    '''
    j = binary_search(state,i)
    if j != -1:
        stateNew = state[:j] + state[j+1:]
        amp = 1 if j%2 == 0 else -1
        return stateNew, amp
    else:
        return (), 0


def remove_i(n_spin_orbitals, i, state):
    """
    Remove electron at orbital i in state.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    i : int
        Spin-orbital index
    state : int
        Product state.

    Returns
    -------
    state_new : int
        Product state
    amp : int
        Amplitude

    """
    binary = psr.int2binary(state, n_spin_orbitals)
    if binary[i] == "0":
        return -1, 0
    elif binary[i] == "1":
        state_new = state - 2**(n_spin_orbitals-i-1)
        amp = 1 if binary[:i].count("1") % 2 == 0 else -1
        return state_new, amp
    else:
        sys.exit("Binary representation of state is wrong.")

def remove_b(i, state):
    """
    Remove electron at orbital i in state.

    Parameters
    ----------
    i : int
        Spin-orbital index
    state : str
        Product state.

    Returns
    -------
    state_new : str
        Product state
    amp : int
        Amplitude

    """
    if state[i] == "0":
        return -1, 0
    elif state[i] == "1":
        state_new = state[:i] + "0" + state[i+1:]
        amp = 1 if state[:i].count("1") % 2 == 0 else -1
        return state_new, amp
    else:
        sys.exit("Binary representation of state is wrong.")


def create_t(i, state):
    '''
    Create electron at orbital i in state.

    Parameters
    ----------
    i : int
        Spin-orbital index
    state : tuple
        Product state.
        Elements are indices of occupied orbitals.

    Returns
    -------
    stateNew : tuple
        Product state
    amp : int
        Amplitude

    '''
    j = binary_search_bigger(state, i)
    if j != -1:
        amp = 1 if j%2==0 else -1
        cstate = state[:j] + (i,) + state[j:]
        return cstate, amp
    else:
        return (), 0


def create_i(n_spin_orbitals, i, state):
    """
    Create electron at orbital i in state.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    i : int
        Spin-orbital index.
    state : int
        Product state.

    Returns
    -------
    stateNew : int
        Product state.
    amp : int
        Amplitude.

    """
    binary = psr.int2binary(state, n_spin_orbitals)
    if binary[i] == "1":
        return -1, 0
    elif binary[i] == "0":
        state_new = state + 2**(n_spin_orbitals-i-1)
        amp = 1 if binary[:i].count("1") % 2 == 0 else -1
        return state_new, amp
    else:
        sys.exit("Binary representation of state is wrong.")

def create_b(i, state):
    """
    Create electron at orbital i in state.

    Parameters
    ----------
    i : int
        Spin-orbital index.
    state : str
        Product state.

    Returns
    -------
    stateNew : str
        Product state.
    amp : int
        Amplitude.

    """
    if state[i] == "1":
        return -1, 0
    elif state[i] == "0":
        state_new = state[:i] + "1" + state[i+1:]
        amp = 1 if state[:i].count("1") % 2 == 0 else -1
        return state_new, amp
    else:
        sys.exit("Binary representation of state is wrong.")


def a(n_spin_orbitals, i, psi):
    r'''
    Return :math:`|psi' \rangle = c_i |psi \rangle`.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    i : int
        Spin-orbital index
    psi : dict
        Multi configurational state

    Returns
    -------
    ret : dict
        New multi configurational state

    '''
    ret={}
    for state, amp in psi.items():
        state_new, sign = remove_i(n_spin_orbitals, i, state)
        if sign != 0: ret[state_new] = amp*sign
    return ret


def c(n_spin_orbitals, i, psi):
    r'''
    Return :math:`|psi' \rangle = c_i^\dagger |psi \rangle`.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    i : int
        Spin-orbital index
    psi : dict
        Multi configurational state

    Returns
    -------
    ret : dict
        New multi configurational state

    '''
    ret={}
    for state, amp in psi.items():
        state_new, sign = create_i(n_spin_orbitals, i, state)
        if sign != 0: ret[state_new] = amp*sign
    return ret


def gauntC(k,l,m,lp,mp,prec=16):
    '''
    return "nonvanishing" Gaunt coefficients of
    Coulomb interaction expansion.
    '''
    c = sqrt(4*pi/(2*k+1))*(-1)**m*gaunt(l,k,lp,-m,m-mp,mp,prec=prec)
    return float(c)


def getU(l1,m1,l2,m2,l3,m3,l4,m4,R):
    r'''
    Return Hubbard U term for four spherical harmonics functions.

    Scattering process:

    :math:`u_{l_1,m_1,l_2,m_2,l_3,m_3,l_4,m_4}
    * c_{l_1,m_1}^\dagger c_{l_2,m_2}^\dagger c_{l_3,m_3} c_{l_4,m_4}`.

    Parameters
    ----------
    l1 : int
        angular momentum of orbital 1
    m1 : int
        z projected angular momentum of orbital 1
    l2 : int
        angular momentum of orbital 2
    m2 : int
        z projected angular momentum of orbital 2
    l3 : int
        angular momentum of orbital 3
    m3 : int
        z projected angular momentum of orbital 3
    l4 : int
        angular momentum of orbital 4
    m4 : int
        z projected angular momentum of orbital 4
    R : list
        Slater-Condon parameters.
        Elements R[k] fullfill
        :math:`0<=k<=\textrm{min}(|l_1+l_4|,|l_2+l_3|)`.
        Note, U is nonzero if :math:`k+l_1+l_4` is an even integer
        and :math:`k+l_3+l_2` is an even integer.
        For example: if :math:`l_1=l_2=l_3=l_4=2`,
        R = [R0,R1,R2,R3,R4] and only R0,R2 and R4 will
        give nonzero contribution.

    Returns
    -------
    u - float
        Hubbard U term.

'''
    # Check if angular momentum is conserved
    if m1+m2 == m3+m4:
        u = 0
        for k,Rk in enumerate(R):
            u += Rk*gauntC(k,l1,m1,l4,m4)*gauntC(k,l3,m3,l2,m2)
    else:
        u = 0
    return u


def printGaunt(l=2,lp=2):
    '''
    print Gaunt coefficients.

    Parameters
    ----------
    l : int
        angular momentum
    lp : int
        angular momentum
    '''
    # Print Gauent coefficients
    for k in range(l+lp+1):
        if rank == 0: print('k={:d}'.format(k))
        for m in range(-l,l+1):
            s = ''
            for mp in range(-lp,lp+1):
                s += ' {:3.2f}'.format(gauntC(k,l,m,lp,mp))
            if rank == 0: print(s)
        if rank == 0: print('')


def getNoSpinUop(l1,l2,l3,l4,R):
    '''
    Return non-spin polarized U operator.

    Scattering processes:

    :math:`1/2 \sum_{m_1,m_2,m_3,m_4}
    u_{l_1,m_1,l_2,m_2,l_3,m_3,l_4,m_4}
    c_{l_1,m_1}^\dagger c_{l_2,m_2}^\dagger c_{l_3,m_3} c_{l_4,m_4}`.

    No spin polarization considered, thus basis is: (l,m)

    '''
    #uMatrix = np.zeros((2*l1+1,2*l2+1,2*l3+1,2*l4+1))
    uDict = {}
    for i1,m1 in enumerate(range(-l1,l1+1)):
        for i2,m2 in enumerate(range(-l2,l2+1)):
            for i3,m3 in enumerate(range(-l3,l3+1)):
                for i4,m4 in enumerate(range(-l4,l4+1)):
                    u = getU(l1,m1,l2,m2,l3,m3,l4,m4,R)
                    if u != 0:
                        #uMatrix[i1,i2,i3,i4] = u
                        uDict[((l1,m1),(l2,m2),(l3,m3),(l4,m4))] = u/2.
    return uDict


def getUop(l1,l2,l3,l4,R):
    r'''
    Return U operator.

    Scattering processes:
    :math:`1/2 \sum_{m_1,m_2,m_3,m_4} u_{l_1,m_1,l_2,m_2,l_3,m_3,l_4,m_4}
    * \sum_{s,sp} c_{l_1,m_1,s}^\dagger c_{l_2,m_2,sp}^\dagger
    c_{l_3,m_3,sp} c_{l_4,m_4,s}`.

    Spin polarization is considered, thus basis: (l,m,s),
    where :math:`s \in \{0, 1 \}` and these indices respectively
    corresponds to the physical values
    :math:`\{-\frac{1}{2},\frac{1}{2} \}`.

    Returns
    -------
    uDict : dict
        Elements of the form:
        ((sorb1,'c'),(sorb2,'c'),(sorb3,'a'),(sorb4,'a')) : u/2
        where sorb1 is a superindex of (l,m,s).

    '''
    uDict = {}
    for m1 in range(-l1,l1+1):
        for m2 in range(-l2,l2+1):
            for m3 in range(-l3,l3+1):
                for m4 in range(-l4,l4+1):
                    u = getU(l1,m1,l2,m2,l3,m3,l4,m4,R)
                    if u != 0:
                        for s in range(2):
                            for sp in range(2):
                                proccess = (((l1,m1,s),'c'),((l2,m2,sp),'c'),
                                            ((l3,m3,sp),'a'),((l4,m4,s),'a'))
                                # Pauli exclusion principle
                                if not(s == sp and
                                       ((l1,m1)==(l2,m2) or
                                        (l3,m3)==(l4,m4))):
                                    uDict[proccess] = u/2.
    return uDict


def addOps(ops):
    '''
    Return one operator, represented as a dictonary.

    Parameters
    ----------
    ops : list
        Operators

    Returns
    -------
    opSum : dict

    '''
    opSum = {}
    for op in ops:
        for sOp,value in op.items():
            if value != 0:
              if sOp in opSum:
                  opSum[sOp] += value
              else:
                  opSum[sOp] = value
    return opSum


def get2p3dSlaterCondonUop(Fdd=[9,0,8,0,6], Fpp=[20,0,8],
                            Fpd=[10,0,8], Gpd=[0,3,0,2]):
    '''
    Return a 2p-3d U operator containing a sum of
    different Slater-Condon proccesses.

    Parameters
    ----------
    Fdd : list
    Fpp : list
    Fpd : list
    Gpd : list

    '''
    # Calculate F_dd^{0,2,4}
    FddOp = getUop(l1=2,l2=2,l3=2,l4=2,R=Fdd)
    # Calculate F_pp^{0,2}
    FppOp = getUop(l1=1,l2=1,l3=1,l4=1,R=Fpp)
    # Calculate F_pd^{0,2}
    FpdOp1 = getUop(l1=1,l2=2,l3=2,l4=1,R=Fpd)
    FpdOp2 = getUop(l1=2,l2=1,l3=1,l4=2,R=Fpd)
    FpdOp = addOps([FpdOp1,FpdOp2])
    # Calculate G_pd^{1,3}
    GpdOp1 = getUop(l1=1,l2=2,l3=1,l4=2,R=Gpd)
    GpdOp2 = getUop(l1=2,l2=1,l3=2,l4=1,R=Gpd)
    GpdOp = addOps([GpdOp1,GpdOp2])
    # Add operators
    uOp = addOps([FddOp,FppOp,FpdOp,GpdOp])
    return uOp


def getSOCop(xi,l=2):
    '''
    Return SOC operator for one l-shell.
    '''
    opDict = {}
    for m in range(-l,l+1):
        for s in range(2):
            value = xi*m*(1/2. if s==1 else -1/2.)
            opDict[(((l,m,s),'c'),((l,m,s),'a'))] = value
    for m in range(-l,l):
        value = xi/2.*sqrt((l-m)*(l+m+1))
        opDict[(((l,m,1),'c'),((l,m+1,0),'a'))] = value
        opDict[(((l,m+1,0),'c'),((l,m,1),'a'))] = value
    return opDict


def c2i(nBaths,spinOrb):
    '''
    Return an index, representing a spin-orbital.

    Parameters
    ----------
    nbaths : dict
        angular momentum : number of bath sets
    spinOrb : tuple
        (l,m,s) or (l,m,s,bathSet)

    '''
    i = 0
    for lp in nBaths.keys():
        for mp in range(-lp,lp+1):
            for sp in range(2):
                if (lp,mp,sp) == spinOrb:
                    return i
                i += 1
    for lp,nBathSets in nBaths.items():
        for bathSet in range(nBathSets):
            for mp in range(-lp,lp+1):
                for sp in range(2):
                    if (lp,mp,sp,bathSet) == spinOrb:
                        return i
                    i += 1


def i2c(nBaths,i):
    '''
    Return an coordinate tuple, representing a spin-orbital.

    Parameters
    ----------
    nbaths : dict
        angular momentum : number of bath sets
    i : int
        Spin orbital index.

    Returns
    -------
    spinOrb : tuple
        (l,m,s) or (l,m,s,bathSet)

    '''
    k = 0
    for lp in nBaths.keys():
        # This if statement is just
        # for speed-up. Not really needed
        if k+2*(2*lp+1) <= i:
            k += 2*(2*lp+1)
            continue
        for mp in range(-lp,lp+1):
            for sp in range(2):
                if k == i:
                    return (lp,mp,sp)
                k += 1
    for lp,nBathSets in nBaths.items():
        # This if statement is just
        # for speed-up. Not really needed
        if k+nBathSets*2*(2*lp+1) <= i:
            k += nBathSets*2*(2*lp+1)
            continue
        for bathSet in range(nBathSets):
            # This if statement is just
            # for speed-up. Not really needed
            if k+2*(2*lp+1) <= i:
                k += 2*(2*lp+1)
                continue
            for mp in range(-lp,lp+1):
                for sp in range(2):
                    if k==i:
                        return (lp,mp,sp,bathSet)
                    k += 1


def getLz3d(nBaths, psi):
    r'''
    Return expectation value :math:`\langle psi| Lz_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi configurational state.

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    Lz = 0
    for state, amp in psi.items():
        tmp = 0
        for i in psr.int2tuple(state, n_spin_orbitals):
            spinOrb = i2c(nBaths,i)
            if len(spinOrb)==3 and spinOrb[0]==2:
                tmp += spinOrb[1]
        Lz += tmp*abs(amp)**2
    return Lz


def getSz3d(nBaths, psi):
    r'''
    Return expectation value :math:`\langle psi| Sz_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi configurational state.

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    Sz = 0
    for state,amp in psi.items():
        tmp = 0
        for i in psr.int2tuple(state, n_spin_orbitals):
            spinOrb = i2c(nBaths,i)
            if len(spinOrb)==3 and spinOrb[0]==2:
                tmp += -1/2. if spinOrb[2]==0 else 1/2.
        Sz += tmp*abs(amp)**2
    return Sz


def getSsqr3dWithBath(nBaths, psi, tol=1e-8):
    r'''
    Return expectation value :math:`\langle psi| S^2 |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        normalized multi configurational state.

    '''
    psi1 = applySz3dWithBath(nBaths, psi)
    psi2 = applySplus3dWithBath(nBaths, psi)
    psi3 = applySminus3dWithBath(nBaths, psi)
    S2 = norm2(psi1)+1/2.*(norm2(psi2)+norm2(psi3))
    if S2.imag > tol:
        print('Warning: <S^2> complex valued!')
    return S2.real


def getSsqr3d(nBaths, psi, tol=1e-8):
    r'''
    Return expectation value :math:`\langle psi| S^2_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        normalized multi configurational state.

    '''
    psi1 = applySz3d(nBaths, psi)
    psi2 = applySplus3d(nBaths, psi)
    psi3 = applySminus3d(nBaths, psi)
    S2 = norm2(psi1)+1/2.*(norm2(psi2)+norm2(psi3))
    if S2.imag > tol:
        print('Warning: <S^2> complex valued!')
    return S2.real


def getLsqr3dWithBath(nBaths, psi, tol=1e-8):
    r'''
    Return expectation value :math:`\langle psi| L^2 |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        normalized multi configurational state.

    '''
    psi1 = applyLz3dWithBath(nBaths, psi)
    psi2 = applyLplus3dWithBath(nBaths, psi)
    psi3 = applyLminus3dWithBath(nBaths, psi)
    L2 = norm2(psi1)+1/2.*(norm2(psi2)+norm2(psi3))
    if L2.imag > tol:
        print('Warning: <L^2> complex valued!')
    return L2.real


def getLsqr3d(nBaths, psi, tol=1e-8):
    r'''
    Return expectation value :math:`\langle psi| L^2_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        normalized multi configurational state.

    '''
    psi1 = applyLz3d(nBaths, psi)
    psi2 = applyLplus3d(nBaths, psi)
    psi3 = applyLminus3d(nBaths, psi)
    L2 = norm2(psi1)+1/2.*(norm2(psi2)+norm2(psi3))
    if L2.imag > tol:
        print('Warning: <L^2> complex valued!')
    return L2.real


def getTraceDensityMatrix(nBaths, psi, l=2):
    r'''
    Return  :math:`\langle psi| \sum_i c_i^\dagger c_i |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi configurational state.
    l : int (optional)
        Angular momentum

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    n = 0
    for state, amp in psi.items():
        binary = psr.int2binary(state, n_spin_orbitals)
        nState = 0
        for m in range(-l,l+1):
            for s in range(2):
                i = c2i(nBaths,(l,m,s))
                if binary[i] == "1":
                    nState += 1
        nState *= abs(amp)**2
        n += nState
    return n


def getDensityMatrix(nBaths, psi, l=2):
    r'''
    Return density matrix in spherical harmonics basis.

    :math:`n_{ij} = \langle i| \tilde{n} |j \rangle =
    \langle psi| c_j^\dagger c_i |psi \rangle`.

    Returns
    -------
    densityMatrix : dict
        keys of the form: :math:`((l,mi,si),(l,mj,sj))`.
        values of the form: :math:`\langle psi| c_j^\dagger c_i |psi \rangle`.

    Notes
    -----
    The perhaps suprising index notation is because
    of the following four equations:

    :math:`G_{ij}(\tau->0^-) = \langle c_j^\dagger c_i \rangle`.

    :math:`G_ij(\tau->0^-) = \langle i|\tilde{G}(\tau->0^-)|j \rangle`.

    :math:`\tilde{G}(\tau->0^-) = \tilde{n}`.

    :math:`n_{ij} = \langle i| \tilde{n} |j \rangle`.

    Note: Switched index order compared to the order of operators,
    where :math:`op[((li,mi,si),(lj,mj,sj))] = value`
    means operator: :math:`value * c_{li,mi,si}^\dagger c_{lj,mj,sj}`

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    densityMatrix = OrderedDict()
    for mi in range(-l,l+1):
            for mj in range(-l,l+1):
                for si in range(2):
                    for sj in range(2):
                        i = c2i(nBaths,(l,mi,si))
                        j = c2i(nBaths,(l,mj,sj))
                        psi_new = a(n_spin_orbitals, i, psi)
                        psi_new = c(n_spin_orbitals, j, psi_new)
                        tmp = inner(psi, psi_new)
                        if tmp != 0:
                            densityMatrix[((l,mi,si),(l,mj,sj))] = tmp
    return densityMatrix


def getDensityMatrixCubic(nBaths, psi):
    r'''
    Return density matrix in cubic harmonics basis.

    :math:`n_{ic,jc} = \langle ic| \tilde{n} |jc \rangle =
    \langle psi| c_{jc}^\dagger c_{ic} |psi \rangle`,
    where ic is a index containing a cubic harmonics and a spin.

    :math:`c_{ic}^\dagger = \sum_j u[j,i] c_j^\dagger`

    This gives:
    :math:`\langle psi| c_{jc}^\dagger c_{ic} |psi \rangle
    = \sum_{k,m} u[k,j] u[m,i]^{*}
    * \langle psi| c_{k,sj}^\dagger c_{m,si} |psi \rangle
    = \sum_{k,m} u[m,i]^* n[{m,si},{k,sj}] u[k,j]`

    Returns
    -------
    densityMatrix : dict
        keys of the form: :math:`((i,si),(j,sj))`.
        values of the form: :math:`\langle psi| c_{jc}^\dagger c_{ic}
         |psi \rangle`.

    '''
    # density matrix in spherical harmonics
    nSph = getDensityMatrix(nBaths, psi)
    l = 2
    # |i(cubic)> = sum_j u[j,i] |j(spherical)>
    u = get_spherical_2_cubic_matrix()
    nCub = OrderedDict()
    for i in range(2*l+1):
        for j in range(2*l+1):
            for si in range(2):
                for sj in range(2):
                    for k, mk in enumerate(range(-l,l+1)):
                        for m, mm in enumerate(range(-l,l+1)):
                            eSph = ((l,mm,si),(l,mk,sj))
                            if eSph in nSph:
                                tmp = np.conj(u[m,i])*nSph[eSph]*u[k,j]
                                if tmp != 0:
                                    eCub = ((i,si),(j,sj))
                                    if eCub in nCub:
                                        nCub[eCub] += tmp
                                    else:
                                        nCub[eCub] = tmp
    return nCub


def getEgT2gOccupation(nBaths, psi):
    r'''
    Return occupations of :math:`eg_\downarrow, eg_\uparrow,
    t2g_\downarrow, t2g_\uparrow` states.

    Calculate from density matrix diagonal:
    :math:`n_{ic,ic} = \langle psi| c_{ic}^\dagger c_{ic} |psi \rangle`,
    where `ic` is a cubic harmonics index, and
    :math:`c_{ic}^\dagger = \sum_j u[j,ic] c_j^\dagger`,
    where `j` is a spherical harmonics index.

    This gives:
    :math:`\langle psi| c_{ic,s}^\dagger c_{ic,s} |psi \rangle
    = \sum_{j,k} u[j,ic] u[k,ic]^{*}
    * \langle psi| c_{j,s}^\dagger c_{k,s} |psi \rangle
    = \sum_{j,k} u[k,ic]^*  n[{k,s},{j,s}] u[j,ic]`

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    l = 2
    # |i(cubic)> = sum_j u[j,i] |j(spherical)>
    u = get_spherical_2_cubic_matrix()
    eg_dn,eg_up,t2g_dn,t2g_up = 0,0,0,0
    for i in range(2*l+1):
        for j,mj in enumerate(range(-l,l+1)):
            for k,mk in enumerate(range(-l,l+1)):
                for s in range(2):
                    jj = c2i(nBaths,(l,mj,s))
                    kk = c2i(nBaths,(l,mk,s))
                    psi_new = a(n_spin_orbitals, kk, psi)
                    psi_new = c(n_spin_orbitals, jj, psi_new)
                    v = u[j,i]*np.conj(u[k,i])*inner(psi, psi_new)
                    if i<2:
                        if s==0:
                            eg_dn += v
                        else:
                            eg_up += v
                    else:
                        if s==0:
                            t2g_dn += v
                        else:
                            t2g_up += v
    occs = [eg_dn,eg_up,t2g_dn,t2g_up]
    for i in range(len(occs)):
        if abs(occs[i].imag) < 1e-12:
            occs[i] = occs[i].real
        else:
            print('Warning: Complex occupation')
    return occs


def applySz3dWithBath(nBaths, psi):
    r'''
    Return :math:`|psi' \rangle = S^{z} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state. 
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        for s in range(2):
            # Impurity
            i = c2i(nBaths,(l,m,s))
            psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, i, psi))
            addToFirst(psiNew, psiP, 1/2. if s==1 else -1/2.)
            # Bath
            for bathSet in range(nBaths[l]):
                i = c2i(nBaths,(l,m,s,bathSet))
                psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, i, psi))
                addToFirst(psiNew, psiP, 1/2. if s==1 else -1/2.)
    return psiNew


def applySz3d(nBaths, psi):
    r'''
    Return :math:`|psi' \rangle = S^{z}_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state. 
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        for s in range(2):
            i = c2i(nBaths,(l,m,s))
            psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, i, psi))
            addToFirst(psiNew, psiP, 1/2. if s==1 else -1/2.)
    return psiNew


def applyLz3dWithBath(nBaths, psi):
    r'''
    Return :math:`|psi' \rangle = L^{z} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state. 
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        for s in range(2):
            # Impurity
            i = c2i(nBaths,(l,m,s))
            psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, i, psi))
            addToFirst(psiNew,psiP,m)
            # Bath
            for bathSet in range(nBaths[l]):
                i = c2i(nBaths,(l,m,s,bathSet))
                psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, i, psi))
                addToFirst(psiNew, psiP, m)
    return psiNew


def applyLz3d(nBaths, psi):
    r'''
    Return :math:`|psi' \rangle = L^{z}_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state. 
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        for s in range(2):
            i = c2i(nBaths,(l,m,s))
            psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, i, psi))
            addToFirst(psiNew, psiP, m)
    return psiNew


def applySplus3dWithBath(nBaths, psi):
    r'''
    Return :math:`|psi' \rangle = S^{+} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state. 
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        # Impurity
        i = c2i(nBaths,(l,m,1))
        j = c2i(nBaths,(l,m,0))
        psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
        # sQ = 1/2.
        # sqrt((sQ-(-sQ))*(sQ+(-sQ)+1)) == 1
        addToFirst(psiNew,psiP)
        # Bath
        for bathSet in range(nBaths[l]):
            i = c2i(nBaths,(l,m,1,bathSet))
            j = c2i(nBaths,(l,m,0,bathSet))
            psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
            addToFirst(psiNew, psiP)
    return psiNew


def applySplus3d(nBaths, psi):
    r'''
    Return :math:`|psi' \rangle = S^{+}_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state. 
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        i = c2i(nBaths,(l,m,1))
        j = c2i(nBaths,(l,m,0))
        psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
        # sQ = 1/2.
        # sqrt((sQ-(-sQ))*(sQ+(-sQ)+1)) == 1
        addToFirst(psiNew,psiP)
    return psiNew


def applyLplus3dWithBath(nBaths, psi):
    r'''
    Return :math:`|psi' \rangle = L^{+} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state. 
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    psiNew = {}
    l = 2
    for m in range(-l,l):
        for s in range(2):
            # Impurity
            i = c2i(nBaths,(l,m+1,s))
            j = c2i(nBaths,(l,m,s))
            psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
            addToFirst(psiNew,psiP,sqrt((l-m)*(l+m+1)))
            # Bath
            for bathSet in range(nBaths[l]):
                i = c2i(nBaths,(l,m+1,s,bathSet))
                j = c2i(nBaths,(l,m,s,bathSet))
                psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
                addToFirst(psiNew,psiP,sqrt((l-m)*(l+m+1)))
    return psiNew


def applyLplus3d(nBaths, psi):
    r'''
    Return :math:`|psi' \rangle = L^{+}_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state. 
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    psiNew = {}
    l = 2
    for m in range(-l,l):
        for s in range(2):
            i = c2i(nBaths,(l,m+1,s))
            j = c2i(nBaths,(l,m,s))
            psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
            addToFirst(psiNew,psiP,sqrt((l-m)*(l+m+1)))
    return psiNew


def applySminus3dWithBath(nBaths, psi):
    r'''
    Return :math:`|psi' \rangle = S^{-} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state. 
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        # Impurity
        i = c2i(nBaths,(l,m,0))
        j = c2i(nBaths,(l,m,1))
        psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
        # sQ = 1/2.
        # sqrt((sQ+sQ)*(sQ-sQ+1)) == 1
        addToFirst(psiNew,psiP)
        # Impurity
        for bathSet in range(nBaths[l]):
            i = c2i(nBaths,(l,m,0,bathSet))
            j = c2i(nBaths,(l,m,1,bathSet))
            psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
            addToFirst(psiNew,psiP)
    return psiNew


def applySminus3d(nBaths, psi):
    r'''
    Return :math:`|psi' \rangle = S^{-}_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state. 
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        i = c2i(nBaths,(l,m,0))
        j = c2i(nBaths,(l,m,1))
        psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
        # sQ = 1/2.
        # sqrt((sQ+sQ)*(sQ-sQ+1)) == 1
        addToFirst(psiNew,psiP)
    return psiNew


def applyLminus3dWithBath(nBaths, psi):
    r"""
    Return :math:`|psi' \rangle = L^{-} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state. 
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    """
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    psiNew = {}
    l = 2
    for m in range(-l+1,l+1):
        for s in range(2):
            # Impurity
            i = c2i(nBaths,(l,m-1,s))
            j = c2i(nBaths,(l,m,s))
            psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
            addToFirst(psiNew,psiP,sqrt((l+m)*(l-m+1)))
            # Bath
            for bathSet in range(nBaths[l]):
                i = c2i(nBaths,(l,m-1,s,bathSet))
                j = c2i(nBaths,(l,m,s,bathSet))
                psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
                addToFirst(psiNew,psiP,sqrt((l+m)*(l-m+1)))
    return psiNew


def applyLminus3d(nBaths, psi):
    r'''
    Return :math:`|psi' \rangle = L^{-}_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state. 
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    '''
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)*(1+nset) for ang, nset in nBaths.items())
    psiNew = {}
    l = 2
    for m in range(-l+1,l+1):
        for s in range(2):
            i = c2i(nBaths,(l,m-1,s))
            j = c2i(nBaths,(l,m,s))
            psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
            addToFirst(psiNew,psiP,sqrt((l+m)*(l-m+1)))
    return psiNew


def applyOp(n_spin_orbitals, op, psi, slaterWeightMin=1e-12, restrictions=None,
            opResult=None):
    r"""
    Return :math:`|psi' \rangle = op |psi \rangle`.

    If opResult is not None, it is updated to contain information of how the
    operator op acted on the product states in psi.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    op : dict
        Operator of the format
        tuple : amplitude,
        where each tuple describes a scattering
        process. Examples of possible tuples (and their meanings) are:
        ((i,'c'))  <-> c_i^dagger
        ((i,'a'))  <-> c_i
        ((i,'c'),(j,'a'))  <-> c_i^dagger c_j
        ((i,'c'),(j,'c'),(k,'a'),(l,'a')) <-> c_i^dagger c_j^dagger c_k c_l
    psi : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.
    slaterWeightMin : float
        Restrict the number of product states by
        looking at `|amplitudes|^2`.
    restrictions : dict
        Restriction the occupation of generated
        product states.
    opResult : dict
        In and output argument.
        If present, the results of the operator op acting on each
        product state in the state psi is added and stored in this
        variable.

    Returns
    -------
    psiNew : dict
        New state of the same format as psi.


    Note
    ----
    Different implementations exist.
    They return the same result, but calculations vary a bit.

    """
    psiNew = {}
    if opResult is None and restrictions != None:
        # Loop over product states in psi.
        for state, amp in psi.items():
            #assert amp != 0
            binary = psr.int2binary(state, n_spin_orbitals)
            for process, h in op.items():
                #assert h != 0
                # Initialize state
                stateA_binary = binary
                signTot = 1
                for i, action in process[-1::-1]:
                    if action == 'a':
                        stateB_binary, sign = remove_b(i, stateA_binary)
                    elif action == 'c':
                        stateB_binary, sign = create_b(i, stateA_binary)
                    if sign == 0:
                        break
                    stateA_binary = stateB_binary
                    signTot *= sign
                else:
                    stateB = psr.binary2int(stateB_binary)
                    if stateB in psiNew:
                        psiNew[stateB] += amp*h*signTot
                    else:
                        # Convert product state to the tuple representation.
                        stateB_tuple = psr.binary2tuple(stateB_binary)
                        #sB_t = psr.int2tuple(sB, n_spin_orbitals)
                        # Check that product state sB fulfills
                        # occupation restrictions.
                        for restriction, occupations in restrictions.items():
                            n = len(restriction.intersection(stateB_tuple))
                            if n < occupations[0] or occupations[1] < n:
                                break
                        else:
                            # Occupations ok, so add contributions
                            psiNew[stateB] = amp*h*signTot
    elif opResult is None and restrictions == None:
        # Loop over product states in psi.
        for state, amp in psi.items():
            #assert amp != 0
            binary = psr.int2binary(state, n_spin_orbitals)
            for process, h in op.items():
                #assert h != 0
                # Initialize state
                stateA_binary = binary
                signTot = 1
                for i, action in process[-1::-1]:
                    if action == 'a':
                        stateB_binary, sign = remove_b(i, stateA_binary)
                    elif action == 'c':
                        stateB_binary, sign = create_b(i, stateA_binary)
                    if sign == 0:
                        break
                    stateA_binary = stateB_binary
                    signTot *= sign
                else:
                    stateB = psr.binary2int(stateB_binary)
                    if stateB in psiNew:
                        psiNew[stateB] += amp*h*signTot
                    else:
                        psiNew[stateB] = amp*h*signTot
    elif restrictions != None:
        # Loop over product states in psi.
        for state, amp in psi.items():
            #assert amp != 0
            if state in opResult:
                addToFirst(psiNew, opResult[state], amp)
            else:
                binary = psr.int2binary(state, n_spin_orbitals)
                # Create new element in opResult
                # Store H|PS> for product states |PS> not yet in opResult
                opResult[state] = {}
                for process, h in op.items():
                    #assert h != 0
                    # Initialize state
                    stateA_binary = binary
                    signTot = 1
                    for i, action in process[-1::-1]:
                        if action == 'a':
                            stateB_binary, sign = remove_b(i, stateA_binary)
                        elif action == 'c':
                            stateB_binary, sign = create_b(i, stateA_binary)
                        if sign == 0:
                            break
                        stateA_binary = stateB_binary
                        signTot *= sign
                    else:
                        stateB = psr.binary2int(stateB_binary)
                        if stateB in psiNew:
                            # Occupations ok, so add contributions
                            psiNew[stateB] += amp*h*signTot
                            if stateB in opResult[state]:
                                opResult[state][stateB] += h*signTot
                            else:
                                opResult[state][stateB] = h*signTot
                        else:
                            # Convert product state to the tuple representation.
                            stateB_tuple = psr.binary2tuple(stateB_binary)
                            # Check that product state sB fulfills the
                            # occupation restrictions.
                            for restriction,occupations in restrictions.items():
                                n = len(restriction.intersection(stateB_tuple))
                                if n < occupations[0] or occupations[1] < n:
                                    break
                            else:
                                # Occupations ok, so add contributions
                                psiNew[stateB] = amp*h*signTot
                                opResult[state][stateB] = h*signTot
    elif restrictions == None:
        # Loop over product states in psi.
        for state, amp in psi.items():
            #assert amp != 0
            if state in opResult:
                addToFirst(psiNew, opResult[state], amp)
            else:
                binary = psr.int2binary(state, n_spin_orbitals)
                # Create new element in opResult
                # Store H|PS> for product states |PS> not yet in opResult
                opResult[state] = {}
                for process, h in op.items():
                    #assert h != 0
                    # Initialize state
                    stateA_binary = binary
                    signTot = 1
                    for i, action in process[-1::-1]:
                        if action == 'a':
                            stateB_binary, sign = remove_b(i, stateA_binary)
                        elif action == 'c':
                            stateB_binary, sign = create_b(i, stateA_binary)
                        if sign == 0:
                            break
                        stateA_binary = stateB_binary
                        signTot *= sign
                    else:
                        stateB = psr.binary2int(stateB_binary)
                        if stateB in opResult[state]:
                            opResult[state][stateB] += h*signTot
                        else:
                            opResult[state][stateB] = h*signTot
                        if stateB in psiNew:
                            psiNew[stateB] += amp*h*signTot
                        else:
                            psiNew[stateB] = amp*h*signTot
    else:
        print('Warning: method not implemented.')

    # Remove product states with small weight
    for state, amp in list(psiNew.items()):
        if abs(amp)**2 < slaterWeightMin:
            psiNew.pop(state)
    return psiNew


def get_hamiltonian_matrix(n_spin_orbitals, hOp, basis, mode='sparse_MPI'):
    """
    Return Hamiltonian expressed in the provided basis of product states.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    hOp : dict
        tuple : float or complex
        The Hamiltonain operator to diagonalize.
        Each keyword contains ordered instructions
        where to add or remove electrons.
        Values indicate the strengths of
        the corresponding processes.
    basis : tuple
        All product states included in the basis.
    mode : str
        Algorithm for calculating the Hamiltonian.

    """
    # Number of basis states
    n = len(basis)
    basis_index = {basis[i]:i for i in range(n)}
    if rank == 0: print('Filling the Hamiltonian...')
    progress = 0
    if mode == 'dense_serial':
        h = np.zeros((n,n),dtype=np.complex)
        for j in range(n):
            if rank == 0 and progress + 10 <= int(j*100./n):
                progress = int(j*100./n)
                print('{:d}% done'.format(progress))
            res = applyOp(n_spin_orbitals, hOp, {basis[j]:1})
            for k,v in res.items():
                if k in basis_index:
                    h[basis_index[k], j] = v
    elif mode == 'dense_MPI':
        h = np.zeros((n,n),dtype=np.complex)
        hRank = {}
        jobs = get_job_tasks(rank, ranks, range(n))
        for j in jobs:
            hRank[j] = {}
            if rank == 0 and progress + 10 <= int(j*100./len(jobs)):
                progress = int(j*100./len(jobs))
                print('{:d}% done'.format(progress))
            res = applyOp(n_spin_orbitals, hOp, {basis[j]:1})
            for k,v in res.items():
                if k in basis_index:
                    hRank[j][basis_index[k]] = v
        # Broadcast Hamiltonian dicts
        for r in range(ranks):
            hTmp = comm.bcast(hRank, root=r)
            for j,hj in hTmp.items():
                for i,hij in hj.items():
                    h[i,j] = hij
    elif mode == 'sparse_serial':
        data = []
        row = []
        col = []
        for j in range(n):
            if rank == 0 and progress + 10 <= int(j*100./n):
                progress = int(j*100./n)
                print('{:d}% done'.format(progress))
            res = applyOp(n_spin_orbitals, hOp, {basis[j]:1})
            for k,v in res.items():
                if k in basis_index:
                    data.append(v)
                    col.append(j)
                    row.append(basis_index[k])
        h = scipy.sparse.csr_matrix((data,(row,col)),shape=(n,n))
    elif mode == 'sparse_MPI':
        h = scipy.sparse.csr_matrix(([],([],[])),shape=(n,n))
        data = []
        row = []
        col = []
        jobs = get_job_tasks(rank, ranks, range(n))
        for j, job in enumerate(jobs):
            res = applyOp(n_spin_orbitals, hOp, {basis[job]:1})
            for k, v in res.items():
                if k in basis_index:
                    data.append(v)
                    col.append(job)
                    row.append(basis_index[k])
            if rank == 0 and progress + 10 <= int((j+1)*100./len(jobs)):
                progress = int((j+1)*100./len(jobs))
                print('{:d}% done'.format(progress))
        # Print out that the construction of Hamiltonian is done
        if rank == 0 and progress != 100:
            progress = 100
            print('{:d}% done'.format(progress))
        hSparse = scipy.sparse.csr_matrix((data,(row,col)),shape=(n,n))
        # Different ranks have information about different basis states.
        # Therefor, need to broadcast and append sparse Hamiltonians
        for r in range(ranks):
            h += comm.bcast(hSparse, root=r)
    return h


def get_hamiltonian_matrix_from_h_dict(h_big, basis,
                                       parallelization_mode='serial',
                                       mode='sparse'):
    """
    Return Hamiltonian expressed in the provided basis of product states
    in matrix format.

    Also return dictionary with product states in basis as keys,
    and basis indices as values.

    Parameters
    ----------
    h_big : dict
        Elements of the form `|PS> : {H|PS>}`,
        where `|PS>` is a product state.
        Thus, product states as keys.
        The result of the Hamiltonian acting on each product state is
        stored as the dictionary keys (in the format of dictionaries).
        h_big may contain more product states keys than in the active basis.
    basis : tuple
        All product states included in the basis.
    parallelization_mode : str
        Parallelization mode. Either: "serial" or "H_build".
    mode : str
        Algorithm for calculating the Hamiltonian and type format of
        returned Hamiltonian.
        'dense' or 'sparse'.

    """
    # Number of basis states
    n = len(basis)
    basis_index = {basis[i]:i for i in range(n)}
    #if rank == 0: print('Filling the Hamiltonian...')
    #progress = 0
    if mode == 'dense' and parallelization_mode == 'serial':
        h = np.zeros((n,n),dtype=np.complex)
        for j in range(n):
            #if rank == 0 and progress + 10 <= int(j*100./n):
            #    progress = int(j*100./n)
            #    print('{:d}% done'.format(progress))
            res = h_big[basis[j]]
            for k, v in res.items():
                h[basis_index[k], j] = v
    elif mode == 'sparse' and parallelization_mode == 'serial':
        data = []
        row = []
        col = []
        for j in range(n):
            #if rank == 0 and progress + 10 <= int(j*100./n):
            #    progress = int(j*100./n)
            #    print('{:d}% done'.format(progress))
            res = h_big[basis[j]]
            for k, v in res.items():
                data.append(v)
                col.append(j)
                row.append(basis_index[k])
        h = scipy.sparse.csr_matrix((data,(row,col)),shape=(n,n))
    elif mode == 'sparse' and parallelization_mode == 'H_build':
        h = scipy.sparse.csr_matrix(([],([],[])),shape=(n,n))
        data = []
        row = []
        col = []
        for job in get_job_tasks(rank, ranks, range(n)):
            res = h_big[basis[job]]
            for k, v in res.items():
                data.append(v)
                col.append(job)
                row.append(basis_index[k])
            #if rank == 0 and progress + 10 <= int((j+1)*100./len(jobs)):
            #    progress = int((j+1)*100./len(jobs))
            #    print('{:d}% done'.format(progress))
        # Print out that the construction of Hamiltonian is done
        #if rank == 0 and progress != 100:
        #    progress = 100
        #    print('{:d}% done'.format(progress))
        hSparse = scipy.sparse.csr_matrix((data,(row,col)),shape=(n,n))
        # Different ranks have information about different basis states.
        # Therefor, need to broadcast and append sparse Hamiltonians
        for r in range(ranks):
            h += comm.bcast(hSparse, root=r)
    else:
        sys.exit("Wrong input parameters")
    return h, basis_index


def expand_basis(n_spin_orbitals, h_big, hOp, basis0, restrictions,
                 parallelization_mode="serial"):
    """
    Return basis.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    h_big : dict
        Elements of the form `|PS> : {H|PS>}`,
        where `|PS>` is a product state.
        New product states might be added to this variable.
    hOp : dict
        The Hamiltonian. With elements of the form process : h_value
    basis0 : tuple
        List of product states.
        These product states are used to generate more basis states.
    restrictions : dict
        Restriction the occupation of generated product states.
    parallelization_mode : str
        Parallelization mode. Either: "serial", "serial_slow" or "H_build".


    Returns
    -------
    basis : tuple
        The restricted active space basis of product states.

    """
    # Copy basis0, to avoid changing it when the basis grows
    basis = list(basis0)
    i = 0
    n = len(basis)
    if parallelization_mode == "serial":
        while i < n :
            basis_set = frozenset(basis)
            basis_new = set()
            for b in basis[i:n]:
                res = applyOp(n_spin_orbitals, hOp, {b:1},
                              restrictions=restrictions, opResult=h_big)
                basis_new.update(set(res.keys()).difference(basis_set))
                #basis_new.update(basis_set.difference(res.keys()))
            i = n
            basis += list(basis_new)
            n = len(basis)
    elif parallelization_mode == "serial_slow":
        while i < n :
            res = applyOp(n_spin_orbitals, hOp, {basis[i]:1},
                          restrictions=restrictions, opResult=h_big)
            for ps in res.keys():
                if ps not in basis:
                    basis.append(ps)
            i += 1
            n = len(basis)
    elif parallelization_mode == "H_build":
        h_big_new_local = {}
        while i < n :
            #if rank == 0: print("i=",i,", n=",n)
            basis_set = frozenset(basis)
            basis_new_local = set()
            for state_index in get_job_tasks(rank, ranks, range(i,n)):
                state = basis[state_index]
                # Obtain H|state>
                if state in h_big:
                    res = h_big[state]
                else:
                    res = applyOp(n_spin_orbitals, hOp, {state:1},
                                  restrictions=restrictions)
                    h_big_new_local[state] = res
                basis_new_local.update(set(res.keys()).difference(basis_set))
            # Add unique elements of basis_new_local into basis_new
            basis_new = set()
            for r in range(ranks):
                basis_new.update(comm.bcast(basis_new_local, root=r))
            # Add basis_new to basis
            basis += list(basis_new)
            # Updated total number of product states |ps> where know H|ps>
            i = n
            # Updated total number of product states needed to consider.
            n = len(basis)
        # Merge h_big_new_local into h_big.
        for r in range(ranks):
            # Add rank r's h_big_new_local to h_big.
            # The keys in h_big_new_local are unique for each rank, i.e.
            # ps_i for rank r does not exist as a key in h_big_new_local
            # for any other rank than rank r.
            # Neither does it exist in the initial h_big.
            append_to_first(h_big, comm.bcast(h_big_new_local, root=r))
    else:
        sys.exit("Wrong parallelization parameter.")
    return tuple(basis)


def expand_basis_and_hamiltonian(n_spin_orbitals, h_big, hOp, basis0,
                                 restrictions, parallelization_mode="serial"):
    """
    Return Hamiltonian in matrix format.

    Also return dictionary with product states in basis as keys,
    and basis indices as values.

    Also possibly add new product state keys to h_big.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    h_big : dict
        Elements of the form `|PS> : {H|PS>}`,
        where `|PS>` is a product state.
        New product states might be added to this variable.
    hOp : dict
        The Hamiltonian. With elements of the form process : h_value
    basis0 : tuple
        List of product states.
        These product states are used to generate more basis states.
    restrictions : dict
        Restriction the occupation of generated product states.
    parallelization_mode : str
        Parallelization mode. Either: "serial" or "H_build".

    Returns
    -------
    h : scipy sparse csr_matrix
        The Hamiltonian acting on the relevant product states.
    basis_index : dict
        Elements of the form `|PS> : i`,
        where `|PS>` is a product state and i and integer.

    """
    # Obtain tuple containing different product states.
    # Possibly add new product state keys to h_big.
    basis = expand_basis(n_spin_orbitals, h_big, hOp, basis0, restrictions,
                         parallelization_mode)
    # Obtain Hamiltonian in matrix format.
    h, basis_index = get_hamiltonian_matrix_from_h_dict(
        h_big, basis, parallelization_mode)
    return h, basis_index


def add(psi1,psi2,mul=1):
    r"""
    Return :math:`|\psi\rangle = |\psi_1\rangle + mul * |\psi_2\rangle`

    Parameters
    ----------
    psi1 : dict
    psi2 : dict
    mul : int, float or complex
        Optional

    Returns
    -------
    psi : dict

    """
    psi = psi1.copy()
    for s,a in psi2.items():
        if s in psi:
            psi[s] += mul*a
        else:
            psi[s] = mul*a
    return psi


def norm2(psi):
    r'''
    Return :math:`\langle psi|psi \rangle`.

    Parameters
    ----------
    psi : dict
        Multi configurational state.

    '''
    return sum(abs(a)**2 for a in psi.values())
