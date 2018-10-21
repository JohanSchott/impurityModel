#!/usr/bin/env python2

from math import pi,sqrt
import numpy as np
from sympy.physics.wigner import gaunt
import itertools
from bisect import bisect_left
from collections import OrderedDict

from py4rspt.unitarytransform import get_spherical_2_cubic_matrix 
from py4rspt.quantyt import thermal_average
from py4rspt.constants import k_B
#from removecreate import fortran

def getBasis(nBaths,valBaths,dnValBaths,dnConBaths,dnTol,n0imp):
    '''
    Return restricted basis of product states.

    '''
    
    # Sanity check
    for l in nBaths.keys():
        assert valBaths[l] <= nBaths[l]
        
    # Angular momentum
    l1,l2 = nBaths.keys()
    
    # For each partition, create all configurations
    # given the occupation in that partition.
    basisL = OrderedDict()
    for l in nBaths.keys():
        print 'l=',l
        # Add configurations to this list
        basisL[l] = []
        # Loop over different partion occupations
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

                    print 'New partition occupations:'
                    #print 'nImp,dnVal,dnCon = {:d},{:d},{:d}'.format(
                    #    nImp,dnVal,dnCon)
                    print 'nImp,nVal,nCon = {:d},{:d},{:d}'.format(
                        nImp,nVal,nCon)
                    # Impurity electrons
                    indices = range(c2i(nBaths,(l,-l,0)),
                                    c2i(nBaths,(l,l,1))+1)
                    basisImp = tuple(itertools.combinations(indices,nImp))
                    # Valence bath electrons
                    if valBaths[l] == 0:
                        # One way of having zero electrons in zero spin-orbitals
                        basisVal = ((),) 
                    else:
                        indices = range(c2i(nBaths,(l,-l,0,0)),
                                        c2i(nBaths,(l,l,1,valBaths[l]-1))+1)
                        basisVal = tuple(itertools.combinations(indices,nVal))
                    # Conduction bath electrons
                    if nBaths[l]-valBaths[l] == 0:
                        # One way of having zero electrons in zero spin-orbitals
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
    basis = []
    assert len(nBaths) == 2
    # This is only valid for two impurity blocks
    for b1 in basisL[l1]:
        for b2 in basisL[l2]:
            basis.append(tuple(sorted(b1+b2)))
    basis = tuple(basis)
    return basis


def binary_search(a, x):
    '''
    Return index to the leftmost value exactly equal to x.
    
    If x is not in the list, return -1.

    '''
    i = bisect_left(a, x)
    return i if i != len(a) and a[i] == x else -1

def binary_search_bigger(a, x):
    '''
    Return the index to the leftmost value bigger than than x, 
    if x is not in the list. 
        
    If x is in the list, return -1.
    
    '''
    i = bisect_left(a, x)
    return i if i == len(a) or a[i] != x else -1

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

def addToFirst(psi1,psi2,mul=1):
    r"""
    To state :math:`|\psi_1\rangle`, add  :math:`mul * |\psi_2\rangle`.
    
    Acknowledgement: Written entirely by Petter Saterskog.

    Parameters
    ----------
    psi1 : dict
    psi2 : dict
    mul : int, float or complex
        Optional

    """
    for s,a in psi2.items():
    	if s in psi1:
    		psi1[s]+=a*mul
    	else:
    		psi1[s]=a*mul

def c(i,psi):
    r'''
    Return :math:`|psi' \rangle = c_i |psi \rangle`.
    
    Acknowledgement: Written mostly by Petter Saterskog

    Parameters
    ----------
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
    for state,amp in psi.items():
        j = binary_search(state,i)
        if j != -1:
            cstate = state[:j] + state[j+1:]
            camp = amp if j%2==0 else -amp
            ret[cstate] = camp
    return ret

def cd(i,psi):
    r'''
    Return :math:`|psi' \rangle = c_i^\dagger |psi \rangle`.

    Acknowledgement: Written mostly by Petter Saterskog

    Parameters
    ----------
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
    for state,amp in psi.items():		
        j = binary_search_bigger(state,i)
        if j != -1:
    		camp = amp if j%2==0 else -amp
    		cstate = state[:j] + (i,) + state[j:]
    		ret[cstate] = camp
    return ret

def remove(i,state):
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
        amp = 1 if j%2==0 else -1
        return stateNew,amp
    else:
        return (),0

def removeList(i,state):
    '''
    Update state by removing electron at orbital i.

    Parameters
    ----------
    i : int
        Spin-orbital index
    state : list
        Product state.
        Elements are indices of occupied orbitals.
    
    Returns
    -------
    amp : int
        Amplitude

    ''' 
    j = binary_search(state,i)
    if j != -1:
        state.remove(i)
        amp = 1 if j%2==0 else -1
        return amp
    else:
        state[:] = []
        return 0

def create(i,state):
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
    j = binary_search_bigger(state,i)
    if j != -1:
        amp = 1 if j%2==0 else -1
        cstate = state[:j] + (i,) + state[j:]
        return cstate,amp
    else:
        return (),0

def createList(i,state):
    '''
    Update state by Creating electron at orbital i.

    Parameters
    ----------
    i : int
        Spin-orbital index
    state : list
        Product state.
        Elements are indices of occupied orbitals.

    Returns
    -------
    amp : int
        Amplitude

    ''' 
    j = binary_search_bigger(state,i)
    if j != -1:
        amp = 1 if j%2==0 else -1
        state.insert(j,i)
        return amp
    else:
        state[:] = []
        return 0

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
        print 'k={:d}'.format(k)
        for m in range(-l,l+1):
            s = ''
            for mp in range(-lp,lp+1):
                s += ' {:3.2f}'.format(gauntC(k,l,m,lp,mp))
            print s
        print

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
        for sOp,value in op.iteritems():
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

def getLz3d(nBaths,psi):
    r'''
    Return expectation value :math:`\langle psi| Lz_{3d} |psi \rangle`.
   
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets 
    psi : dict
        Multi configurational state.
    
    '''
    Lz = 0
    for state,amp in psi.items():
        tmp = 0
        for i in state:
            spinOrb = i2c(nBaths,i)
            if len(spinOrb)==3 and spinOrb[0]==2:
                tmp += spinOrb[1]
        Lz += tmp*abs(amp)**2
    return Lz

def getSz3d(nBaths,psi):
    r'''
    Return expectation value :math:`\langle psi| Sz_{3d} |psi \rangle`.
   
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets 
    psi : dict
        Multi configurational state.
    
    '''
    Sz = 0
    for state,amp in psi.items():
        tmp = 0
        for i in state:
            spinOrb = i2c(nBaths,i)
            if len(spinOrb)==3 and spinOrb[0]==2:
                tmp += -1/2. if spinOrb[2]==0 else 1/2.
        Sz += tmp*abs(amp)**2
    return Sz

def getSsqr3dWithBath(nBaths,psi,tol=1e-8):
    r'''
    Return expectation value :math:`\langle psi| S^2 |psi \rangle`.
   
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets 
    psi : dict
        normalized multi configurational state.
    
    '''
    psi1 = applySz3dWithBath(nBaths,psi)
    psi2 = applySplus3dWithBath(nBaths,psi)
    psi3 = applySminus3dWithBath(nBaths,psi)
    S2 = norm2(psi1)+1/2.*(norm2(psi2)+norm2(psi3))
    if S2.imag > tol:
        print 'Warning: <S^2> complex valued!'
    return S2.real    

def getSsqr3d(nBaths,psi,tol=1e-8):
    r'''
    Return expectation value :math:`\langle psi| S^2_{3d} |psi \rangle`.
   
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets 
    psi : dict
        normalized multi configurational state.
    
    '''
    psi1 = applySz3d(nBaths,psi)
    psi2 = applySplus3d(nBaths,psi)
    psi3 = applySminus3d(nBaths,psi)
    S2 = norm2(psi1)+1/2.*(norm2(psi2)+norm2(psi3))
    if S2.imag > tol:
        print 'Warning: <S^2> complex valued!'
    return S2.real    

def getLsqr3dWithBath(nBaths,psi,tol=1e-8):
    r'''
    Return expectation value :math:`\langle psi| L^2 |psi \rangle`.
   
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets 
    psi : dict
        normalized multi configurational state.

    '''
    psi1 = applyLz3dWithBath(nBaths,psi)
    psi2 = applyLplus3dWithBath(nBaths,psi)
    psi3 = applyLminus3dWithBath(nBaths,psi)
    L2 = norm2(psi1)+1/2.*(norm2(psi2)+norm2(psi3))
    if L2.imag > tol:
        print 'Warning: <L^2> complex valued!'
    return L2.real    

def getLsqr3d(nBaths,psi,tol=1e-8):
    r'''
    Return expectation value :math:`\langle psi| L^2_{3d} |psi \rangle`.
    
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets 
    psi : dict
        normalized multi configurational state.
    
    '''
    psi1 = applyLz3d(nBaths,psi)
    psi2 = applyLplus3d(nBaths,psi)
    psi3 = applyLminus3d(nBaths,psi)
    L2 = norm2(psi1)+1/2.*(norm2(psi2)+norm2(psi3))
    if L2.imag > tol:
        print 'Warning: <L^2> complex valued!'
    return L2.real    

def getTraceDensityMatrix(nBaths,psi,l=2):
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
    n = 0
    for state,amp in psi.items():
        nState = 0
        for m in range(-l,l+1):
            for s in range(2):
                i = c2i(nBaths,(l,m,s))
                if i in state:
                    nState += 1
        nState *= abs(amp)**2
        n += nState
    return n

def getDensityMatrix(nBaths,psi,l=2):
    r'''
    Return density matrix in spherical harmonics basis.

    :math:`n_{ij} = \langle i| \tilde{n} |j \rangle = \langle psi| c_j^\dagger c_i |psi \rangle`.
    
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
    densityMatrix = OrderedDict()
    for mi in range(-l,l+1):
            for mj in range(-l,l+1):
                for si in range(2):
                    for sj in range(2):
                        i = c2i(nBaths,(l,mi,si))
                        j = c2i(nBaths,(l,mj,sj))
                        tmp = inner(psi,cd(j,c(i,psi)))
                        if tmp != 0:
                            densityMatrix[((l,mi,si),(l,mj,sj))] = tmp
    return densityMatrix

def getDensityMatrixCubic(nBaths,psi):
    r'''
    Return density matrix in cubic harmonics basis.
    
    :math:`n_{ic,jc} = \langle ic| \tilde{n} |jc \rangle =  \langle psi| c_{jc}^\dagger c_{ic} |psi \rangle`,
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
        values of the form: :math:`\langle psi| c_{jc}^\dagger c_{ic} |psi \rangle`.

    '''
    # density matrix in spherical harmonics
    nSph = getDensityMatrix(nBaths,psi)
    l = 2
    # |i(cubic)> = sum_j u[j,i] |j(spherical)>
    u = get_spherical_2_cubic_matrix()     
    nCub = OrderedDict()
    for i in range(2*l+1):
        for j in range(2*l+1):
            for si in range(2):
                for sj in range(2):
                    for k,mk in enumerate(range(-l,l+1)):
                        for m,mm in enumerate(range(-l,l+1)):
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

def getEgT2gOccupation(nBaths,psi):
    r'''
    Return occupations of :math:`eg_\downarrow, eg_\uparrow, t2g_\downarrow, t2g_\uparrow` states.
    
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
                    v = u[j,i]*np.conj(u[k,i])*inner(psi,cd(jj,c(kk,psi)))
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
            print 'Warning: Complex occupation'
    return occs

def applySz3dWithBath(nBaths,psi):
    r'''
    Return :math:`|psi' \rangle = S^{z} |psi \rangle`. 
    
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state of 
        format tuple : amplitude
        where each tuple describes a
        Fock state.
    
    Returns
    -------
    psiNew : dict
        With the same format as psi. 
    
    '''
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        for s in range(2):
            # Impurity
            i = c2i(nBaths,(l,m,s))
            psiP = cd(i,c(i,psi))
            addToFirst(psiNew,psiP,1/2. if s==1 else -1/2.)
            # Bath
            for bathSet in range(nBaths[l]):
                i = c2i(nBaths,(l,m,s,bathSet))
                psiP = cd(i,c(i,psi))
                addToFirst(psiNew,psiP,1/2. if s==1 else -1/2.)
    return psiNew

def applySz3d(nBaths,psi):
    r'''
    Return :math:`|psi' \rangle = S^{z}_{3d} |psi \rangle`. 
    
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state of 
        format tuple : amplitude
        where each tuple describes a
        Fock state.
    
    Returns
    -------
    psiNew : dict
        With the same format as psi. 
    
    '''
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        for s in range(2):
            i = c2i(nBaths,(l,m,s))
            psiP = cd(i,c(i,psi))
            addToFirst(psiNew,psiP,1/2. if s==1 else -1/2.)
    return psiNew

def applyLz3dWithBath(nBaths,psi):
    r'''
    Return :math:`|psi' \rangle = L^{z} |psi \rangle`.
    
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state of 
        format tuple : amplitude
        where each tuple describes a
        Fock state.
    
    Returns
    -------
    psiNew : dict
        With the same format as psi. 
    
    '''
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        for s in range(2):
            # Impurity
            i = c2i(nBaths,(l,m,s))
            psiP = cd(i,c(i,psi))
            addToFirst(psiNew,psiP,m)
            # Bath
            for bathSet in range(nBaths[l]):
                i = c2i(nBaths,(l,m,s,bathSet))
                psiP = cd(i,c(i,psi))
                addToFirst(psiNew,psiP,m)
    return psiNew

def applyLz3d(nBaths,psi):
    r'''
    Return :math:`|psi' \rangle = L^{z}_{3d} |psi \rangle`.
    
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state of 
        format tuple : amplitude
        where each tuple describes a
        Fock state.
    
    Returns
    -------
    psiNew : dict
        With the same format as psi. 
    
    '''
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        for s in range(2):
            i = c2i(nBaths,(l,m,s))
            psiP = cd(i,c(i,psi))
            addToFirst(psiNew,psiP,m)
    return psiNew

def applySplus3dWithBath(nBaths,psi):
    r'''
    Return :math:`|psi' \rangle = S^{+} |psi \rangle`.
    
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state of 
        format tuple : amplitude
        where each tuple describes a
        Fock state.
    
    Returns
    -------
    psiNew : dict
        With the same format as psi. 
    
    '''
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        # Impurity
        i = c2i(nBaths,(l,m,1))
        j = c2i(nBaths,(l,m,0))
        psiP = cd(i,c(j,psi))
        # sQ = 1/2.
        # sqrt((sQ-(-sQ))*(sQ+(-sQ)+1)) == 1
        addToFirst(psiNew,psiP)
        # Bath
        for bathSet in range(nBaths[l]):
            i = c2i(nBaths,(l,m,1,bathSet))
            j = c2i(nBaths,(l,m,0,bathSet))
            psiP = cd(i,c(j,psi))
            addToFirst(psiNew,psiP)
    return psiNew

def applySplus3d(nBaths,psi):
    r'''
    Return :math:`|psi' \rangle = S^{+}_{3d} |psi \rangle`.
    
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state of 
        format tuple : amplitude
        where each tuple describes a
        Fock state.
    
    Returns
    -------
    psiNew : dict
        With the same format as psi. 
    
    '''
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        i = c2i(nBaths,(l,m,1))
        j = c2i(nBaths,(l,m,0))
        psiP = cd(i,c(j,psi))
        # sQ = 1/2.
        # sqrt((sQ-(-sQ))*(sQ+(-sQ)+1)) == 1
        addToFirst(psiNew,psiP)
    return psiNew

def applyLplus3dWithBath(nBaths,psi):
    r'''
    Return :math:`|psi' \rangle = L^{+} |psi \rangle`.
    
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state of 
        format tuple : amplitude
        where each tuple describes a
        Fock state.
    
    Returns
    -------
    psiNew : dict
        With the same format as psi. 

    '''
    psiNew = {}
    l = 2
    for m in range(-l,l):
        for s in range(2):
            # Impurity
            i = c2i(nBaths,(l,m+1,s))
            j = c2i(nBaths,(l,m,s))
            psiP = cd(i,c(j,psi))
            addToFirst(psiNew,psiP,sqrt((l-m)*(l+m+1)))
            # Bath
            for bathSet in range(nBaths[l]):
                i = c2i(nBaths,(l,m+1,s,bathSet))
                j = c2i(nBaths,(l,m,s,bathSet))
                psiP = cd(i,c(j,psi))
                addToFirst(psiNew,psiP,sqrt((l-m)*(l+m+1)))
    return psiNew

def applyLplus3d(nBaths,psi):
    r'''
    Return :math:`|psi' \rangle = L^{+}_{3d} |psi \rangle`.
    
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state of 
        format tuple : amplitude
        where each tuple describes a
        Fock state.
    
    Returns
    -------
    psiNew : dict
        With the same format as psi. 

    '''
    psiNew = {}
    l = 2
    for m in range(-l,l):
        for s in range(2):
            i = c2i(nBaths,(l,m+1,s))
            j = c2i(nBaths,(l,m,s))
            psiP = cd(i,c(j,psi))
            addToFirst(psiNew,psiP,sqrt((l-m)*(l+m+1)))
    return psiNew

def applySminus3dWithBath(nBaths,psi):
    r'''
    Return :math:`|psi' \rangle = S^{-} |psi \rangle`.
    
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state of 
        format tuple : amplitude
        where each tuple describes a
        Fock state.
    
    Returns
    -------
    psiNew : dict
        With the same format as psi. 
    
    '''
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        # Impurity
        i = c2i(nBaths,(l,m,0))
        j = c2i(nBaths,(l,m,1))
        psiP = cd(i,c(j,psi))
        # sQ = 1/2.
        # sqrt((sQ+sQ)*(sQ-sQ+1)) == 1
        addToFirst(psiNew,psiP)
        # Impurity
        for bathSet in range(nBaths[l]):
            i = c2i(nBaths,(l,m,0,bathSet))
            j = c2i(nBaths,(l,m,1,bathSet))
            psiP = cd(i,c(j,psi))
            addToFirst(psiNew,psiP)
    return psiNew

def applySminus3d(nBaths,psi):
    r'''
    Return :math:`|psi' \rangle = S^{-}_{3d} |psi \rangle`.
    
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state of 
        format tuple : amplitude
        where each tuple describes a
        Fock state.
    
    Returns
    -------
    psiNew : dict
        With the same format as psi. 
    
    '''
    psiNew = {}
    l = 2
    for m in range(-l,l+1):
        i = c2i(nBaths,(l,m,0))
        j = c2i(nBaths,(l,m,1))
        psiP = cd(i,c(j,psi))
        # sQ = 1/2.
        # sqrt((sQ+sQ)*(sQ-sQ+1)) == 1
        addToFirst(psiNew,psiP)
    return psiNew

def applyLminus3dWithBath(nBaths,psi):
    r"""
    Return :math:`|psi' \rangle = L^{-} |psi \rangle`.
    
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state of 
        format tuple : amplitude
        where each tuple describes a
        Fock state.
    
    Returns
    -------
    psiNew : dict
        With the same format as psi. 

    """
    psiNew = {}
    l = 2
    for m in range(-l+1,l+1):
        for s in range(2):
            # Impurity
            i = c2i(nBaths,(l,m-1,s))
            j = c2i(nBaths,(l,m,s))
            psiP = cd(i,c(j,psi))
            addToFirst(psiNew,psiP,sqrt((l+m)*(l-m+1)))
            # Bath
            for bathSet in range(nBaths[l]):
                i = c2i(nBaths,(l,m-1,s,bathSet))
                j = c2i(nBaths,(l,m,s,bathSet))
                psiP = cd(i,c(j,psi))
                addToFirst(psiNew,psiP,sqrt((l+m)*(l-m+1)))
    return psiNew

def applyLminus3d(nBaths,psi):
    r'''
    Return :math:`|psi' \rangle = L^{-}_{3d} |psi \rangle`. 
    
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    psi : dict
        Multi-configurational state of 
        format tuple : amplitude
        where each tuple describes a
        Fock state.
    
    Returns
    -------
    psiNew : dict
        With the same format as psi. 

    '''
    psiNew = {}
    l = 2
    for m in range(-l+1,l+1):
        for s in range(2):
            i = c2i(nBaths,(l,m-1,s))
            j = c2i(nBaths,(l,m,s))
            psiP = cd(i,c(j,psi))
            addToFirst(psiNew,psiP,sqrt((l+m)*(l-m+1)))
    return psiNew
    
def printExpValues(nBaths,es,psis,n=None):
    '''
    print several expectation values, e.g. E, N, L^2.
    '''
    if n == None:
        n = len(es)
    print 'E0 = {:5.2f}'.format(es[0])
    print ('  i  E-E0  N(3d) N(egDn) N(egUp) N(t2gDn) '
           'N(t2gUp) Lz(3d) Sz(3d) L^2(3d) S^2(3d) L^2(3d+B) S^2(3d+B)')
    for i,(e,psi) in enumerate(zip(es-es[0],psis)):
        if i < n:
            oc = getEgT2gOccupation(nBaths,psi)
            print ('{:3d} {:6.3f} {:5.2f} {:6.3f} {:7.3f} {:8.3f} {:7.3f}' 
                   ' {:7.2f} {:6.2f} {:7.2f} {:7.2f} {:8.2f} {:8.2f}').format(
                i,e,getTraceDensityMatrix(nBaths,psi),
                oc[0],oc[1],oc[2],oc[3],
                getLz3d(nBaths,psi),getSz3d(nBaths,psi),
                getLsqr3d(nBaths,psi),getSsqr3d(nBaths,psi),
                getLsqr3dWithBath(nBaths,psi),getSsqr3dWithBath(nBaths,psi))

def printThermalExpValues(nBaths,es,psis,T=300,cutOff=10):
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
    print '<E-E0> = {:4.3f}'.format(thermal_average(e,e,T=T))
    print '<N(3d)> = {:4.3f}'.format(thermal_average(
            e,
            [getTraceDensityMatrix(nBaths,psi) for psi in psis],
            T=T))
    occs = thermal_average(
        e,np.array([getEgT2gOccupation(nBaths,psi) for psi in psis]),
        T=T)
    print '<N(egDn)> = {:4.3f}'.format(occs[0])
    print '<N(egUp)> = {:4.3f}'.format(occs[1])
    print '<N(t2gDn)> = {:4.3f}'.format(occs[2])
    print '<N(t2gUp)> = {:4.3f}'.format(occs[3])
    print '<Lz(3d)> = {:4.3f}'.format(thermal_average(
            e,[getLz3d(nBaths,psi) for psi in psis],
            T=T))
    print '<Sz(3d)> = {:4.3f}'.format(thermal_average(
            e,[getSz3d(nBaths,psi) for psi in psis],
            T=T))
    print '<L^2(3d)> = {:4.3f}'.format(thermal_average(
            e,[getLsqr3d(nBaths,psi) for psi in psis],
            T=T))
    print '<S^2(3d)> = {:4.3f}'.format(thermal_average(
            e,[getSsqr3d(nBaths,psi) for psi in psis],
            T=T))


def applyOp(op,psi,slaterWeightMin=1e-12,restrictions=None,method='compact'):
    r'''
    Return :math:`|psi' \rangle = op |psi \rangle`. 
    
    Parameters
    ----------
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
        Multi-configurational state of 
        format tuple : amplitude
        where each tuple describes a
        Fock state.
    slaterWeightMin : float
        Restrict the number of product states by
        looking at |amplitudes|^2. 
    restrictions : dict
        Restriction the occupation of generated 
        product states.
    method : str
        Determine which way to calculate the result.
         
    Returns
    ------- 
    psiNew : dict
        New state of the same format as psi. 


    Note
    ----
    Different implementations exist.
    They return the same result, but calculations vary a bit.

    '''
    psiNew = {}
    if method == 'compact':
        for process,h in op.items():
            #assert h != 0
            # Initialize state
            psiA = psi
            for i,action in process[-1::-1]:
                if action == 'a':
                    psiB = c(i,psiA)     
                elif action == 'c':
                    psiB = cd(i,psiA)     
                psiA = psiB
            addToFirst(psiNew,psiB,h)
    elif method == 'newTuple':
        for process,h in op.items():
            #assert h != 0
            for state,amp in psi.items():
                #assert amp != 0
                # Initialize state
                sA = state
                signTot = 1
                for i,action in process[-1::-1]:
                    if action == 'a':
                        sB,sign = remove(i,sA)
                    elif action == 'c':
                        sB,sign = create(i,sA)
                    if sign == 0:
                        break
                    sA = sB
                    signTot *= sign
                else:
                    if sB in psiNew:
                        psiNew[sB] += h*amp*signTot
                    else:
                        psiNew[sB] = h*amp*signTot
    elif method == 'oneList':
        for process,h in op.items():
            assert h != 0
            for state,amp in psi.items():
                assert amp != 0
                # Initialize state
                s = list(state)
                signTot = 1
                for i,action in process[-1::-1]:
                    if action == 'a':
                        signTot *= removeList(i,s)
                    elif action == 'c':
                        signTot *= createList(i,s)
                    if signTot == 0:
                        break
                else:
                    # Convert back to tuple
                    s = tuple(s)
                    if s in psiNew:
                        psiNew[s] += h*amp*signTot
                    else:
                        psiNew[s] = h*amp*signTot
    else:
        print 'Warning: method not implemented.'
    if restrictions != None:
        psiTmp = {}
        for state,amp in psiNew.items():
            for restriction,occupations in restrictions.items():
                n = 0
                for i in restriction:
                    if i in state:
                        n += 1
                if n < occupations[0] or occupations[1] < n:
                    break
            else:
                psiTmp[state] = amp
        psiNew = psiTmp
    # Remove product states with small weight
    psiNew = {state:amp for state,amp in psiNew.items() if abs(amp)**2 > slaterWeightMin}
    return psiNew

def getHamiltonianMatrix(hOp,basis):
    '''
    return matrix Hamiltonian. 
    '''
    basisIndex = {basis[i]:i for i in range(len(basis))}
    h = np.zeros((len(basis),len(basis)),dtype=np.complex)
    print 'Filling the Hamiltonian...'
    progress = 0
    for j in range(len(basis)):
        if progress + 10 <= int(j*100./len(basis)): 
            progress = int(j*100./len(basis))
            print '{:d}% done'.format(progress)
        res = applyOp(hOp,{basis[j]:1})
        for k,v in res.items():
            if k in basisIndex:
                h[basisIndex[k],j] = v
    return h

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
    psi = {}
    for s,a in psi1.items():
        if s in psi:
            psi[s] += a
        else:
            psi[s] = a
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

