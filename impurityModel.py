#!/usr/bin/env python3

from math import pi,sqrt
import numpy as np
from sympy.physics.wigner import gaunt
from collections import OrderedDict
import itertools
import matplotlib.pylab as plt
import cProfile

from py4rspt.unitarytransform import get_spherical_2_cubic_matrix 
from py4rspt.quantyt import thermal_average
from py4rspt.constants import k_B
from py4rspt.rsptt import dc_MLFT
from removecreate import fortran

def main():
    
    #printGaunt()
    
    # -----------------------
    # System imformation  
    l1,l2 = 1,2 # Angular momentum
    # Number of bath sets
    nBaths = OrderedDict()
    nBaths[l1] = 0
    nBaths[l2] = 1
    # Number of valence bath sets
    valBaths = OrderedDict()
    valBaths[l1] = 0
    valBaths[l2] = 1
    # -----------------------
    # Occupation imformation. 
    # Angular momentum : initial impurity occupation 
    n0imp = OrderedDict()
    n0imp[l1] = 6 # 0 = empty, 2*(2*l1+1) = Full occupation
    n0imp[l2] = 8 # 8 for Ni+2
    # Angular momentum : max devation of initial impurity occupation
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
    Fdd=[7.5,0,9.9,0,6.6]
    Fpp=[0,0,0]
    Fpd=[8.9,0,6.8]
    Gpd=[0,5,0,2.8] 
    # SOC values
    xi_2p = 11.629
    xi_3d = 0.096
    # Double counting parameter
    chargeTransferCorrection = 1.5
    # Onsite 3d energy parameters
    eImp3d = -1.31796
    deltaO = 0.60422
    hz = 0.00001
    # Bath energies and hoppings for 3d orbitals
    eValEg = -4.4
    eValT2g = -6.5
    eConEg = 3
    eConT2g = 2
    vValEg = 1.883
    vValT2g = 1.395
    vConEg = 0.6
    vConT2g = 0.4
    # -----------------------
    # Slater determinant truncation parameters
    # Removes Slater determinants with weights
    # smaller than this optimization parameter.
    slaterWeightMin = 1e-7
    # -----------------------
    # Printing parameters
    nPrintExpValues = 30
    nPrintSlaterWeights = 0
    tolPrintOccupation = 1.1
    # -----------------------
    # Spectra parameters
    # Polarization vectors
    epsilons = [[1,0,0],[0,1,0],[0,0,1]] 
    # Temperature
    T = 300
    # How much above lowest eigenenergy to consider 
    energyCut = 10*k_B*T
    w = np.linspace(-10,15,1000)
    delta = 0.2
    krylovSize = 5
    # -----------------------
    
    # Hamiltonian
    print 'Construct the Hamiltonian operator...'
    hOp = getHamiltonianOperator(nBaths,valBaths,[Fdd,Fpp,Fpd,Gpd],
                                 [xi_2p,xi_3d],
                                 [n0imp,chargeTransferCorrection],
                                 [eImp3d,deltaO],hz,
                                 [vValEg,vValT2g,vConEg,vConT2g],
                                 [eValEg,eValT2g,eConEg,eConT2g])
    # Many body basis for the ground state
    print 'Create basis...'    
    basis = getBasis(nBaths,valBaths,dnValBaths,dnConBaths,
                     dnTol,n0imp)
    print '#basis states = {:d}'.format(len(basis)) 

    # Full diagonalization of restricted active space Hamiltonian
    print 'Create Hamiltonian matrix...' 
    h = getHamiltonianMatrix(hOp,basis)    
    print '<#Hamiltonian elements/column> = {:d}'.format(
        int(len(np.nonzero(h)[0])*1./len(basis)))     
    print 'Diagonalize the Hamiltonian...'
    es, vecs = np.linalg.eigh(h)
    print '{:d} eigenstates found!'.format(len(es))
    psis = [({basis[i]:vecs[i,vi] for i in range(len(basis)) 
              if slaterWeightMin <= abs(vecs[i,vi])**2 }) 
            for vi in range(len(es))]
    
    printThermalExpValues(nBaths,es,psis)
    printExpValues(nBaths,es,psis,n=nPrintExpValues) 

    
    print 'Create spectra...'
    # Dipole transition operators
    tOps = []
    for epsilon in epsilons:
        tOps.append(getDipoleOperator(nBaths,epsilon))
    # Green's function 
    gs = getSpectra(hOp,tOps,psis,es,w,delta,krylovSize,energyCut)
    
    # Sum over transition operators
    aTs = -np.sum(gs.imag,axis=0)
    # thermal average
    aAvg = thermal_average(es[:np.shape(aTs)[0]],aTs,T=T)
    print '#polarization = {:d}'.format(np.shape(gs)[0])
    print '#relevant eigenstates = {:d}'.format(np.shape(gs)[1])
    print '#mesh points = {:d}'.format(np.shape(gs)[2])
    # Save spectra to disk
    print 'Save spectra to disk...'
    tmp = [w,aAvg]
    # Each transition operator seperatly
    for i in range(np.shape(gs)[0]):
        a = thermal_average(es[:np.shape(gs)[1]],-np.imag(gs[i,:,:]))
        tmp.append(a)
    filename = 'output/spectra_krylovSize' + str(krylovSize) + '.dat'
    np.savetxt(filename,np.array(tmp).T,fmt='%8.4f',
               header='E  sum  T1  T2  T3 ...')

    # Print Slater determinants and weights 
    print 'Slater determinants and weights'
    weights = []
    for i,psi in enumerate(psis[:np.shape(gs)[1]]):
        print 'Eigenstate {:d}'.format(i)
        print 'Number of Slater determinants: {:d}'.format(len(psi))
        ws = np.array([ abs(a)**2 for s,a in psi.items() ])
        s = np.array([ s for s,a in psi.items() ])
        j = np.argsort(ws)
        ws = ws[j[-1::-1]]
        s = s[j[-1::-1]]
        weights.append(ws)
        print 'Highest weights:'
        print ws[:nPrintSlaterWeights]
        print 'Corresponding Slater determinantss:'
        print s[:nPrintSlaterWeights]
        print 
    ## Plot Slater determinant weigths
    #plt.figure()
    #for i,psi in range(np.shape(gs)[1]):
    #    plt.plot(weights[i],'-o',label=str(i))
    #plt.legend()
    #plt.show()

    # Test to calculate density matrix 
    print 'Density matrix (in cubic basis):'
    for i,psi in enumerate(psis[:np.shape(gs)[1]]):
        print 'Eigenstate {:d}'.format(i)
        n = getDensityMatrixCubic(nBaths,psi)
        print '#element={:d}'.format(len(n))
        for e,ne in n.items():
            if abs(ne) > tolPrintOccupation: 
                if e[0] == e[1]:
                    print 'Diagonal: (i,s)=',e[0],', occupation = {:7.2f}'.format(ne) 
                else:
                    print 'Off-diagonal: (i,si),(j,sj)=',e,', {:7.2f}'.format(ne) 
        print  

    # Plot spectra
    
    ## Plot all spectra
    #plt.figure()
    #for t in range(np.shape(gs)[0]):
    #    for i in range(np.shape(gs)[1]):
    #        plt.plot(w,-gs[t,i,:].imag,
    #                 label='t={:d},i={:d}'.format(t,i))
    #plt.xlabel('w')
    #plt.ylabel('intensity')
    #plt.legend()
    #plt.tight_layout()
    #plt.show()
    #
    #
    ## Plot spectra for each transition operator
    #plt.figure()
    #plt.plot(w,aAvg,'-k',label='sum')
    #for i in range(np.shape(gs)[0]):
    #    a = thermal_average(es[:np.shape(gs)[1]],-np.imag(gs[i,:,:]))
    #    plt.plot(w,a,label='T_{:d}'.format(i))
    #plt.xlabel('w')
    #plt.ylabel('intensity')
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

    ## Plot spectra for each eigenstate
    #plt.figure()
    #plt.plot(w,aAvg,'-k',label='avg')
    #for i in range(np.shape(aTs)[0]):
    #    plt.plot(w,aTs[i,:],label='eig_{:d}'.format(i))
    #plt.xlabel('w')
    #plt.ylabel('intensity')
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

    print('Script finished.')

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
    
    Acknowledgement: Written entirely by Petter Saterskog

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
    	for j in range(len(state)):
    		if i==state[j]:
    			cstate=state[:j] + state[j+1:]
    			camp=amp if j%2==0 else -amp
    			# if cstate in ret:
    			# 	ret[cstate]+=camp
    			# else:
    			ret[cstate]=camp
    			break
    return ret

def cd(i,psi):
    r'''
    Return :math:`|psi' \rangle = c_i^\dagger |psi \rangle`.
    
    Acknowledgement: Written entirely by Petter Saterskog

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
    	ip=len(state)
    	for j in range(len(state)):
    		p=state[j]
    		if i==p:
    			ip=-1
    			break
    		if i<p:
    			ip=j
    			break
    	if ip!=-1:
    		camp=amp if ip%2==0 else -amp
    		cstate=state[:ip] + (i,) + state[ip:]
    		# if cstate in ret:
    		# 	ret[cstate]+=camp
    		# else:
    		ret[cstate]=camp
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
    if i in state:
        j = state.index(i)
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
    if i in state:
        j = state.index(i)
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
    if i in state:
        return (),0
    else:
    	ip = len(state)
    	for j in range(len(state)):
    		p = state[j]
    		if i<p:
    			ip=j
    			break
        amp = 1 if ip%2==0 else -1
        stateNew = state[:ip] + (i,) + state[ip:]
        return stateNew,amp

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
    if i in state:
        state[:] = []
        return 0
    else:
    	ip = len(state)
    	for j in range(len(state)):
    		p = state[j]
    		if i<p:
    			ip=j
    			break
        amp = 1 if ip%2==0 else -1
        state.insert(ip,i)
        return amp

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
        (sorb1,sorb2,sorb3,sorb4) : u/2
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
                                proccess = ((l1,m1,s),(l2,m2,sp),
                                            (l3,m3,sp),(l4,m4,s))
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
    FddOpp = getUop(l1=2,l2=2,l3=2,l4=2,R=Fdd)
    # Calculate F_pp^{0,2}
    FppOpp = getUop(l1=1,l2=1,l3=1,l4=1,R=Fpp)
    # Calculate F_pd^{0,2}
    FpdOpp1 = getUop(l1=1,l2=2,l3=2,l4=1,R=Fpd)
    FpdOpp2 = getUop(l1=2,l2=1,l3=1,l4=2,R=Fpd)
    FpdOpp = addOps([FpdOpp1,FpdOpp2])
    # Calculate G_pd^{1,3}
    GpdOpp1 = getUop(l1=1,l2=2,l3=1,l4=2,R=Gpd)
    GpdOpp2 = getUop(l1=2,l2=1,l3=2,l4=1,R=Gpd)
    GpdOpp = addOps([GpdOpp1,GpdOpp2])
    # Add operators
    uOpp = addOps([FddOpp,FppOpp,FpdOpp,GpdOpp])
    return uOpp

def getSOCopp(xi,l=2):
    '''
    Return SOC operator for one l-shell.
    '''
    oppDict = {}
    for m in range(-l,l+1):
        for s in range(2):
            value = xi*m*(1/2. if s==1 else -1/2.)
            oppDict[((l,m,s),(l,m,s))] = value  
    for m in range(-l,l):
        value = xi/2.*sqrt((l-m)*(l+m+1))
        oppDict[((l,m,1),(l,m+1,0))] = value
        oppDict[((l,m+1,0),(l,m,1))] = value
    return oppDict


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
    print ('i  E-E0  N(3d) N(egDn) N(egUp) N(t2gDn) '
           'N(t2gUp) Lz(3d) Sz(3d) L^2(3d) S^2(3d) L^2(3d+B) S^2(3d+B)')
    for i,(e,psi) in enumerate(zip(es-es[0],psis)):
        if i < n:
            oc = getEgT2gOccupation(nBaths,psi)
            print ('{:d} {:6.3f} {:5.2f} {:6.3f} {:7.3f} {:8.3f} {:7.3f}' 
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


def applyOp(op,psi,method='newTuple'):
    r'''
    Return :math:`|psi' \rangle = op |psi \rangle`. 
    
    Parameters
    ----------
    op : dict
        Operator of the format
        tuple : amplitude,
        where each tuple describes a 
        (one or two particle) scattering
        process.
    psi : dict
        Multi-configurational state of 
        format tuple : amplitude
        where each tuple describes a
        Fock state.
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
    if method == 'compact':
        psiNew = {}
        for i,h in op.items():
            if len(i) == 4:
                psiP = cd(i[0],cd(i[1],c(i[2],c(i[3],psi))))
                addToFirst(psiNew,psiP,h)
            elif len(i) == 2:
                psiP = cd(i[0],c(i[1],psi))
                addToFirst(psiNew,psiP,h)
            else:
                print 'Warning: Strange operator!'
    elif method == 'newTuple':
        psiNew = {}
        for i,h in op.items():
            assert h != 0
            if len(i) == 4:
                for s0,A0 in psi.items():
                    assert A0 != 0
                    # Remove electron
                    s1,A1 = remove(i[3],s0)
                    if A1 == 0:
                        continue
                    # Remove electron
                    s2,A2 = remove(i[2],s1)
                    if A2 == 0:
                        continue
                    # Create electron
                    s3,A3 = create(i[1],s2)
                    if A3 == 0:
                        continue
                    # Create electron
                    s4,A4 = create(i[0],s3)
                    if A4 == 0:
                        continue
                    # Add state to return variable
                    if s4 in psiNew:
                        psiNew[s4] += h*A0*A1*A2*A3*A4
                    else:
                        psiNew[s4] = h*A0*A1*A2*A3*A4
            elif len(i) == 2:
                for s0,A0 in psi.items():
                    assert A0 != 0
                    # Remove electron
                    s1,A1 = remove(i[1],s0)
                    if A1 == 0:
                        continue
                    # Create electron
                    s2,A2 = create(i[0],s1)
                    if A2 == 0:
                        continue
                    # Add state to return variable
                    if s2 in psiNew:
                        psiNew[s2] += h*A0*A1*A2
                    else:
                        psiNew[s2] = h*A0*A1*A2
            else:
                print 'Warning: Strange operator!'
    elif method == 'oneList':
        psiNew = {}
        for i,h in op.items():
            assert h != 0
            if len(i) == 4:
                for s0,A0 in psi.items():
                    assert A0 != 0
                    # Initialize state
                    s = list(s0)
                    amp = A0
                    # Remove electron
                    amp *= removeList(i[3],s)
                    if amp == 0:
                        continue
                    # Remove electron
                    amp *= removeList(i[2],s)
                    if amp == 0:
                        continue
                    # Create electron
                    amp *= createList(i[1],s)
                    if amp == 0:
                        continue
                    # Create electron
                    amp *= createList(i[0],s)
                    if amp == 0:
                        continue
                    # Convert back to tuple
                    s = tuple(s)
                    # Add state to return variable
                    if s in psiNew:
                        psiNew[s] += h*amp
                    else:
                        psiNew[s] = h*amp
            elif len(i) == 2:
                for s0,A0 in psi.items():
                    assert A0 != 0
                    # Initialize state
                    s = list(s0)
                    amp = A0
                    # Remove electron
                    amp *= removeList(i[1],s)
                    if amp == 0:
                        continue
                    # Create electron
                    amp *= createList(i[0],s)
                    if amp == 0:
                        continue
                    # Convert back to tuple
                    s = tuple(s)
                    # Add state to return variable
                    if s in psiNew:
                        psiNew[s] += h*amp
                    else:
                        psiNew[s] = h*amp
            else:
                print 'Warning: Strange operator!'
    elif method == 'fortran':
        psiNew = {}
        for i,h in op.items():
            assert h != 0
            if len(i) == 4:
                for s0,A0 in psi.items():
                    assert A0 != 0
                    # Initialize fortran variables
                    fortran.s = s0
                    fortran.amp = 1
                    # Remove electron
                    fortran.remove(i[3])
                    if fortran.amp == 0:
                        continue
                    # Remove electron
                    fortran.remove(i[2])
                    if fortran.amp == 0:
                        continue
                    # Transfer back result to Python
                    s2 = tuple(fortran.s)
                    # Create electron
                    s3,A3 = create(i[1],s2)
                    if A3 == 0:
                        continue
                    # Create electron
                    s4,A4 = create(i[0],s3)
                    if A4 == 0:
                        continue
                    # Add state to return variable
                    if s4 in psiNew:
                        psiNew[s4] += h*A0*fortran.amp*A3*A4
                    else:
                        psiNew[s4] = h*A0*fortran.amp*A3*A4
            elif len(i) == 2:
                for s0,A0 in psi.items():
                    assert A0 != 0
                    # Initialize fortran variables
                    fortran.s = s0
                    fortran.amp = 1
                    # Remove electron
                    fortran.remove(i[1])
                    if fortran.amp == 0:
                        continue
                    # Transfer back result to Python
                    s1 = tuple(fortran.s)
                    # Create electron
                    s2,A2 = create(i[0],s1)
                    if A2 == 0:
                        continue
                    # Add state to return variable
                    if s2 in psiNew:
                        psiNew[s2] += h*A0*fortran.amp*A2
                    else:
                        psiNew[s2] = h*A0*fortran.amp*A2
            else:
                print 'Warning: Strange operator!'
    else:
        print 'Warning: method not implemented.'
    return psiNew
    
def getHamiltonianOperator(nBaths,valBaths,slaterCondon,SOCs,
                           DCinfo,impurityInfo,hz,
                           vHoppings,eBaths):
    '''
    Return the Hamiltonian, in operator form.
    
    Parameters
    ----------
    nBaths : dict
        Number of bath sets for each angular momentum.
    nBaths : dict
        Number of valence bath sets for each angular momentum.
    slaterCondon : list
        List of Slater-Condon parameters.
    SOCs : list
        List of SOC parameters.
    DCinfo : list
        Contains information needed for the double counting energy.
    impurityInfo : list
        Contains information of 3d single particle energies.
    hz : float
        External magnetic field.  
    vHoppings : list
        Contains information about hybridization hoppings.
    eBaths : list
        Contains information about bath energies.

    '''
    
    # Divide up input parameters to more concrete variables 
    Fdd,Fpp,Fpd,Gpd = slaterCondon
    xi_2p,xi_3d = SOCs
    n0imp,chargeTransferCorrection = DCinfo
    eImp3d,deltaO = impurityInfo
    vValEg,vValT2g,vConEg,vConT2g = vHoppings
    eValEg,eValT2g,eConEg,eConT2g = eBaths
    
    # Angular momentum
    l1,l2 = nBaths.keys()
     
    # Calculate U operator  
    uOpperator = get2p3dSlaterCondonUop(Fdd=Fdd,Fpp=Fpp,
                                         Fpd=Fpd,Gpd=Gpd)
    # Add SOC 
    SOC2pOperator = getSOCopp(xi_2p,l=l1)
    SOC3dOperator = getSOCopp(xi_3d,l=l2)
    
    # Double counting (DC) correction
    # MLFT DC 
    dc = dc_MLFT(n3d_i=n0imp[l2],c=chargeTransferCorrection,Fdd=Fdd,
                 n2p_i=n0imp[l1],Fpd=Fpd,Gpd=Gpd)
    eDCOperator = {}
    for il,l in enumerate([2,1]):
        for m in range(-l,l+1):
            for s in range(2):
                eDCOperator[((l,m,s),(l,m,s))] = -dc[il]
    
    # Calculate impurity 3d Hamiltonian
    # (Either by reading matrix or parameterize it)
    eImpEg = eImp3d + 3./5*deltaO
    eImpT2g = eImp3d - 2./5*deltaO  
    hImp3d = np.zeros((2*l2+1,2*l2+1))
    np.fill_diagonal(hImp3d,(eImpEg,eImpEg,eImpT2g,eImpT2g,eImpT2g)) 
    # Convert to spherical harmonics basis
    u = get_spherical_2_cubic_matrix(spinpol=False,l=l2)
    hImp3d = np.dot(u,np.dot(hImp3d,np.conj(u.T)))
    # Convert from matrix to operator form
    # Also add spin
    hImp3dOperator = {}
    for i,mi in enumerate(range(-l2,l2+1)):
        for j,mj in enumerate(range(-l2,l2+1)):
            if hImp3d[i,j] != 0:
                for s in range(2):
                    hImp3dOperator[((l2,mi,s),(l2,mj,s))] = hImp3d[i,j]
    
    # Magnetic field
    hHzOperator = {}
    for m in range(-l2,l2+1):
        for s in range(2):
            hHzOperator[((l2,m,s),(l2,m,s))] = hz*1/2. if s==1 else -hz*1/2.        
    
    # Bath (3d) on-site energies and hoppings
    # Calculate hopping terms between bath and (3d) impurity
    # either by reading matrix or parameterize it
    vVal3d = np.zeros((2*l2+1,2*l2+1))
    vCon3d = np.zeros((2*l2+1,2*l2+1))
    eBathVal3d = np.zeros((2*l2+1,2*l2+1))
    eBathCon3d = np.zeros((2*l2+1,2*l2+1))
    np.fill_diagonal(vVal3d,(vValEg,vValEg,vValT2g,vValT2g,vValT2g))
    np.fill_diagonal(vCon3d,(vConEg,vConEg,vConT2g,vConT2g,vConT2g))
    np.fill_diagonal(eBathVal3d,(eValEg,eValEg,eValT2g,eValT2g,eValT2g))
    np.fill_diagonal(eBathCon3d,(eConEg,eConEg,eConT2g,eConT2g,eConT2g))
    # Convert to spherical harmonics basis
    vVal3d = np.dot(u,np.dot(vVal3d,np.conj(u.T)))
    vCon3d = np.dot(u,np.dot(vCon3d,np.conj(u.T)))
    eBathVal3d = np.dot(u,np.dot(eBathVal3d,np.conj(u.T)))
    eBathCon3d = np.dot(u,np.dot(eBathCon3d,np.conj(u.T)))
    # Convert from matrix to operator form
    # Also add spin
    hHoppOperator = {}
    eBath3dOperator = {}
    for bathSet in range(nBaths[l2]):
        for i,mi in enumerate(range(-l2,l2+1)):
            for j,mj in enumerate(range(-l2,l2+1)):
                if bathSet in range(valBaths[l2]):
                    vHopp = vVal3d[i,j]
                    eBath = eBathVal3d[i,j]
                else:
                    vHopp = vCon3d[i,j]
                    eBath = eBathCon3d[i,j]
                if vHopp != 0:
                    for s in range(2):
                        hHoppOperator[((l2,mi,s),(l2,mj,s,bathSet))] = vHopp
                        hHoppOperator[((l2,mj,s,bathSet),(l2,mi,s))] = vHopp.conjugate()
                if eBath != 0:
                    for s in range(2):
                        eBath3dOperator[((l2,mi,s,bathSet),(l2,mj,s,bathSet))] = eBath
    
    # Add Hamiltonian terms to one operator 
    hOperator = addOps([uOpperator,
                         hImp3dOperator,
                         hHzOperator,
                         SOC2pOperator,
                         SOC3dOperator,
                         eDCOperator,
                         hHoppOperator,
                         eBath3dOperator])
    # Convert spin-orbital indices to a single index
    hOp = {}
    for op,value in hOperator.items():
        hOp[tuple(c2i(nBaths,spinOrb) for spinOrb in op)] = value

    return hOp

def getDipoleOperator(nBaths,n):
    r'''
    Return dipole transition operator :math:`\hat{T}`.
    
    Transition between states of different angular momentum,
    defined by the keys in the nBaths dictionary.
    
    Parameters
    ----------
    nBaths : dict
        angular momentum: number of bath sets
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
                        tOp[(i,j)] = tij                      
    return tOp

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

def getBasis(nBaths,valBaths,dnValBaths,dnConBaths,dnTol,n0imp):
    
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

def getSpectra(hOp,tOps,psis,es,w,delta,krylovSize,energyCut):
    r'''
    Return Green's function for states with low enough energy.
    
    For states :math:`|psi \rangle` with e < e[0] + energyCut, calculate: 

    :math:`g(w+1j*delta) = 
    = \langle psi| tOp^\dagger ((w+1j*delta+e)*\hat{1} - hOp)^{-1} tOp 
    |psi \rangle`,

    where :math:`e = \langle psi| hOp |psi \rangle` 
    
    Lanczos algorithm is used.
     
    Parameters
    ----------
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
        Deviation from real axis
    krylovSize : int
        Size of the Krylov space
    energyCut : float
        Restrict the number of considered states
    
    '''
    # Relevant eigen energies  
    esR = [e for e in es if e-es[0] < energyCut]
    # Green's functions
    gs = np.zeros((len(tOps),len(esR),len(w)),dtype=np.complex)
    # Loop over transition operators
    for t,tOp in enumerate(tOps): 
        psisR = [applyOp(tOp,psi) for psi in psis[:len(esR)]]  
        normalizations = [sqrt(norm2(psi)) for psi in psisR]
        for i in range(len(psisR)):
            for state in psisR[i].keys(): 
                psisR[i][state] /= normalizations[i] 
        for i,(e,psi) in enumerate(zip(esR,psisR)):
            gs[t,i,:] = getGreen(e,psi,hOp,w,delta,krylovSize)
    return gs

def getGreen(e,psi,hOp,omega,delta,krylovSize):
    r'''
    return Green's function 
    :math:`\langle psi|((omega+1j*delta+e)\hat{1} - hOp)^{-1} |psi \rangle`.
    
    Parameters
    ----------
    e : float
        Total energy
    psi : dict
        Multi state
    hOp : dict
        Operator
    omega : list 
        Real axis energy mesh
    delta : float
        Deviation from real axis
    krylovSize : int
        Size of the Krylov space

    '''
    # Allocations
    g = np.zeros(len(omega),dtype=np.complex)
    v = list(np.zeros(krylovSize))
    w = list(np.zeros(krylovSize))
    wp = list(np.zeros(krylovSize))
    alpha = np.zeros(krylovSize,dtype=np.float)
    beta = np.zeros(krylovSize-1,dtype=np.float)
    # Initialization 
    v[0] = psi
    wp[0] = applyOp(hOp,v[0])
    alpha[0] = inner(wp[0],v[0]).real
    w[0] = add(wp[0],v[0],-alpha[0])
    
    # Approximate position of spectrum
    #print 'alpha[0]-E_i = {:5.1f}'.format(alpha[0]-e)

    # Construct Krylov states, 
    # and elements alpha and beta
    for j in range(1,krylovSize):
        beta[j-1] = sqrt(norm2(w[j-1]))
        if beta[j-1] != 0:
            v[j] = {s:1./beta[j-1]*a for s,a in w[j-1].items()}
        else:
            # Pick normalized state v[j],
            # orthogonal to v[0],v[1],v[2],...,v[j-1]
            print 'Warning: beta==0, implementation missing!'
        wp[j] = applyOp(hOp,v[j])
        alpha[j] = inner(wp[j],v[j]).real
        w[j] = add(add(wp[j],v[j],-alpha[j]),v[j-1],-beta[j-1])

    # Construct Green's function from
    # continued fraction
    omegaP = omega+1j*delta+e
    for i in range(krylovSize-1,-1,-1):
        if i == krylovSize-1:
            g = 1./(omegaP - alpha[i]) 
        elif i == 0:
            g = 1./(omegaP-alpha[i]-beta[i]**2*g)
        else:
            g = 1./(omegaP-alpha[i]-beta[i]**2*g)
    return g



if __name__== "__main__":
    #main()
    cProfile.run('main()',sort='cumulative')

