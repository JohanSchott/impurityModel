#!/usr/bin/env python2

from math import sqrt
import numpy as np
from mpi4py import MPI

from finite import gauntC,applyOp,c2i,norm2,inner,add,getJobs

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

def getSpectra(hOp,tOps,psis,es,w,delta,krylovSize,slaterWeightMin,
               energyCut,restrictions=None):
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
    slaterWeightMin : float
        Restrict the number of product states by
        looking at |amplitudes|^2. 
    energyCut : float
        Restrict the number of considered states
    restrictions : dict
        Restriction the occupation of generated 
        product states.
    
    '''
    
    ## Relevant eigen energies  
    #esR = [e for e in es if e-es[0] < energyCut]
    ## Green's functions
    #gs = np.zeros((len(tOps),len(esR),len(w)),dtype=np.complex)
    ## Hamiltonian dict of the form  |PS> : {H|PS>} 
    #h = {}
    ## Loop over eigen states
    #for i in range(len(esR)):
    #    psi =  psis[i]
    #    e = esR[i]
    #    # Loop over transition operators
    #    for t,tOp in enumerate(tOps): 
    #        psiR = applyOp(tOp,psi,slaterWeightMin,restrictions)
    #        normalization = sqrt(norm2(psiR))
    #        for state in psiR.keys(): 
    #            psiR[state] /= normalization
    #        gs[t,i,:] = normalization**2*getGreen(e,psiR,hOp,w,delta,
    #                                              krylovSize,
    #                                              slaterWeightMin,
    #                                              restrictions,h)
    #return gs
    
    # Relevant eigen energies  
    esR = [e for e in es if e-es[0] < energyCut]
    n = len(esR)
    # Green's functions
    gs = np.zeros((len(tOps),n,len(w)),dtype=np.complex)
    g = {}
    # Hamiltonian dict of the form  |PS> : {H|PS>} 
    h = {}
    # Loop over eigen states, unique for each MPI rank
    for i in getJobs(rank,ranks,n):
        psi =  psis[i]
        e = esR[i]
        # Initialize Green's functions
        g[i] = np.zeros((len(tOps),len(w)),dtype=np.complex)
        # Loop over transition operators
        for t,tOp in enumerate(tOps): 
            psiR = applyOp(tOp,psi,slaterWeightMin,restrictions)
            normalization = sqrt(norm2(psiR))
            for state in psiR.keys(): 
                psiR[state] /= normalization
            g[i][t,:] = normalization**2*getGreen(e,psiR,hOp,w,delta,
                                                  krylovSize,
                                                  slaterWeightMin,
                                                  restrictions,h)
    # Distribute the Green's functions among the ranks
    for r in range(ranks):
        gTmp = comm.bcast(g, root=r)
        for i,gValue in gTmp.items():
            gs[:,i,:] = gValue
    return gs

def getGreen(e,psi,hOp,omega,delta,krylovSize,slaterWeightMin,
             restrictions=None,h=None):
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
    slaterWeightMin : float
        Restrict the number of product states by
        looking at |amplitudes|^2. 
    restrictions : dict
        Restriction the occupation of generated 
        product states.
    h : dict
        In and output argument.
        If present, the results of the operator hOp acting on each
        product state in the state psi is added and stored in this 
        variable. Format: |PS> : H|PS>,
        where |PS> is a product state and H|PS> is stored as a dictionary.

    '''

    print 'Start getGreen'

    # Allocations
    g = np.zeros(len(omega),dtype=np.complex)
    v = list(np.zeros(krylovSize))
    w = list(np.zeros(krylovSize))
    wp = list(np.zeros(krylovSize))
    alpha = np.zeros(krylovSize,dtype=np.float)
    beta = np.zeros(krylovSize-1,dtype=np.float)
    # Initialization 
    if h is None:
        h = {}
    v[0] = psi
    wp[0] = applyOp(hOp,v[0],slaterWeightMin,restrictions,h)
    alpha[0] = inner(wp[0],v[0]).real
    w[0] = add(wp[0],v[0],-alpha[0])
    
    # Approximate position of spectrum
    #print 'alpha[0]-E_i = {:5.1f}'.format(alpha[0]-e)

    # Construct Krylov states, 
    # and elements alpha and beta
    for j in range(1,krylovSize):
        beta[j-1] = sqrt(norm2(w[j-1]))
        #print 'beta[',j-1,'] = ',beta[j-1]
        if beta[j-1] != 0:
            v[j] = {s:1./beta[j-1]*a for s,a in w[j-1].items()}
        else:
            # Pick normalized state v[j],
            # orthogonal to v[0],v[1],v[2],...,v[j-1]
            print 'Warning: beta==0, implementation missing!'
        #print 'len(v[',j,'] =',len(v[j])
        wp[j] = applyOp(hOp,v[j],slaterWeightMin,restrictions,h)
        alpha[j] = inner(wp[j],v[j]).real
        w[j] = add(add(wp[j],v[j],-alpha[j]),v[j-1],-beta[j-1])

    # Construct Green's function from
    # continued fraction
    omegaP = omega+1j*delta+e
    for i in range(krylovSize-1,-1,-1):
        if i == krylovSize-1:
            g = 1./(omegaP - alpha[i]) 
        else:
            g = 1./(omegaP-alpha[i]-beta[i]**2*g)
    return g

