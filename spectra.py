#!/usr/bin/env python3

from math import sqrt
import numpy as np
from mpi4py import MPI
import scipy.sparse
import scipy.sparse.linalg

from finite import gauntC,c2i,getJobs
from finite import daggerOp,applyOp,inner,add,norm2,expandBasisAndHamiltonian

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

def getGreen(e,psi,hOp,omega,delta,krylovSize,slaterWeightMin,
             restrictions=None,hBig=None,mode='numpy'):
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
        Deviation from real axis.
        Broadening/resolution parameter.
    krylovSize : int
        Size of the Krylov space
    slaterWeightMin : float
        Restrict the number of product states by
        looking at |amplitudes|^2. 
    restrictions : dict
        Restriction the occupation of generated 
        product states.
    hBig : dict
        In and output argument.
        If present, the results of the operator hOp acting on each
        product state in the state psi is added and stored in this 
        variable. Format: |PS> : H|PS>,
        where |PS> is a product state and H|PS> is stored as a dictionary.
    mode : str
        'dict' or 'numpy'. Determines which algorithm to use.

    '''

    #print('Start getGreen')

    # Allocations
    g = np.zeros(len(omega),dtype=np.complex)
    alpha = np.zeros(krylovSize,dtype=np.float)
    beta = np.zeros(krylovSize-1,dtype=np.float)
    # Initialization 
    if hBig is None:
        hBig = {}
    if mode == 'dict':
        v = list(np.zeros(krylovSize))
        w = list(np.zeros(krylovSize))
        wp = list(np.zeros(krylovSize))
        v[0] = psi
        #print('len(hBig) = ',len(hBig),', len(v[0]) = ',len(v[0]))
        wp[0] = applyOp(hOp,v[0],slaterWeightMin,restrictions,hBig)
        #print('#len(hBig) = ',len(hBig),', len(wp[0]) = ',len(wp[0]))
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
            wp[j] = applyOp(hOp,v[j],slaterWeightMin,restrictions,hBig)
            alpha[j] = inner(wp[j],v[j]).real
            w[j] = add(add(wp[j],v[j],-alpha[j]),v[j-1],-beta[j-1])
            #print('len(hBig) = ',len(hBig),', len(w[j]) = ',len(w[j]))
    elif mode == 'numpy':
        # Hamiltonian in dict format.
        # Possibly new product state keys are added to hBig.
        h = expandBasisAndHamiltonian(hBig,hOp,psi.keys(),restrictions)
        index = {ps:i for i,ps in enumerate(h.keys())} 
        basis = {i:ps for i,ps in enumerate(h.keys())} 
        # Number of basis states
        n = len(h)
        # Express Hamiltonian in matrix form
        hValues, rows, cols = [],[],[]
        for psJ,res in h.items():
            for psI,hValue in res.items():
                hValues.append(hValue)
                rows.append(index[psI])
                cols.append(index[psJ])
        # store Hamiltonian in sparse matrix form
        h = scipy.sparse.csr_matrix((hValues,(rows,cols)),shape=(n,n))
        # Store all Krylov state vectors.
        # Not really needed but so far memory has not been a probelem.
        v = np.zeros((krylovSize,n),dtype=np.complex)
        w = np.zeros((krylovSize,n),dtype=np.complex)
        wp = np.zeros((krylovSize,n),dtype=np.complex)
        # Express psi as a vector
        for ps,amp in psi.items():
            v[0,index[ps]] = amp
        wp[0,:] = h.dot(v[0,:])
        alpha[0] = np.dot(np.conj(wp[0,:]),v[0,:]).real
        w[0,:] = wp[0,:] - alpha[0]*v[0,:]
        
        # Construct Krylov states, 
        # and elements alpha and beta
        for j in range(1,krylovSize):
            beta[j-1] = sqrt(np.sum(np.abs(w[j-1,:])**2))
            if beta[j-1] != 0:
                v[j,:] = w[j-1,:]/beta[j-1]
            else:
                # Pick normalized state v[j],
                # orthogonal to v[0],v[1],v[2],...,v[j-1]
                raise ValueError('Warning: beta==0, implementation missing!')
            wp[j,:] = h.dot(v[j,:])
            alpha[j] = np.dot(np.conj(wp[j,:]),v[j,:]).real
            w[j,:] = wp[j,:] - alpha[j]*v[j,:] - beta[j-1]*v[j-1,:]

    # Construct Green's function from
    # continued fraction
    omegaP = omega+1j*delta+e
    for i in range(krylovSize-1,-1,-1):
        if i == krylovSize-1:
            g = 1./(omegaP - alpha[i]) 
        else:
            g = 1./(omegaP-alpha[i]-beta[i]**2*g)
    return g

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
        Deviation from real axis.
        Broadening/resolution parameter.
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
    
    # Relevant eigen energies  
    esR = [e for e in es if e-es[0] < energyCut]
    n = len(esR)
    # Green's functions
    gs = np.zeros((n,len(tOps),len(w)),dtype=np.complex)
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
            gs[i,:,:] = gValue
    return gs

def getRIXSmap(hOp,tOpsIn,tOpsOut,psis,es,wIns,wLoss,delta1,delta2,krylovSize,
               slaterWeightMin,energyCut,restrictions,hGround=None,
               parallelizationMode='wIn'):
    r"""
    Return RIXS Green's function for states with low enough energy.
    
    For states :math:`|psi \rangle` with e < e[0] + energyCut, calculate: 

    :math:`g(w+1j*delta) = 
    = \langle psi| ROp^\dagger ((wLoss+1j*delta2+e)*\hat{1} - hOp)^{-1} ROp 
    |psi \rangle`,

    where :math:`e = \langle psi| hOp |psi \rangle`, and

    :math:`Rop = tOpOut ((wIns+1j*delta1+e)*\hat{1} - hOp)^{-1} tOpIn`.

    Calculations are performed according to:
    1) Calculate state |psi1> = tOpIn |psi>
    2) Calculate state |psi2> = ((wIns+1j*delta1+e)*\hat{1} - hOp)^{-1}|psi1> 
        This is done by introducing operator A = (wIns+1j*delta1+e)*\hat{1} - hOp.
        By applying A from the left on |psi2> = A^{-1}|psi1> gives 
        the inverse problem: A|psi2> = |psi1>.
        This equation can be solved by guessing |psi2> and iteratively 
        improving it.
    3) Calculate state |psi3> = tOpOut |psi2>
    4) Calculate normalization = sqrt(<psi3|psi3>)
    5) Normalize psi3 according to: psi3 /= normalization
    6) Now the Green's function is given by:
        :math:`g(wLoss+1j*delta2) = 
        = normalization^2 * \langle psi3| ((wLoss+1j*delta2+e)*\hat{1} - hOp)^{-1} |psi3 \rangle`,
        which can efficiently be evaluation using Lanczos.    
     
    Parameters
    ----------
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
    hGround : dict
        Optional. The Hamiltonian for product states with no core hole.
        If present, it will be updated by this function.
    parallelizationMode : str
        'serial', 'eigenStates' or 'wIn'
    
    """
    if hGround is None:
        hGround = {}
    # Relevant eigen energies  
    esR = [e for e in es if e-es[0] < energyCut]
    nE = len(esR)
    # Green's functions
    gs = np.zeros((nE,len(tOpsIn),len(tOpsOut),len(wIns),len(wLoss)),
                  dtype=np.complex)
    # Hamiltonian dict of the form  |PS> : {H|PS>} 
    # For product states with a core hole.
    hExcited = {}
    
    if parallelizationMode == 'serial':
        # Loop over eigen states
        for iE in range(nE):
            psi =  psis[iE]
            e = esR[iE]
            # Loop over in-coming transition operators
            for tIn,tOpIn in enumerate(tOpsIn): 
                # Core-hole state
                psi1 = applyOp(tOpIn,psi,slaterWeightMin,restrictions)
                # Hamiltonian acting on relevant product states. |PS> : {H|PS>}
                nTMP = len(hExcited)
                h = expandBasisAndHamiltonian(hExcited,hOp,psi1.keys(),restrictions)
                if rank == 0:
                    print('len(psi1),len(h),len(hExcited),#elements added to hExcited')
                    print(len(psi1),len(h),len(hExcited),len(hExcited)-nTMP)
                index = {ps:i for i,ps in enumerate(h.keys())} 
                basis = {i:ps for i,ps in enumerate(h.keys())} 
                n = len(h)
                # Express psi1 as a vector
                y = np.zeros(n,dtype=np.complex)
                for ps,amp in psi1.items():
                    y[index[ps]] = amp
                # store psi1 as a sparse vector
                #y = scipy.sparse.csr_matrix(y)
                # Express Hamiltonian in matrix form
                hValues, rows, cols = [],[],[]
                for psJ,res in h.items():
                    for psI,hValue in res.items():
                        hValues.append(hValue)
                        rows.append(index[psI])
                        cols.append(index[psJ])
                # store Hamiltonian in sparse matrix form
                h = scipy.sparse.csr_matrix((hValues,(rows,cols)),shape=(n,n))
                if rank == 0: print('Loop over in-coming photon energies...')
                for iwIn,wIn in enumerate(wIns):
                    # A = (wIn+1j*delta1+e)*\hat{1} - hOp.
                    a = scipy.sparse.csr_matrix(([wIn+1j*delta1+e]*n,(range(n),range(n))),
                                                shape=(n,n))
                    a -= h
                    # Find x by solving: a*x = y            
                    # Biconjugate gradient stabilized method.
                    # Pure conjugate gradient does not apply since 
                    # require a Hermitian matrix.
                    x,info = scipy.sparse.linalg.bicgstab(a,y)
                    if info > 0 :
                        print('convergence to tolerance not achieved')
                        print('#iterations = ',info)
                    elif info < 0 :
                        print('illegal input or breakdown in conjugate gradient')
                    # Convert multi state from vector to dict format
                    psi2 = {}
                    for i,amp in enumerate(x):
                        if amp != 0:
                            psi2[basis[i]] = amp
                    # Loop over out-going transition operators
                    for tOut,tOpOut in enumerate(tOpsOut): 
                        # Calculate state |psi3> = tOpOut |psi2>
                        # This state has no core-hole. 
                        psi3 = applyOp(tOpOut,psi2,slaterWeightMin,restrictions)
                        # Normalization factor
                        normalization = sqrt(norm2(psi3))
                        for state in psi3.keys(): 
                            psi3[state] /= normalization
                        # Remove product states with small weight
                        for state,amp in list(psi3.items()):
                            if abs(amp)**2 < slaterWeightMin:
                                psi3.pop(state)
                        # Calculate Green's function
                        gs[iE,tIn,tOut,iwIn,:] = normalization**2*getGreen(
                            e,psi3,hOp,wLoss,delta2,krylovSize,slaterWeightMin,
                            restrictions,hGround)
    elif parallelizationMode == 'wIn':
        # Loop over eigen states
        for iE in range(nE):
            psi =  psis[iE]
            e = esR[iE]
            # Loop over in-coming transition operators
            for tIn,tOpIn in enumerate(tOpsIn): 
                # Core-hole state
                psi1 = applyOp(tOpIn,psi,slaterWeightMin,restrictions)
                # Hamiltonian acting on relevant product states. |PS> : {H|PS>}
                nTMP = len(hExcited)
                h = expandBasisAndHamiltonian(hExcited,hOp,psi1.keys(),restrictions)
                if rank == 0: 
                    print('len(psi1),len(h),len(hExcited),#elements added to hExcited')
                    print(len(psi1),len(h),len(hExcited),len(hExcited)-nTMP)
                index = {ps:i for i,ps in enumerate(h.keys())} 
                basis = {i:ps for i,ps in enumerate(h.keys())} 
                n = len(h)
                # Express psi1 as a vector
                y = np.zeros(n,dtype=np.complex)
                for ps,amp in psi1.items():
                    y[index[ps]] = amp
                # store psi1 as a sparse vector
                #y = scipy.sparse.csr_matrix(y)
                # Express Hamiltonian in matrix form
                hValues, rows, cols = [],[],[]
                for psJ,res in h.items():
                    for psI,hValue in res.items():
                        hValues.append(hValue)
                        rows.append(index[psI])
                        cols.append(index[psJ])
                # store Hamiltonian in sparse matrix form
                h = scipy.sparse.csr_matrix((hValues,(rows,cols)),shape=(n,n))
                # Rank dependent variable
                g = {}
                if rank == 0: print('Loop over in-coming photon energies...')
                # Loop over in-coming photon energies, unique for each MPI rank
                for iwIn in getJobs(rank,ranks,len(wIns)):
                    wIn = wIns[iwIn]
                    # Initialize Green's functions
                    g[iwIn] =  np.zeros((len(tOpsOut),len(wLoss)),dtype=np.complex)        
                    # A = (wIn+1j*delta1+e)*\hat{1} - hOp.
                    a = scipy.sparse.csr_matrix(([wIn+1j*delta1+e]*n,(range(n),range(n))),
                                                shape=(n,n))
                    a -= h
                    # Find x by solving: a*x = y            
                    # Biconjugate gradient stabilized method.
                    # Pure conjugate gradient does not apply since 
                    # require a Hermitian matrix.
                    x,info = scipy.sparse.linalg.bicgstab(a,y)
                    if info > 0 :
                        print('convergence to tolerance not achieved')
                        print('#iterations = ',info)
                    elif info < 0 :
                        print('illegal input or breakdown in conjugate gradient')
                    # Convert multi state from vector to dict format
                    psi2 = {}
                    for i,amp in enumerate(x):
                        if amp != 0:
                            psi2[basis[i]] = amp
                    # Loop over out-going transition operators
                    for tOut,tOpOut in enumerate(tOpsOut): 
                        # Calculate state |psi3> = tOpOut |psi2>
                        # This state has no core-hole. 
                        psi3 = applyOp(tOpOut,psi2,slaterWeightMin,restrictions)
                        # Normalization factor
                        normalization = sqrt(norm2(psi3))
                        for state in psi3.keys(): 
                            psi3[state] /= normalization
                        # Remove product states with small weight
                        for state,amp in list(psi3.items()):
                            if abs(amp)**2 < slaterWeightMin:
                                psi3.pop(state)
                        # Calculate Green's function
                        g[iwIn][tOut,:] = normalization**2*getGreen(
                            e,psi3,hOp,wLoss,delta2,krylovSize,slaterWeightMin,
                            restrictions,hGround)

                # Distribute the Green's functions among the ranks
                for r in range(ranks):
                    gTmp = comm.bcast(g, root=r)
                    for iwIn,gValue in gTmp.items():
                        gs[iE,tIn,:,iwIn,:] = gValue
    return gs

