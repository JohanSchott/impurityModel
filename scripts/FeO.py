#!/usr/bin/env python3

# Script for solving many-body impurity problem.

import numpy as np
import scipy.sparse.linalg
from collections import OrderedDict
import sys,os
from mpi4py import MPI

from finite import c2i,k_B
import finite
import spectra

def main():
   
    # MPI variables 
    comm = MPI.COMM_WORLD
    rank = comm.rank
    ranks = comm.size
   
    #if rank == 0: printGaunt()
    
    # -----------------------
    # System information  
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
    # Basis occupation information. 
    # Angular momentum : initial impurity occupation 
    n0imp = OrderedDict()
    n0imp[l1] = 6 # 0 = empty, 2*(2*l1+1) = Full occupation
    n0imp[l2] = 6 # 8 for Ni+2
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
    Fdd=[6.5,0,9.3,0,6.2]
    Fpp=[0,0,0]
    Fpd=[7.5,0,6.0]
    Gpd=[0,4.3,0,2.4] 
    # SOC values
    xi_2p = 8.301
    xi_3d = 0.064
    # Double counting parameter
    chargeTransferCorrection = 1.5
    # Onsite 3d energy parameters
    eImp3d = -0.620
    deltaO = 0.646
    # Magnetic field
    hField = [0,0,0.00001] # 0.120*np.array([1,1,2])/np.sqrt(6) # [0,0,0.00001]
    # Bath energies and hoppings for 3d orbitals
    eValEg = -4.8
    eValT2g = -6.7
    eConEg = 3
    eConT2g = 2
    vValEg = 2.014
    vValT2g = 1.462
    vConEg = 0.6
    vConT2g = 0.4
    # -----------------------
    # Ground state diagonalization mode
    groundDiagMode = 'Lanczos'  # 'Lanczos' or 'full'
    # Maximum number of eigenstates to consider
    nPsiMax = 17
    eigenValueTol = 1e-9
    # Slater determinant truncation parameters
    # Removes Slater determinants with weights
    # smaller than this optimization parameter.
    slaterWeightMin = 1e-7
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
    energyCut = 10*k_B*T
    w = np.linspace(-25,25,3000)
    delta = 0.2
    krylovSize = 80
    # Occupation restrictions, used when spectra is generated
    l = 2
    restrictions = {}
    restrictions[tuple(c2i(nBaths,(l,m,s)) for m in range(-l,l+1) for s in range(2))] = (n0imp[l]-1,n0imp[l]+3)
    # XAS polarization vectors. 
    epsilons = [[1,0,0],[0,1,0],[0,0,1]] # [[0,0,1]]
    # RIXS parameters
    # Polarization vectors, of in and outgoing photon.
    epsilonsRIXSin = [[1,0,0],[0,1,0],[0,0,1]]  # [[0,0,1]]
    epsilonsRIXSout = [[1,0,0],[0,1,0],[0,0,1]] # [[0,0,1]]
    wIn = np.linspace(-10,15,100)
    wLoss = np.linspace(-2,12,4000)
    deltaRIXS = 0.050
    # NIXS parameters
    qsNIXS = [2*np.array([1,1,1])/np.sqrt(3),7*np.array([1,1,1])/np.sqrt(3)]
    deltaNIXS = 0.100
    # Angular momentum of final and initial orbitals in the NIXS excitation process.
    liNIXS,ljNIXS = 2,2
    # File name of file containing radial mesh and radial part of final 
    # and initial orbitals in the NIXS excitation process.
    radialFileName = os.path.dirname(sys.argv[0])[:-7] + 'radialOrbitals/Fe3d.dat'
    data = np.loadtxt(radialFileName)
    radialMesh = data[:,0]
    RiNIXS = data[:,1]
    RjNIXS = np.copy(RiNIXS)
    # -----------------------

    
    # Hamiltonian
    if rank == 0: print('Construct the Hamiltonian operator...')
    hOp = getHamiltonianOperator(nBaths,valBaths,[Fdd,Fpp,Fpd,Gpd],
                                 [xi_2p,xi_3d],
                                 [n0imp,chargeTransferCorrection],
                                 [eImp3d,deltaO],hField,
                                 [vValEg,vValT2g,vConEg,vConT2g],
                                 [eValEg,eValT2g,eConEg,eConT2g])
    # Many body basis for the ground state
    if rank == 0: print('Create basis...')
    basis = finite.getBasis(nBaths,valBaths,dnValBaths,dnConBaths,
                            dnTol,n0imp)
    if rank == 0: print('#basis states = {:d}'.format(len(basis))) 

    # Diagonalization of restricted active space Hamiltonian
    if rank == 0: print('Create Hamiltonian matrix...')
    h = finite.getHamiltonianMatrix(hOp,basis)    
    if rank == 0: print('<#Hamiltonian elements/column> = {:d}'.format(
        int(len(np.nonzero(h)[0])*1./len(basis)))) 
    if rank == 0: print('Diagonalize the Hamiltonian...')
    if groundDiagMode == 'full':
        es, vecs = np.linalg.eigh(h.todense())
        es = es[:nPsiMax]
        vecs = vecs[:,:nPsiMax]
    elif groundDiagMode == 'Lanczos':
        es, vecs = scipy.sparse.linalg.eigsh(h,k=nPsiMax,which='SA',tol=eigenValueTol)
        # Sort the eigenvalues and eigenvectors in ascending order.
        indices = np.argsort(es)
        es = np.array([es[i] for i in indices])
        vecs = np.array([vecs[:,i] for i in indices]).T
    else:
        print('Wrong diagonalization mode')
    if rank == 0: print('Proceed with {:d} eigenstates!'.format(len(es)))
    psis = [({basis[i]:vecs[i,vi] for i in range(len(basis)) 
              if slaterWeightMin <= abs(vecs[i,vi])**2 }) 
            for vi in range(len(es))]
    
    # Calculate static expectation values
    finite.printThermalExpValues(nBaths,es,psis)
    finite.printExpValues(nBaths,es,psis,n=nPsiMax) 
    
    # Print Slater determinants and weights 
    if rank == 0: print('Slater determinants/product states and correspoinding weights')
    weights = []
    for i,psi in enumerate(psis[:nPsiMax]):
        if rank == 0: print('Eigenstate {:d}.'.format(i))
        if rank == 0: print('Consists of {:d} product states.'.format(len(psi)))
        ws = np.array([ abs(a)**2 for a in psi.values() ])
        s = np.array([ ps for ps in psi.keys() ])
        j = np.argsort(ws)
        ws = ws[j[-1::-1]]
        s = s[j[-1::-1]]
        weights.append(ws)
        if rank == 0 and nPrintSlaterWeights > 0: 
            print('Highest (product state) weights:')
            print(ws[:nPrintSlaterWeights])
            print('Corresponding product states:')
            print(s[:nPrintSlaterWeights])
            print('') 

    # Calculate density matrix 
    if rank == 0: print('Density matrix (in cubic basis):')
    for i,psi in enumerate(psis[:nPsiMax]):
        if rank == 0: print('Eigenstate {:d}'.format(i))
        n = finite.getDensityMatrixCubic(nBaths,psi)
        if rank == 0: print('#element={:d}'.format(len(n)))
        for e,ne in n.items():
            if rank == 0 and abs(ne) > tolPrintOccupation: 
                if e[0] == e[1]:
                    print('Diagonal: (i,s)=',e[0],', occupation = {:7.2f}'.format(ne))
                else:
                    print('Off-diagonal: (i,si),(j,sj)=',e,', {:7.2f}'.format(ne)) 
        if rank == 0: print('') 

    # Save some information to disk
    if rank == 0:
        # Most of the input parameters. Dictonaries can be stored in this file format.
        np.savez_compressed('data',l1=l1,l2=l2,nBaths=nBaths,valBaths=valBaths,
                            n0imp=n0imp,dnTol=dnTol,
                            dnValBaths=dnValBaths,dnConBaths=dnConBaths,
                            Fdd=Fdd,Fpp=Fpp,Fpd=Fpd,Gpd=Gpd,
                            xi_2p=xi_2p,xi_3d=xi_3d,
                            chargeTransferCorrection=chargeTransferCorrection,
                            eImp3d=eImp3d,deltaO=deltaO,
                            hField=hField,
                            eBath=[eValEg,eValT2g,eConEg,eConT2g],
                            vBath=[vValEg,vValT2g,vConEg,vConT2g],
                            groundDiagMode=groundDiagMode,
                            nPsiMax=nPsiMax,
                            eigenValueTol=eigenValueTol,
                            slaterWeightMin=slaterWeightMin,
                            T=T,energyCut=energyCut,delta=delta,
                            krylovSize=krylovSize,restrictions=restrictions,
                            epsilons=epsilons,
                            epsilonsRIXSin=epsilonsRIXSin,epsilonsRIXSout=epsilonsRIXSout,
                            deltaRIXS=deltaRIXS,
                            deltaNIXS=deltaNIXS,
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
            np.savez_compressed('spectraInfo',E=es,w=w,wIn=wIn,wLoss=wLoss,
                                qsNIXS=qsNIXS,r=radialMesh,RiNIXS=RiNIXS,
                                RjNIXS=RjNIXS)


    if rank == 0: print('Create 3d inverse photoemission and photoemission spectra...')
    # Transition operators
    tOpsIPS = spectra.getInversePhotoEmissionOperators(nBaths,l=2)
    tOpsPS = spectra.getPhotoEmissionOperators(nBaths,l=2)
    if rank == 0: print("Inverse photoemission Green's function..")
    gsIPS = spectra.getSpectra(hOp,tOpsIPS,psis,es,w,delta,krylovSize,
                               slaterWeightMin,energyCut,restrictions)
    if rank == 0: print("Photoemission Green's function..")
    gsPS = spectra.getSpectra(hOp,tOpsPS,psis,es,-w,-delta,krylovSize,
                              slaterWeightMin,energyCut,restrictions)
    gsPS *= -1
    gs = gsPS + gsIPS
    if rank == 0: 
        print('#relevant eigenstates = {:d}'.format(np.shape(gs)[0]))
        print('#spin orbitals = {:d}'.format(np.shape(gs)[1]))
        print('#mesh points = {:d}'.format(np.shape(gs)[2]))
    # thermal average
    a = finite.thermal_average(es[:np.shape(gs)[0]],-gs.imag,T=T)
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
        print('Save spectra to disk...')
        np.savetxt('PS.dat',np.array(tmp).T,fmt='%8.4f',
                   header='E  sum  T1  T2  T3 ...')
        print('')
    

    if rank == 0: print('Create core 2p x-ray photoemission spectra (XPS) ...')
    # Transition operators
    tOpsPS = spectra.getPhotoEmissionOperators(nBaths,l=1)
    # Photoemission Green's function 
    gs = spectra.getSpectra(hOp,tOpsPS,psis,es,-w,-delta,krylovSize,
                            slaterWeightMin,energyCut,restrictions)
    gs *= -1
    if rank == 0: 
        print('#relevant eigenstates = {:d}'.format(np.shape(gs)[0]))
        print('#spin orbitals = {:d}'.format(np.shape(gs)[1]))
        print('#mesh points = {:d}'.format(np.shape(gs)[2]))
    # thermal average
    a = finite.thermal_average(es[:np.shape(gs)[0]],-gs.imag,T=T)
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
        print('Save spectra to disk...')
        np.savetxt('XPS.dat',np.array(tmp).T,fmt='%8.4f',
                   header='E  sum  T1  T2  T3 ...')
        print('')
   

    if rank == 0: print('Create NIXS spectra...')    
    # Transition operator: exp(iq*r) 
    tOps = spectra.getNIXSOperators(nBaths,qsNIXS,liNIXS,ljNIXS,
                                    RiNIXS,RjNIXS,radialMesh)
    # Green's function 
    gs = spectra.getSpectra(hOp,tOps,psis,es,wLoss,deltaNIXS,krylovSize,slaterWeightMin,
                            energyCut,restrictions)
    if rank == 0: 
        print('#relevant eigenstates = {:d}'.format(np.shape(gs)[0]))
        print('#q-points = {:d}'.format(np.shape(gs)[1]))
        print('#mesh points = {:d}'.format(np.shape(gs)[2]))
    # thermal average
    a = finite.thermal_average(es[:np.shape(gs)[0]],-gs.imag,T=T)
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
        print('Save spectra to disk...')
        np.savetxt('NIXS.dat',np.array(tmp).T,fmt='%8.4f',
                   header='E  sum  T1  T2  T3 ...')
        print('')


    if rank == 0: print('Create XAS spectra...')
    # Dipole transition operators
    tOps = spectra.getDipoleOperators(nBaths,epsilons)
    # Green's function 
    gs = spectra.getSpectra(hOp,tOps,psis,es,w,delta,krylovSize,slaterWeightMin,
                            energyCut,restrictions)
    if rank == 0: 
        print('#relevant eigenstates = {:d}'.format(np.shape(gs)[0]))
        print('#polarizations = {:d}'.format(np.shape(gs)[1]))
        print('#mesh points = {:d}'.format(np.shape(gs)[2]))
    # thermal average
    a = finite.thermal_average(es[:np.shape(gs)[0]],-gs.imag,T=T)
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
        print('Save spectra to disk...')
        np.savetxt('XAS.dat',np.array(tmp).T,fmt='%8.4f',
                   header='E  sum  T1  T2  T3 ...')
        print('')
    
    
    if rank == 0: print('Create RIXS spectra...')
    # Dipole 2p -> 3d transition operators
    tOpsIn = spectra.getDipoleOperators(nBaths,epsilonsRIXSin)
    # Dipole 3d -> 2p transition operators
    tOpsOut = spectra.getDaggeredDipoleOperators(nBaths,epsilonsRIXSout)
    # Green's function 
    gs = spectra.getRIXSmap(hOp,tOpsIn,tOpsOut,psis,es,wIn,wLoss,delta,deltaRIXS,
                            krylovSize,slaterWeightMin,energyCut,restrictions)
    if rank == 0: 
        print('#relevant eigenstates = {:d}'.format(np.shape(gs)[0]))
        print('#in-polarizations = {:d}'.format(np.shape(gs)[1]))
        print('#out-polarizations = {:d}'.format(np.shape(gs)[2]))
        print('#mesh points of input energy = {:d}'.format(np.shape(gs)[3]))
        print('#mesh points of energy loss = {:d}'.format(np.shape(gs)[4]))
    # Thermal average
    a = finite.thermal_average(es[:np.shape(gs)[0]],-gs.imag,T=T)
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
        print('Save spectra to disk...')
        # I[wLoss,wIn], with wLoss on first column and wIn on first row.
        tmp = np.zeros((len(wLoss)+1,len(wIn)+1),dtype=np.float32)
        tmp[0,0] = len(wIn)
        tmp[0,1:] = wIn
        tmp[1:,0] = wLoss
        tmp[1:,1:] = aSum.T
        tmp.tofile('RIXS.bin')
        print('')
    
    
    if rank == 0 and printH5: h5f.close()
    print('Script finished for rank:',rank)
    
def getHamiltonianOperator(nBaths,valBaths,slaterCondon,SOCs,
                           DCinfo,impurityInfo,hField,
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
    hField : list
        External magnetic field.  
        Elements hx,hy,hz
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
    hx,hy,hz = hField
    vValEg,vValT2g,vConEg,vConT2g = vHoppings
    eValEg,eValT2g,eConEg,eConT2g = eBaths
     
    # Calculate U operator  
    uOperator = finite.get2p3dSlaterCondonUop(Fdd=Fdd,Fpp=Fpp,
                                              Fpd=Fpd,Gpd=Gpd)
    # Add SOC 
    SOC2pOperator = finite.getSOCop(xi_2p,l=1)
    SOC3dOperator = finite. getSOCop(xi_3d,l=2)
    
    # Double counting (DC) correction
    # MLFT DC 
    dc = finite.dc_MLFT(n3d_i=n0imp[2],c=chargeTransferCorrection,Fdd=Fdd,
                        n2p_i=n0imp[1],Fpd=Fpd,Gpd=Gpd)
    eDCOperator = {}
    for il,l in enumerate([2,1]):
        for m in range(-l,l+1):
            for s in range(2):
                eDCOperator[(((l,m,s),'c'),((l,m,s),'a'))] = -dc[il]
    
    # Calculate impurity 3d Hamiltonian
    # (Either by reading matrix or parameterize it)
    l = 2 
    eImpEg = eImp3d + 3./5*deltaO
    eImpT2g = eImp3d - 2./5*deltaO  
    hImp3d = np.zeros((2*l+1,2*l+1))
    np.fill_diagonal(hImp3d,(eImpEg,eImpEg,eImpT2g,eImpT2g,eImpT2g)) 
    # Convert to spherical harmonics basis
    u = finite.get_spherical_2_cubic_matrix(spinpol=False,l=l)
    hImp3d = np.dot(u,np.dot(hImp3d,np.conj(u.T)))
    # Convert from matrix to operator form
    # Also add spin
    hImp3dOperator = {}
    for i,mi in enumerate(range(-l,l+1)):
        for j,mj in enumerate(range(-l,l+1)):
            if hImp3d[i,j] != 0:
                for s in range(2):
                    hImp3dOperator[(((l,mi,s),'c'),((l,mj,s),'a'))] = hImp3d[i,j]
    
    # Magnetic field
    hHfieldOperator = {}
    for m in range(-l,l+1):
        hHfieldOperator[(((l,m,1),'c'),((l,m,0),'a'))] = hx*1/2.
        hHfieldOperator[(((l,m,0),'c'),((l,m,1),'a'))] = hx*1/2.
        hHfieldOperator[(((l,m,1),'c'),((l,m,0),'a'))] = -hy*1/2.*1j
        hHfieldOperator[(((l,m,0),'c'),((l,m,1),'a'))] = hy*1/2.*1j
        for s in range(2):
            hHfieldOperator[(((l,m,s),'c'),((l,m,s),'a'))] = hz*1/2. if s==1 else -hz*1/2.        
    # Bath (3d) on-site energies and hoppings
    # Calculate hopping terms between bath and (3d) impurity
    # either by reading matrix or parameterize it
    vVal3d = np.zeros((2*l+1,2*l+1))
    vCon3d = np.zeros((2*l+1,2*l+1))
    eBathVal3d = np.zeros((2*l+1,2*l+1))
    eBathCon3d = np.zeros((2*l+1,2*l+1))
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
    for bathSet in range(nBaths[l]):
        for i,mi in enumerate(range(-l,l+1)):
            for j,mj in enumerate(range(-l,l+1)):
                if bathSet in range(valBaths[l]):
                    vHopp = vVal3d[i,j]
                    eBath = eBathVal3d[i,j]
                else:
                    vHopp = vCon3d[i,j]
                    eBath = eBathCon3d[i,j]
                if vHopp != 0:
                    for s in range(2):
                        hHoppOperator[(((l,mi,s),'c'),((l,mj,s,bathSet),'a'))] = vHopp
                        hHoppOperator[(((l,mj,s,bathSet),'c'),((l,mi,s),'a'))] = vHopp.conjugate()
                if eBath != 0:
                    for s in range(2):
                        eBath3dOperator[(((l,mi,s,bathSet),'c'),((l,mj,s,bathSet),'a'))] = eBath
    
    # Add Hamiltonian terms to one operator 
    hOperator = finite.addOps([uOperator,
                               hImp3dOperator,
                               hHfieldOperator,
                               SOC2pOperator,
                               SOC3dOperator,
                               eDCOperator,
                               hHoppOperator,
                               eBath3dOperator])
    # Convert spin-orbital indices to a single index
    hOp = {}
    for process,value in hOperator.items():
        hOp[tuple((c2i(nBaths,spinOrb),action) for spinOrb,action in process)] = value
    return hOp


if __name__== "__main__":
    main()
    #cProfile.run('main()',sort='cumulative')

