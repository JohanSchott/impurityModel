#!/bin/bash

# Script folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Ni in NiO
# Number of valence bath states coupled to 3d-orbitals. 10, 20, 50, 100 or 300
nBath3d=10
# Filename of the non-relativistic non-interacting Hamiltonian
h0_filename="${DIR}/../h0/h0_NiO_${nBath3d}bath.pickle"
# Filename of the radial part of the correlated orbitals.
radial_filename=${DIR}"/../radialOrbitals/Ni3d.dat"
ls $h0_filename
ls $radial_filename
mpirun -n 2 python -m impurityModel.ed.get_spectra $h0_filename $radial_filename --nBaths 0 $nBath3d --nValBaths 0 $nBath3d

