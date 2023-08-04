#!/bin/bash -e

# Script folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Mno in MnO
# Non-interacting Hamiltonian constructed from CF-parameters.

# Filename of the non-relativistic non-interacting Hamiltonian
h0_filename="${DIR}/../h0/h0_MnO_CF.json"
# Filename of the radial part of the correlated orbitals.
radial_filename=${DIR}"/../radialOrbitals/Mn3d.dat"

echo "H0 filename: $h0_filename"
echo "Radial wavefunction filename: $radial_filename"

mpirun -n 1 python -m impurityModel.ed.get_spectra_using_CF $h0_filename $radial_filename \
    --n0imps 6 5 --Fdd 6 0 9.0 0 6.1 --Fpd 7.5 0 5.6 --Gpd 0 4 0 2.3 --xi_2p 6.936 --xi_3d 0.051 --nPsiMax 7
