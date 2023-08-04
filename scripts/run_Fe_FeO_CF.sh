#!/bin/bash -e

# Script folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Feo in FeO
# Non-interacting Hamiltonian constructed from CF-parameters.

# Filename of the non-relativistic non-interacting Hamiltonian
h0_filename="${DIR}/../h0/h0_FeO_CF.json"
# Filename of the radial part of the correlated orbitals.
radial_filename=${DIR}"/../radialOrbitals/Fe3d.dat"

echo "H0 filename: $h0_filename"
echo "Radial wavefunction filename: $radial_filename"

mpirun -n 1 python -m impurityModel.ed.get_spectra_using_CF $h0_filename $radial_filename \
    --n0imps 6 6 --Fdd 6.5 0 9.3 0 6.2 --Fpd 7.5 0 6 --Gpd 0 4.3 0 2.4 --xi_2p 8.301 --xi_3d 0.064 --nPsiMax 17
