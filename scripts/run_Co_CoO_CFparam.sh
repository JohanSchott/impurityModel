#!/bin/bash

# Script folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Co in CoO
# Non-interacting Hamiltonian constructed from CF-parameters.

# Filename of the radial part of the correlated orbitals.
radial_filename=${DIR}"/../radialOrbitals/Co3d.dat"
echo "Radial wavefunction filename: $radial_filename"

mpirun -n 1 python -m impurityModel.ed.get_spectra_using_CF $radial_filename --n0imps 6 7 --Fdd 7 0 9.6 0 6.4 --Fpd 8 0 6.4 --Gpd 0 4.6 0 2.6 --xi_2p 9.859 --xi_3d 0.079  --e_imp -0.802  --e_deltaO_imp 0.612 --e_val_eg -4.5 --e_val_t2g -6.5  --v_val_eg 1.919 --v_val_t2g 1.412  --nPsiMax 13 


