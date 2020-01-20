#!/bin/bash

# Script folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Fe in FeO
# Non-interacting Hamiltonian constructed from CF-parameters.

# Filename of the radial part of the correlated orbitals.
radial_filename=${DIR}"/../radialOrbitals/Fe3d.dat"
echo "Radial wavefunction filename: $radial_filename"

mpirun -n 1 python -m impurityModel.ed.get_spectra_using_CF $radial_filename --n0imps 6 6 --Fdd 6.5 0 9.3 0 6.2 --Fpd 7.5 0 6 --Gpd 0 4.3 0 2.4 --xi_2p 8.301 --xi_3d 0.064  --e_imp -0.620  --e_deltaO_imp 0.646 --e_val_eg -4.8 --e_val_t2g -6.7  --v_val_eg 2.014 --v_val_t2g 1.462  --nPsiMax 17 


