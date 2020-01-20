#!/bin/bash

# Script folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Mn in MnO
# Non-interacting Hamiltonian constructed from CF-parameters.

# Filename of the radial part of the correlated orbitals.
radial_filename=${DIR}"/../radialOrbitals/Mn3d.dat"
echo "Radial wavefunction filename: $radial_filename"

mpirun -n 1 python -m impurityModel.ed.get_spectra_using_CF $radial_filename --n0imps 6 5 --Fdd 6 0 9.0 0 6.1 --Fpd 7.5 0 5.6 --Gpd 0 4 0 2.3 --xi_2p 6.936 --xi_3d 0.051  --e_imp -0.463  --e_deltaO_imp 0.638 --e_val_eg -4.8 --e_val_t2g -6.7  --v_val_eg 1.910 --v_val_t2g 1.408  --nPsiMax 7 


