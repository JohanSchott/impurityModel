#!/bin/bash -e

# Script folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Ni in NiO
# Non-interacting Hamiltonian constructed from CF-parameters.

# Filename of the non-relativistic non-interacting Hamiltonian
h0_filename="${DIR}/../h0/h0_NiO_CF.json"
# Filename of the radial part of the correlated orbitals.
radial_filename=${DIR}"/../radialOrbitals/Ni3d.dat"

echo "H0 filename: $h0_filename"
echo "Radial wavefunction filename: $radial_filename"

mpirun -n 1 python -m impurityModel.ed.get_spectra_using_CF $h0_filename $radial_filename
