#!/bin/bash -e

# Number of MPI ranks to use.
# Check if the first input parameter is empty.
if [[ -z "$1" ]]; then
    ranks=1
else
    ranks=$1
fi

# Script folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Ni in NiO
# 50 valence bath states, 10 conduction bath states.
# Filename of the non-relativistic non-interacting Hamiltonian
h0_filename="${DIR}/../h0/h0_NiO_50p10bath.pickle"
# Filename of the radial part of the correlated orbitals.
radial_filename=${DIR}"/../radialOrbitals/Ni3d.dat"

echo "Number of MPI ranks to use: $ranks"
echo "H0 filename: $h0_filename"
echo "Radial wavefunction filename: $radial_filename"

mpirun -n $ranks python -m impurityModel.ed.get_spectra $h0_filename $radial_filename --nBaths 0 60 --nValBaths 0 10 --dnConBaths 0 1

