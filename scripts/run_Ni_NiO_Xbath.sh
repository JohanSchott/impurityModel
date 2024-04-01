#!/bin/bash -e

# Number of valence bath states coupled to 3d-orbitals.
# Currently accepted values are 10, 20, 50, 100 or 300.
# Check if the first input parameter is empty.
if [[ -z "$1" ]]; then
    nBath3d=10
else
    nBath3d=$1
fi
# Number of MPI ranks to use.
# Check if the second input parameter is empty.
if [[ -z "$2" ]]; then
    ranks=1
else
    ranks=$2
fi

# Script folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Ni in NiO
# Filename of the non-relativistic non-interacting Hamiltonian
h0_filename="${DIR}/../h0/h0_NiO_${nBath3d}bath.pickle"
# Filename of the radial part of the correlated orbitals.
radial_filename=${DIR}"/../radialOrbitals/Ni3d.dat"

echo "Number of MPI ranks to use: $ranks"
echo "Number of bath states: $nBath3d"
echo "H0 filename: $h0_filename"
echo "Radial wavefunction filename: $radial_filename"

echo "PYTHONPATH:"
echo $PYTHONPATH
echo "After PYTHONPATH."
python --version
python3 --version
which python
which python3

which mpiexec
which mpirun

mpiexec --help

mpiexec -n 2 --verbose echo "hej"
mpiexec -n 2 --verbose python -c "print('hello')"
mpiexec -n $ranks --verbose python -m impurityModel.ed.get_spectra $h0_filename $radial_filename --nBaths 0 $nBath3d --nValBaths 0 $nBath3d

#mpirun -n $ranks python -m impurityModel.ed.get_spectra $h0_filename $radial_filename --nBaths 0 $nBath3d --nValBaths 0 $nBath3d
#python -m impurityModel.ed.get_spectra $h0_filename $radial_filename --nBaths 0 $nBath3d --nValBaths 0 $nBath3d

