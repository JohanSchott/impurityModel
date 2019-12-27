#!/bin/bash

# Ni in NiO
# 50 valence bath states, 10 conduction bath states.
# Filename of the non-relativistic non-interacting Hamiltonian
h0_filename="${DIR}/../h0/h0_NiO_50p10bath.pickle"
# Filename of the radial part of the correlated orbitals.
radial_filename=${DIR}"/../radialOrbitals/Ni3d.dat"
ls $h0_filename
ls $radial_filename
mpirun -n 2 python -m impurityModel.ed.get_spectra $h0_filename $radial_filename --nBaths 0 60 --nValBaths 0 10 --dnConBaths 0 1

