#!/usr/bin/env python3

"""
average
=======

Module containing functions for performing averaging.

"""

import numpy as np

# Boltzmann constant. Unit: eV/K. E = k_B * T,
# energy in eV and temperature in Kelvin.
k_B = 8.61701580807947e-05

def thermal_average(energies,observable,T=300):
    """
    Returns thermally averaged observables.

    Assumes all relevant states are included.
    Thus, no not check e.g. if the Boltzmann weight
    of the last state is small.

    Parameters
    ----------
    energies - list(N)
        energies[i] is the energy of state i.
    observable - list(N,...)
        observable[i,...] are observables for state i.
    T : float
        Temperature

    """
    if len(energies) != np.shape(observable)[0]:
        raise ValueError("Passed array is not of the right shape")
    z = 0
    e_average = 0
    o_average = 0
    weights = np.zeros(np.shape(energies),dtype=np.float)
    shift = np.min(energies)
    for j,(e,o) in enumerate(zip(energies,observable)):
        weight = np.exp(-(e-shift)/(k_B*T))
        z += weight
        weights[j] = weight
        e_average += weight*e
        o_average += weight*o
    e_average /= z
    o_average /= z
    weights /= z
    return o_average

