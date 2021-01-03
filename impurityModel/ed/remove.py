#!/usr/bin/env python3


"""

remove
======

This module contains functions to remove/annihilate an electron to a product state.
Depending on the representation type of the product state, different functions should be used.
Supported types are: tuple, str, int, bitarray and bytes.

The ordering convention is such that the normal ordering of a product state is |psi> = c2 c5 |0>, (and not c5 c2 |0>).

"""


import sys
from bisect import bisect_left
# Local imports
from impurityModel.ed import product_state_representation as psr


def binary_search(a, x):
    '''
    Return index to the leftmost value exactly equal to x.

    If x is not in the list, return -1.

    '''
    i = bisect_left(a, x)
    return i if i != len(a) and a[i] == x else -1


def utuple(i, state):
    '''
    Remove electron at orbital i in state.

    Parameters
    ----------
    i : int
        Spin-orbital index
    state : tuple
        Product state.
        Elements are indices of occupied orbitals.

    Returns
    -------
    stateNew : tuple
        Product state
    amp : int
        Amplitude. 0, -1 or 1.

    '''
    j = binary_search(state,i)
    if j != -1:
        stateNew = state[:j] + state[j+1:]
        amp = 1 if j%2 == 0 else -1
        return stateNew, amp
    else:
        return (), 0


def uint(n_spin_orbitals, i, state):
    """
    Remove electron at orbital i in state.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    i : int
        Spin-orbital index
    state : int
        Product state.

    Returns
    -------
    state_new : int
        Product state
    amp : int
        Amplitude. 0, -1 or 1.

    """
    # String representation of product state.
    s = psr.int2str(state, n_spin_orbitals)
    if s[i] == "0":
        return -1, 0
    elif s[i] == "1":
        state_new = state - 2**(n_spin_orbitals-i-1)
        amp = 1 if s[:i].count("1") % 2 == 0 else -1
        return state_new, amp
    else:
        raise Exception("Integer representation of state is wrong.")


def ustr(i, state):
    """
    Remove electron at orbital i in state.

    Parameters
    ----------
    i : int
        Spin-orbital index
    state : str
        Product state.

    Returns
    -------
    state_new : str
        Product state
    amp : int
        Amplitude. 0, -1 or 1.

    """
    if state[i] == "0":
        return "", 0
    elif state[i] == "1":
        state_new = state[:i] + "0" + state[i+1:]
        amp = 1 if state[:i].count("1") % 2 == 0 else -1
        return state_new, amp
    else:
        raise Exception("String representation of state is wrong.")


def ubitarray(i, state):
    """
    Remove electron at orbital i in state.

    Parameters
    ----------
    i : int
        Spin-orbital index
    state : bitarray(N)
        Product state.

    Returns
    -------
    amp : int
        Amplitude. 0, -1 or 1.

    """
    if state[i]:
        # Modify the product state by removing an electron
        state[i] = False
        # Amplitude
        return 1 if state[:i].count() % 2 == 0 else -1
    else:
        return 0


def ubytes(n_spin_orbitals, i, state):
    """
    Remove electron at orbital i in state.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    i : int
        Spin-orbital index
    state : bytes
        Product state.

    Returns
    -------
    state_new : bytes
        Product state.
    amp : int
        Amplitude. 0, -1 or 1.

    """
    # bitarray representation of product state.
    bits = psr.bytes2bitarray(state, n_spin_orbitals)
    # remove an electron at spin-orbital index i.
    amp = ubitarray(i, bits)
    # Convert back the updated product state to bytes representation.
    state_new = psr.bitarray2bytes(bits)
    return state_new, amp
