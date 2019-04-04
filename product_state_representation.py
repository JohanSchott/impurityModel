#!/usr/bin/env python3


"""

product_state_representation
============================

This module contains functions for translating a product state from one format
to another format. The possible formats are: tuple, str, int

Here is one product state example, expressed in the different formats:
(2, 5)
"0010010"
18
where there are two electrons and 7 spin-orbitals in total.
The tuple expresses which spin-orbitals are occupied by electrons.
The string shows the occupation of all spin-orbitals.
The integer is a compact format. It is the integer representation of the binary string expressed in base 2.

In the finite.py module, the ordering convention is such that this product state example represents |psi> = c2 c5 |0>, (and not c5 c2 |0>).
 
"""



def binary2int(b):
    """
    Returns integer representation of product state.

    Parameters
    ----------
    b : str

    """
    return int(b, 2)


def int2binary(i, n):
    """
    Returns binary string representation of product state.

    Parameters
    ----------
    i : int
        Product state.
    n : int
        Total number of spin-orbitals in the system.

    """
    b = bin(i)[2:]
    b = "0"*(n - len(b)) + b
    return b


def binary2tuple(b):
    """
    Returns tuple representation of product state.

    Parameters
    ----------
    b : str
        Product state.

    """
    return tuple( pos for pos, char in enumerate(b) if char == "1" )


def tuple2binary(t, n):
    """
    Returns binary string representation of product state.

    Parameters
    ----------
    t : tuple
        Product state.
    n : int
        Total number of spin-orbitals in the system.

    """
    s = ""
    for i in range(n):
        if i in t:
            s += "1"
        else:
            s+= "0"
    return s


def tuple2int(t, n):
    """
    Returns integer representation of product state.

    Parameters
    ----------
    t : tuple
        Product state.
    n : int
        Total number of spin-orbitals in the system.

    """
    return binary2int(tuple2binary(t, n))


def int2tuple(i, n):
    """
    Returns tuple representation of product state.

    Parameters
    ----------
    i : int
        Product state.
    n : int
        Total number of spin-orbitals in the system.

    """
    return binary2tuple(int2binary(i, n))

