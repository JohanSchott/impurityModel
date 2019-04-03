#!/usr/bin/env python3


"""

product_state_representation
============================

This module contains functions for translating a product state from one format
to another format. The possible formats are: tuple, str, int

Here is one product state example, expressed in the different formats:
(2, 5)
"0010010"
10
where there are two electrons and 7 spin-orbitals in total.
The tuple expresses which spin-orbitals are occupied by electrons.
The string shows the occupation of all spin-orbitals.
The integer is a compact format. It is the integer representation of the binary string expressed in base 2.

"""



def binary2int(b):
    return int(b, 2)


def int2binary(i, n):
    b = bin(i)[2:]
    b = "0"*(n - len(b)) + b
    return b


def binary2tuple(b):
    return tuple( pos for pos, char in enumerate(b) if char == "1" )


def tuple2binary(t, n):
    s = ""
    for i in range(n):
        if i in t:
            s += "1"
        else:
            s+= "0"
    return s


def tuple2int(t, n):
    return binary2int(tuple2binary(t, n))


def int2tuple(i, n):
    return binary2tuple(int2binary(i, n))
