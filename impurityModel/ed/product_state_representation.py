"""
This module contains functions for translating a product state from one format
to another format. The possible types are:
tuple, str, int, bitarray and bytes.

Here is one product state example, expressed in the different formats:
(2, 5)
"0010010"
18
bitarray('0010010')
b'$'

where there are two electrons and 7 spin-orbitals in total.
The tuple expresses which spin-orbitals are occupied by electrons.
The string shows the occupation of all spin-orbitals.
The integer is a compact format. It is the integer representation of the string expressed in base 2.

The bitarray representation is the only mutable type among the different product state representation types.
This makes bitarrays suitable for manipulating product states, i.e. removing or adding electrons.
The bytes representation is constructed from the bitarray representation by calling the ".tobytes" method.
The bytes type is, like tuple, str and int, immutable and uses about the same amount of memory as the integer
representation. However, from an bitarray, it takes less time to convert to a bytes representation than to an integer
representation.

The tuple, integer, and the bytes representation needs to know the total number of spin-orbitals in the system.

In the create.py and remove.py modules, the ordering convention is such that this product state example represents:
`|psi> = c2 c5 |0>`, (and not `c5 c2 |0>`).

"""


from bitarray import bitarray


def str2int(s):
    """
    Returns integer representation of product state.

    Parameters
    ----------
    s : str

    """
    return int(s, 2)


def int2str(i, n):
    """
    Returns string representation of product state.

    Parameters
    ----------
    i : int
        Product state.
    n : int
        Total number of spin-orbitals in the system.

    """
    s = bin(i)[2:]
    s = "0" * (n - len(s)) + s
    return s


def str2tuple(s):
    """
    Returns tuple representation of product state.

    Parameters
    ----------
    s : str
        Product state.

    """
    return tuple(pos for pos, char in enumerate(s) if char == "1")


def tuple2str(t, n):
    """
    Returns string representation of product state.

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
            s += "0"
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
    return str2int(tuple2str(t, n))


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
    return str2tuple(int2str(i, n))


# Functions below are related to bitarray and bytes.


def bitarray2bytes(bits):
    """
    Returns bytes representation of product state.

    Parameters
    ----------
    bits : bitarray
        Representation of a product state, in terms of a bitarray.

    """
    return bits.tobytes()


def bytes2bitarray(bytestr, n):
    """
    Returns bitarray representation of product state.

    Parameters
    ----------
    bytestr : bytes
        Represenation of a product state, in terms of a bytes object.
    n : int
        Total number of spin-orbitals in the system.

    """
    # Generate a empty bitarray
    bits = bitarray(0)
    # Load this bitarray with the information stored in the byte string.
    bits.frombytes(bytestr)
    # Return the bitarray
    return bits[:n]


def int2bitarray(i, n):
    """
    Returns bitarray representation of product state.

    Parameters
    ----------
    i : int
        Product state.
    n : int
        Total number of spin-orbitals in the system.

    """
    return bitarray(int2str(i, n))


def bitarray2int(bits):
    """
    Returns integer representation of product state.

    Parameters
    ----------
    bits : bitarray
        Representation of a product state, in terms of a bitarray.

    """
    return int(bits.to01(), 2)


def bitarray2str(bits):
    """
    Returns string representation of product state.

    Parameters
    ----------
    bits : bitarray
        Representation of a product state, in terms of a bitarray.

    """
    return bits.to01()


def str2bitarray(s):
    """
    Returns bitarray representation of product state.

    Parameters
    ----------
    s : str
        Representation of a product state, in terms of a string.

    """
    return bitarray(s)


def tuple2bitarray(t, n):
    """
    Returns bitarray representation of product state.

    Parameters
    ----------
    t : tuple
        Representation of a product state, in terms of a tuple.
    n : int
        Total number of spin-orbitals in the system.

    """
    return bitarray(tuple2str(t, n))


def bitarray2tuple(bits):
    """
    Returns tuple representation of product state.

    Parameters
    ----------
    bits : bitarray
        Representation of a product state, in terms of a bitarray.

    """
    return str2tuple(bitarray2str(bits))


def tuple2bytes(t, n):
    """
    Returns bytes representation of product state.

    Parameters
    ----------
    t : tuple
        Representation of a product state, in terms of a tuple.
    n : int
        Total number of spin-orbitals in the system.

    """
    return tuple2bitarray(t, n).tobytes()


def bytes2tuple(bytestr, n):
    """
    Returns byte string representation of product state.

    Parameters
    ----------
    bytestr : bytes
        Represenation of a product state, in terms of a bytes object.
    n : int
        Total number of spin-orbitals in the system.

    """
    return bitarray2tuple(bytes2bitarray(bytestr, n))


def str2bytes(s):
    """
    Returns byte string representation of product state.

    Parameters
    ----------
    s : str
        Representation of a product state, in terms of a string.

    """
    return bitarray2bytes(str2bitarray(s))


def bytes2str(bytestr, n):
    """
    Returns string representation of product state.

    Parameters
    ----------
    bytestr : bytes
        Represenation of a product state, in terms of a bytes object.
    n : int
        Total number of spin-orbitals in the system.

    """
    return bitarray2str(bytes2bitarray(bytestr, n))


def int2bytes(i, n):
    """
    Returns bytes representation of product state.

    Parameters
    ----------
    i : int
        Representation of a product state, in terms of an integer.
    n : int
        Total number of spin-orbitals in the system.

    """
    return bitarray2bytes(int2bitarray(i, n))


def bytes2int(bytestr, n):
    """
    Returns integer representation of product state.

    Parameters
    ----------
    bytestr : bytes
        Represenation of a product state, in terms of a bytes object.
    n : int
        Total number of spin-orbitals in the system.

    """
    return bitarray2int(bytes2bitarray(bytestr, n))
