"""
Module with tests of functions in product_state_representation.py.
"""

from bitarray import bitarray

# Local
from impurityModel.ed import product_state_representation as psr


def test_tuple2str():
    # Number of spin-orbitals in the system
    n = 7
    # Indices of occupied spin-orbitals
    t = (2, 5)
    assert psr.tuple2str(t, n) == "0010010"


def test_str2tuple():
    # String representation of one product state with particles at indices 2 and 5.
    s = "0010010"
    assert psr.str2tuple(s) == (2, 5)


def test_tuple2int():
    # Number of spin-orbitals in the system
    n = 7
    # Indices of occupied spin-orbitals
    t = (2, 5)
    assert psr.tuple2int(t, n) == 18


def test_int2tuple():
    # Number of spin-orbitals in the system
    n = 7
    # Integer representation of one product state.
    i = 18
    assert psr.int2tuple(i, n) == (2, 5)


def test_tuple2bitarray():
    # Number of spin-orbitals in the system
    n = 7
    # Indices of occupied spin-orbitals
    t = (2, 5)
    assert bitarray("0010010") == psr.tuple2bitarray(t, n)


def test_bitarray2tuple():
    # Bitarray representation of one product state.
    bits = bitarray("0010010")
    assert psr.bitarray2tuple(bits) == (2, 5)


def test_tuple2bytes():
    # Number of spin-orbitals in the system
    n = 7
    # Indices of occupied spin-orbitals
    t = (2, 5)
    assert psr.tuple2bytes(t, n) == b"$"


def test_bytes2tuple():
    # Number of spin-orbitals in the system
    n = 7
    # Bytes representation of one product state.
    bytestr = b"$"
    assert psr.bytes2tuple(bytestr, n) == (2, 5)


def test_str2int():
    # String representation of a product state.
    s = "0010010"
    assert psr.str2int(s) == 18


def test_int2str():
    # Number of spin-orbitals in the system
    n = 7
    # Integer representation of a product state.
    i = 18
    assert psr.int2str(i, n) == "0010010"


def test_str2bitarray():
    # String representation of a product state.
    s = "0010010"
    assert bitarray("0010010") == psr.str2bitarray(s)


def test_bitarray2str():
    # Bitarray representation of a product state.
    bits = bitarray("0010010")
    assert psr.bitarray2str(bits) == "0010010"


def test_str2bytes():
    # String representation of a product state.
    s = "0010010"
    assert psr.str2bytes(s) == b"$"


def test_bytes2str():
    # Number of spin-orbitals in the system
    n = 7
    # Bytes representation of a product state.
    bytestr = b"$"
    assert psr.bytes2str(bytestr, n) == "0010010"


def test_int2bitarray():
    # Number of spin-orbitals in the system
    n = 7
    # Integer representation of a product state.
    i = 18
    assert bitarray("0010010") == psr.int2bitarray(i, n)


def test_bitarray2int():
    # Bitarray representation of a product state.
    bits = bitarray("0010010")
    assert psr.bitarray2int(bits) == 18


def test_int2bytes():
    # Number of spin-orbitals in the system
    n = 7
    # Integer representation of a product state.
    i = 18
    assert psr.int2bytes(i, n) == b"$"


def test_bytes2int():
    # Number of spin-orbitals in the system
    n = 7
    # Bytes representation of a product state.
    bytestr = b"$"
    assert psr.bytes2int(bytestr, n) == 18


def test_bitarray2bytes():
    # Bitarray representation of a product state.
    bits = bitarray("0010010")
    assert psr.bitarray2bytes(bits) == b"$"


def test_bytes2bitarray():
    # Number of spin-orbitals in the system
    n = 7
    # Bytes representation of a product state.
    bytestr = b"$"
    assert bitarray("0010010") == psr.bytes2bitarray(bytestr, n)
