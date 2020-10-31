
"""
test_product_state_representation
=================================

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
    assert "0010010" == psr.tuple2str(t,n)


def test_str2tuple():
    # String representation of one product state with particles at indices 2 and 5.
    s = "0010010"
    assert (2, 5) == psr.str2tuple(s)


def test_tuple2int():
    # Number of spin-orbitals in the system
    n = 7
    # Indices of occupied spin-orbitals
    t = (2, 5)
    assert 18 == psr.tuple2int(t, n)


def test_int2tuple():
    # Number of spin-orbitals in the system
    n = 7
    # Integer representation of one product state.
    i = 18
    assert (2, 5) == psr.int2tuple(i, n)


def test_tuple2bitarray():
    # Number of spin-orbitals in the system
    n = 7
    # Indices of occupied spin-orbitals
    t = (2, 5)
    assert bitarray('0010010') == psr.tuple2bitarray(t, n)


def test_bitarray2tuple():
    # Bitarray representation of one product state.
    bits = bitarray('0010010')
    assert (2, 5) == psr.bitarray2tuple(bits)


def test_tuple2bytes():
    # Number of spin-orbitals in the system
    n = 7
    # Indices of occupied spin-orbitals
    t = (2, 5)
    assert b'$' == psr.tuple2bytes(t, n)


def test_bytes2tuple():
    # Number of spin-orbitals in the system
    n = 7
    # Bytes representation of one product state.
    bytestr = b'$'
    assert (2, 5) == psr.bytes2tuple(bytestr, n)


def test_str2int():
    # String representation of a product state.
    s = "0010010"
    assert 18 == psr.str2int(s)


def test_int2str():
    # Number of spin-orbitals in the system
    n = 7
    # Integer representation of a product state.
    i = 18
    assert "0010010" == psr.int2str(i, n)


def test_str2bitarray():
    # String representation of a product state.
    s = "0010010"
    assert bitarray('0010010') == psr.str2bitarray(s)


def test_bitarray2str():
    # Bitarray representation of a product state.
    bits = bitarray('0010010')
    assert "0010010" == psr.bitarray2str(bits)


def test_str2bytes():
    # String representation of a product state.
    s = "0010010"
    assert b'$' == psr.str2bytes(s)


def test_bytes2str():
    # Number of spin-orbitals in the system
    n = 7
    # Bytes representation of a product state.
    bytestr = b'$'
    assert "0010010" == psr.bytes2str(bytestr, n)


def test_int2bitarray():
    # Number of spin-orbitals in the system
    n = 7
    # Integer representation of a product state.
    i = 18
    assert bitarray('0010010') == psr.int2bitarray(i, n)


def test_bitarray2int():
    # Bitarray representation of a product state.
    bits = bitarray('0010010')
    assert 18 == psr.bitarray2int(bits)


def test_int2bytes():
    # Number of spin-orbitals in the system
    n = 7
    # Integer representation of a product state.
    i = 18
    assert b'$' == psr.int2bytes(i, n)


def test_bytes2int():
    # Number of spin-orbitals in the system
    n = 7
    # Bytes representation of a product state.
    bytestr = b'$'
    assert 18 == psr.bytes2int(bytestr, n)


def test_bitarray2bytes():
    # Bitarray representation of a product state.
    bits = bitarray('0010010')
    assert b'$' == psr.bitarray2bytes(bits)


def test_bytes2bitarray():
    # Number of spin-orbitals in the system
    n = 7
    # Bytes representation of a product state.
    bytestr = b'$'
    assert bitarray('0010010') == psr.bytes2bitarray(bytestr, n)
