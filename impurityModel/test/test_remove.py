from impurityModel.ed import create, remove
from impurityModel.ed import product_state_representation as psr


def test_remove_simple_state():
    # Start with vaccum state (no electrons)
    integer = 0

    n_spin_orbitals = 7

    # Create three electrons and get state: |psi> = c2 c3 c5 |0>

    integer, amp = create.uint(n_spin_orbitals, 5, integer)
    integer, amp = create.uint(n_spin_orbitals, 3, integer)
    integer, amp = create.uint(n_spin_orbitals, 2, integer)
    assert psr.int2tuple(integer, n_spin_orbitals) == (2, 3, 5)

    # Create the other representations of the state
    t = psr.int2tuple(integer, n_spin_orbitals)
    string = psr.int2str(integer, n_spin_orbitals)
    bits = psr.int2bitarray(integer, n_spin_orbitals)
    b = psr.int2bytes(integer, n_spin_orbitals)

    # Remove an electron in spin-orbital 3, and then in 2, and then 4
    integer, amp = remove.uint(n_spin_orbitals, 3, integer)
    assert amp == -1
    assert psr.int2tuple(integer, n_spin_orbitals) == (2, 5)
    integer, amp = remove.uint(n_spin_orbitals, 2, integer)
    assert amp == 1
    assert psr.int2tuple(integer, n_spin_orbitals) == (5,)
    integer, amp = remove.uint(n_spin_orbitals, 4, integer)
    assert amp == 0
    assert integer == -1

    t, amp = remove.utuple(3, t)
    assert amp == -1
    assert t == (2, 5)
    t, amp = remove.utuple(2, t)
    assert amp == 1
    assert t == (5,)
    t, amp = remove.utuple(4, t)
    assert amp == 0
    assert t == ()

    string, amp = remove.ustr(3, string)
    assert amp == -1
    assert psr.str2tuple(string) == (2, 5)
    string, amp = remove.ustr(2, string)
    assert amp == 1
    assert psr.str2tuple(string) == (5,)
    string, amp = remove.ustr(4, string)
    assert amp == 0
    assert string == ""

    amp = remove.ubitarray(3, bits)
    assert amp == -1
    assert psr.bitarray2tuple(bits) == (2, 5)
    amp = remove.ubitarray(2, bits)
    assert amp == 1
    assert psr.bitarray2tuple(bits) == (5,)
    amp = remove.ubitarray(4, bits)
    assert amp == 0
    assert psr.bitarray2tuple(bits) == (5,)

    b, amp = remove.ubytes(n_spin_orbitals, 3, b)
    assert amp == -1
    assert psr.bytes2tuple(b, n_spin_orbitals) == (2, 5)
    b, amp = remove.ubytes(n_spin_orbitals, 2, b)
    assert amp == 1
    assert psr.bytes2tuple(b, n_spin_orbitals) == (5,)
    b, amp = remove.ubytes(n_spin_orbitals, 4, b)
    assert amp == 0
    assert psr.bytes2tuple(b, n_spin_orbitals) == (5,)
