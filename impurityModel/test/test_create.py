from impurityModel.ed import create
from impurityModel.ed import product_state_representation as psr


def test_create_simple_state():
    # Start with vaccum state (no electrons)
    integer = 0

    n_spin_orbitals = 7

    # Create the other representations of the state
    t = psr.int2tuple(integer, n_spin_orbitals)
    string = psr.int2str(integer, n_spin_orbitals)
    bits = psr.int2bitarray(integer, n_spin_orbitals)
    b = psr.int2bytes(integer, n_spin_orbitals)

    # Create two electrons and get state: |psi> = c2 c5 |0>
    integer_new, amp = create.uint(n_spin_orbitals, 5, integer)
    assert amp == 1
    integer_new, amp = create.uint(n_spin_orbitals, 2, integer_new)
    assert amp == 1
    assert psr.int2tuple(integer_new, n_spin_orbitals) == (2, 5)

    t_new, amp = create.utuple(5, t)
    assert amp == 1
    t_new, amp = create.utuple(2, t_new)
    assert amp == 1
    assert t_new == (2, 5)

    string_new, amp = create.ustr(5, string)
    assert amp == 1
    string_new, amp = create.ustr(2, string_new)
    assert amp == 1
    assert psr.str2tuple(string_new) == (2, 5)

    bits_new = bits.copy()
    amp = create.ubitarray(5, bits_new)
    assert amp == 1
    amp = create.ubitarray(2, bits_new)
    assert amp == 1
    assert psr.bitarray2tuple(bits_new) == (2, 5)

    b_new, amp = create.ubytes(n_spin_orbitals, 5, b)
    assert amp == 1
    b_new, amp = create.ubytes(n_spin_orbitals, 2, b_new)
    assert amp == 1
    assert psr.bytes2tuple(b_new, n_spin_orbitals) == (2, 5)

    # Check that get a negative sign when create electron at spin-orbital 3,
    # since there is an odd number of eletrons with lower spin-orbital index (one electron in spin-orbital 2)

    integer_new, amp = create.uint(n_spin_orbitals, 3, integer_new)
    assert amp == -1
    assert psr.int2tuple(integer_new, n_spin_orbitals) == (2, 3, 5)

    t_new, amp = create.utuple(3, t_new)
    assert amp == -1
    assert t_new == (2, 3, 5)

    string_new, amp = create.ustr(3, string_new)
    assert amp == -1
    assert psr.str2tuple(string_new) == (2, 3, 5)

    amp = create.ubitarray(3, bits_new)
    assert amp == -1
    assert psr.bitarray2tuple(bits_new) == (2, 3, 5)

    b_new, amp = create.ubytes(n_spin_orbitals, 3, b_new)
    assert amp == -1
    assert psr.bytes2tuple(b_new, n_spin_orbitals) == (2, 3, 5)

    # Check that get zero amplitude when create electron at spin-orbital 3

    integer_new, amp = create.uint(n_spin_orbitals, 3, integer_new)
    assert amp == 0
    assert integer_new == -1

    t_new, amp = create.utuple(3, t_new)
    assert amp == 0
    assert t_new == ()

    string_new, amp = create.ustr(3, string_new)
    assert amp == 0
    assert string_new == ""

    amp = create.ubitarray(3, bits_new)
    assert amp == 0
    assert psr.bitarray2tuple(bits_new) == (2, 3, 5)

    b_new, amp = create.ubytes(n_spin_orbitals, 3, b_new)
    assert amp == 0
    assert psr.bytes2tuple(b_new, n_spin_orbitals) == (2, 3, 5)
