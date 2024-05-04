from impurityModel.ed import product_state_representation as psr
from impurityModel.ed.finite import applyOp


def test_applyOp_create_one_electron_from_vacuum():
    vacuum_as_string = "000000"
    n_spin_orbitals = len(vacuum_as_string)
    # Multi-configurational state is a single product state
    psi = {psr.str2bytes(vacuum_as_string): 7}
    for i in range(n_spin_orbitals):
        op = {((i, "c"),): 9}
        psi_new = applyOp(n_spin_orbitals, op, psi)
        assert psi_new == {psr.str2bytes(vacuum_as_string[:i] + "1" + vacuum_as_string[i + 1 :]): 7 * 9}


def test_applyOp_create_one_electron_from_one_electron_state():
    product_state = "001000"
    n_spin_orbitals = len(product_state)
    # Multi-configurational state is a single product state
    psi = {psr.str2bytes(product_state): 7}
    for i in range(n_spin_orbitals):
        op = {((i, "c"),): 9}
        psi_new = applyOp(n_spin_orbitals, op, psi)
        index_of_spin_orbital_with_one_electron_already = 2
        if i == index_of_spin_orbital_with_one_electron_already:
            # Can not put two electrons in the same spin orbital
            assert psi_new == {}
        else:
            amp = 7 * 9 * (2 * (i < index_of_spin_orbital_with_one_electron_already) - 1)
            assert psi_new == {psr.str2bytes(product_state[:i] + "1" + product_state[i + 1 :]): amp}


def test_applyOp_create_two_electrons():
    vacuum_as_string = "000000"
    n_spin_orbitals = len(vacuum_as_string)
    # Multi-configurational state is a single product state
    psi = {psr.str2bytes(vacuum_as_string): 7}
    for i in range(n_spin_orbitals):
        for j in range(n_spin_orbitals):
            op = {((i, "c"), (j, "c")): 9}
            psi_new = applyOp(n_spin_orbitals, op, psi)
            if i == j:
                # Can not put two electrons in the same spin orbital
                assert psi_new == {}
            else:
                a, b = min(i, j), max(i, j)
                product_state = psr.str2bytes(
                    vacuum_as_string[:a] + "1" + vacuum_as_string[a + 1 : b] + "1" + vacuum_as_string[b + 1 :]
                )
                amp = 7 * 9 * (2 * (i < j) - 1)
                assert psi_new == {product_state: amp}


def test_applyOp_two_creation_processes():
    vacuum_as_string = "000000"
    n_spin_orbitals = len(vacuum_as_string)
    # Multi-configurational state is a single product state
    psi = {psr.str2bytes(vacuum_as_string): 7}
    for i in range(n_spin_orbitals):
        for j in range(n_spin_orbitals):
            if i == j:
                continue
            op = {((i, "c"),): 9, ((j, "c"),): 11}
            psi_new = applyOp(n_spin_orbitals, op, psi)
            assert psi_new == {
                psr.str2bytes(vacuum_as_string[:i] + "1" + vacuum_as_string[i + 1 :]): 7 * 9,
                psr.str2bytes(vacuum_as_string[:j] + "1" + vacuum_as_string[j + 1 :]): 7 * 11,
            }


def test_applyOp_remove_one_electron():
    vacuum_as_string = "000000"
    n_spin_orbitals = len(vacuum_as_string)
    # Multi-configurational state is a single product state
    psi = {psr.str2bytes(vacuum_as_string): 7}
    for i in range(n_spin_orbitals):
        op = {((i, "a"),): 9}
        psi_new = applyOp(n_spin_orbitals, op, psi)
        # Can't remove an electron from un-occupied spin-orbital
        assert psi_new == {}


def test_applyOp_two_processes_but_same_final_state():
    product_state_as_string = "011000"
    n_spin_orbitals = len(product_state_as_string)
    # Multi-configurational state is a single product state
    psi = {psr.str2bytes(product_state_as_string): 7}
    op = {((1, "c"), (1, "a")): 9, ((2, "c"), (2, "a")): 11}
    psi_new = applyOp(n_spin_orbitals, op, psi)
    assert psi_new == {psr.str2bytes(product_state_as_string): 7 * (9 + 11)}


def test_applyOp_opResult():
    vacuum_as_string = "000000"
    n_spin_orbitals = len(vacuum_as_string)
    # Multi-configurational state is a single product state
    psi = {psr.str2bytes(vacuum_as_string): 7}
    for i in range(n_spin_orbitals):
        for j in range(n_spin_orbitals):
            if i == j:
                continue
            op = {((i, "c"),): 9, ((j, "c"),): 11}
            opResult = {}
            psi_new = applyOp(n_spin_orbitals, op, psi, opResult=opResult)
            psi_new_expected = {
                psr.str2bytes(vacuum_as_string[:i] + "1" + vacuum_as_string[i + 1 :]): 7 * 9,
                psr.str2bytes(vacuum_as_string[:j] + "1" + vacuum_as_string[j + 1 :]): 7 * 11,
            }
            assert psi_new == psi_new_expected
            # opResult stores how the operator acts on individual product states.
            assert opResult == {
                psr.str2bytes(vacuum_as_string): {
                    psr.str2bytes(vacuum_as_string[:i] + "1" + vacuum_as_string[i + 1 :]): 9,
                    psr.str2bytes(vacuum_as_string[:j] + "1" + vacuum_as_string[j + 1 :]): 11,
                }
            }
            # Now opResult is available and is used to (quickly) look-up the result
            psi_new = applyOp(n_spin_orbitals, op, psi, opResult=opResult)
            assert psi_new == psi_new_expected

            # Store a wrong result in opResult and see that it's used instead of op
            opResult = {psr.str2bytes(vacuum_as_string): {psr.str2bytes(n_spin_orbitals * "1"): 2}}
            psi_new = applyOp(n_spin_orbitals, op, psi, opResult=opResult)
            assert psi_new == {psr.str2bytes(n_spin_orbitals * "1"): 7 * 2}


def test_applyOp_restrictions():
    # Specify one restriction of the occupation (for each product state)
    spin_orbital_indices = frozenset([1, 4, 5])
    occupation_lower_and_upper_limits = (0, 1)
    restrictions = {spin_orbital_indices: occupation_lower_and_upper_limits}

    vacuum_as_string = "000000"
    n_spin_orbitals = len(vacuum_as_string)
    # Multi-configurational state is a single product state
    psi = {psr.str2bytes(vacuum_as_string): 7}
    for i in range(n_spin_orbitals):
        for j in range(n_spin_orbitals):
            op = {((i, "c"), (j, "c")): 9}
            psi_new = applyOp(n_spin_orbitals, op, psi, restrictions=restrictions)
            # Sanity check psi_new
            if i == j:
                # Never can not put two electrons in the same spin orbital
                assert psi_new == {}
            elif i in spin_orbital_indices and j in spin_orbital_indices:
                # The specified restrictions allow max one electron in spin-orbitals
                # with indices specified by the variable indices.
                assert psi_new == {}
            else:
                a, b = min(i, j), max(i, j)
                product_state = psr.str2bytes(
                    vacuum_as_string[:a] + "1" + vacuum_as_string[a + 1 : b] + "1" + vacuum_as_string[b + 1 :]
                )
                amp = 7 * 9 * (2 * (i < j) - 1)
                assert psi_new == {product_state: amp}
