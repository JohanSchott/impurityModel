# applyOp(n_spin_orbitals, op, psi, slaterWeightMin=1e-12, restrictions=None, opResult=None)
# test operator contained two processes
# test to create one electron in already occupied spin-orbital
# test to remove one electron in empty spin-orbital
# test to create and remove from an existing product state.
# test opResult
# test restrictions

import numpy as np
import math

from impurityModel.ed.finite import applyOp
from impurityModel.ed import product_state_representation as psr


def test_applyOp_create_one_electron_from_vacuum():
    vacuum_as_string = "000000"
    n_spin_orbitals = len(vacuum_as_string)
    vacuum_as_bytes = psr.str2bytes(vacuum_as_string)
    # Multi-configurational state is a single product state
    psi = {vacuum_as_bytes: 7}
    for i in range(n_spin_orbitals):
        op = {((i, "c"),): 9}
        psi_new = applyOp(n_spin_orbitals, op, psi)
        # Check new multi-configurational state psi_new
        assert len(psi_new) == 1
        for state, amp in psi_new.items():
            assert math.isclose(amp, 7 * 9)
            state_as_string = psr.bytes2str(state, n_spin_orbitals)
            assert state_as_string == vacuum_as_string[:i] + "1" + vacuum_as_string[i + 1 :]


def test_applyOp_create_two_electrons_from_vacuum():
    vacuum_as_string = "000000"
    n_spin_orbitals = len(vacuum_as_string)
    vacuum_as_bytes = psr.str2bytes(vacuum_as_string)
    # Multi-configurational state is a single product state
    psi = {vacuum_as_bytes: 7}
    for i in range(n_spin_orbitals):
        for j in range(n_spin_orbitals):
            op = {((i, "c"), (j, "c")): 9}
            psi_new = applyOp(n_spin_orbitals, op, psi)
            # Check new multi-configurational state psi_new
            if i == j:
                # Can not put two electrons in the same spin orbital
                assert psi_new == {}
            else:
                assert len(psi_new) == 1
                for state, amp in psi_new.items():
                    assert math.isclose(amp, 7 * 9 * (2 * (i < j) - 1))
                    state_as_string = psr.bytes2str(state, n_spin_orbitals)
                    a, b = min(i, j), max(i, j)
                    assert (
                        state_as_string
                        == vacuum_as_string[:a] + "1" + vacuum_as_string[a + 1 : b] + "1" + vacuum_as_string[b + 1 :]
                    )


def test_applyOp_two_creation_processes_from_vacuum():
    vacuum_as_string = "000000"
    n_spin_orbitals = len(vacuum_as_string)
    vacuum_as_bytes = psr.str2bytes(vacuum_as_string)
    # Multi-configurational state is a single product state
    psi = {vacuum_as_bytes: 7}
    for i in range(n_spin_orbitals):
        for j in range(n_spin_orbitals):
            if i == j:
                continue
            op = {((i, "c"),): 9, ((j, "c"),): 11}
            psi_new = applyOp(n_spin_orbitals, op, psi)
            # Check new multi-configurational state psi_new
            assert len(psi_new) == 2
            for state, amp in psi_new.items():
                state_as_string = psr.bytes2str(state, n_spin_orbitals)
                if amp == 7 * 9:
                    a = i
                elif amp == 7 * 11:
                    a = j
                else:
                    raise ValueError(f"{amp = }, {state_as_string = }")
                assert state_as_string == vacuum_as_string[:a] + "1" + vacuum_as_string[a + 1 :]


def main():
    test_applyOp_create_from_vacuum()


if __name__ == "__main__":
    main()
