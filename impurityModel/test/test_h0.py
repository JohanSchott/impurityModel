import os
from glob import glob

from impurityModel.ed.finite import assert_hermitian
from impurityModel.ed.get_spectra import read_pickled_file

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def test_read_all_h0_pickle_files():
    for h0_filename in glob(os.path.join(DIR_PATH, "../../h0/h0*.pickle")):
        h0 = read_pickled_file(h0_filename)

        string = os.path.basename(h0_filename).split(".")[0].split("_")[-1].split("bath")[0]
        nBaths_tot = sum([int(nbath) for nbath in string.split("p")])
        # So far, in all the non-interacting Hamiltonians the angular momentum is equal to two
        # for the correlated orbitals that are coupled to the bath states.
        nBaths = {2: nBaths_tot}

        sanity_check_non_interacting_hamiltonian(h0, nBaths)


def sanity_check_non_interacting_hamiltonian(h0: dict[tuple, complex], nBaths: dict[int, int]):
    """
    Sanity check non-interacting Hamiltonian operator.

    Parameters
    -------
    h0:
        Non-interacting Hamiltonian operator.
        Describes impurity orbitals and bath orbitals.
        Each tuple key describes a process of two steps: annihilation followed by creation.
        Each step is described by a tuple of the form:
        (spin_orb, 'c') for creation, and
        (spin_orb, 'a') for annihilation,
        where spin_orb is a tuple of the form (l, s, m) or (l, b) or ((l_a, l_b), b).
    nBaths:
        angular momentum: number of bath states coupled to the correlated orbitals with this angular momentum.
    """
    assert_hermitian(h0)
    for process, value in h0.items():
        assert isinstance(value, complex)
        assert len(process) == 2  # two events in non-interacting Hamiltonian
        assert process[1][1] == "a"  # First event is annihilation
        assert process[0][1] == "c"  # Second event is creation
        for event in process[::-1]:
            assert len(event) == 2  # spin-orbit and create or remove
            spin_orbit_info = event[0]
            if len(spin_orbit_info) == 2:
                l, bath_index = spin_orbit_info
                assert 0 <= l <= 3
                assert 0 <= bath_index < nBaths[l]
            elif len(spin_orbit_info) == 3:
                l, s, m = spin_orbit_info
                assert l in nBaths
                assert 0 <= l <= 3
                assert s in (0, 1)
                assert m in range(-l, l + 1)
            else:
                raise ValueError(f"{spin_orbit_info}")
