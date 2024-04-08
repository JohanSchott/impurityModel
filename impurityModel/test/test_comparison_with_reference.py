"""
Module with test comparing new simulations with reference data.

test_comparison_with_reference.py is not allowed to import any MPI stuff, except in the subprocess.run.
Otherwise MPI gets confused, since MPI can't handle that both the parent and the child process use MPI.
"""

import math
import os
import subprocess
import tempfile

import h5py
import numpy as np

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

SCRIPT_PATH = os.path.join(DIR_PATH, "../../scripts/run_Ni_NiO_Xbath.sh")
REFERENCE_SPECTRA_PATH = os.path.join(DIR_PATH, "referenceOutput/Ni_NiO_50bath/spectra.h5")


def test_comparison():
    compare_spectra()


def compare_spectra(
    script_path=SCRIPT_PATH,
    script_argument=50,
    reference_spectra_path=REFERENCE_SPECTRA_PATH,
):
    # Create a temporary directory using the context manager
    with tempfile.TemporaryDirectory() as tmpdirname:
        print("Created temporary directory", tmpdirname)
        os.chdir(tmpdirname)
        print("Current working dir:", os.getcwd())
        print(f"{script_path = }")
        print(f"{script_argument = }")

        subprocess.run(args=[script_path, str(script_argument)], check=True)

        files_and_dirs = os.listdir()
        print("Files and folders in temporary folder:", files_and_dirs)
        assert os.path.isfile("spectra.h5")
        assert os.path.isfile(reference_spectra_path), reference_spectra_path
        # Open spectra file and the reference spectra file
        with h5py.File("spectra.h5", "r") as file_handle, h5py.File(reference_spectra_path, "r") as ref_file_handle:
            # Compare file contents
            for key in ref_file_handle:
                print("Compare dataset:", key)
                x = file_handle[key][()]
                x_ref = ref_file_handle[key][()]
                abs_diff = np.abs(x - x_ref)
                i = np.argmax(abs_diff)
                print("Reference value at max diff:", np.ravel(x_ref)[i])
                np.testing.assert_allclose(x, x_ref, atol=3e-2)
                np.testing.assert_allclose(x, x_ref, atol=2e-2, rtol=0.1)
                print("Mean abs diff:", np.mean(abs_diff))
                assert math.isclose(np.mean(abs_diff), 0, abs_tol=2e-5)
                print("Median abs diff:", np.median(abs_diff))
                assert math.isclose(np.median(abs_diff), 0, abs_tol=1e-8)
        print("Comparison successful")


if __name__ == "__main__":
    compare_spectra()
