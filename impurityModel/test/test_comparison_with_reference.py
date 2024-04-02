"""
Module with test comparing new simulations with reference data.
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

        output = subprocess.run(args=[script_path, str(script_argument)], check=False, capture_output=True)
        print(f"{output.returncode = }")
        print("output.stdout:")
        print(str(output.stdout, encoding="utf-8"))
        print("output.stderr:")
        print(str(output.stderr, encoding="utf-8"))
        assert output.returncode == 0

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
                print("Max abs diff:", np.ravel(abs_diff)[i])
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
