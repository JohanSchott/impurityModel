"""
test_comparison_with_reference
==============================

Module with test comparing new simulations with reference data.

"""


import subprocess
import os
import inspect
import tempfile
import h5py
import numpy as np
import math


def test_comparison():
    compare_spectra()


def compare_spectra(
    script_file="scripts/run_Ni_NiO_Xbath.sh",
    script_argument=50,
    reference_file="referenceOutput/Ni_NiO_50bath/spectra.h5",
):
    print("Start comparison of spectra...")
    # Create a temporary directory using the context manager
    with tempfile.TemporaryDirectory() as tmpdirname:
        print("Created temporary directory", tmpdirname)
        os.chdir(tmpdirname)
        print("Current working dir:", os.getcwd())
        path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        cmd = os.path.join(path[:-19], script_file)
        print("Run command:", cmd)
        print("Use command argument:", script_argument)
        subprocess.call([cmd, str(script_argument)])
        files_and_dirs = os.listdir()
        print("Files and folders in temporary folder:", files_and_dirs)
        # Open spectra file and the reference spectra file
        file_handle = h5py.File("spectra.h5", "r")
        ref_file_handle = h5py.File(os.path.join(path, reference_file), "r")
        # Compare file contents
        for key in ref_file_handle:
            print("Compare dataset:", key)
            x = file_handle[key][()]
            x_ref = ref_file_handle[key][()]
            abs_diff = np.abs(x - x_ref)
            i = np.argmax(abs_diff)
            print("Max abs diff:", np.ravel(abs_diff)[i])
            print("Reference value at max diff:", np.ravel(x_ref)[i])
            np.testing.assert_allclose(x, x_ref, atol=2e-2, rtol=0.1)
            print("Mean abs diff:", np.mean(abs_diff))
            assert math.isclose(np.mean(abs_diff), 0, abs_tol=2e-5)
            print("Median abs diff:", np.median(abs_diff))
            assert math.isclose(np.median(abs_diff), 0, abs_tol=1e-5)
        print("Comparison successful")


if __name__ == "__main__":
    compare_spectra()
