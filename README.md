# Impurity model
 
Calculate many-body states of an impurity Anderson model and a few spectra, e.g. resonant inelastic x-ray scattering (RIXS), non-resonant inelastic x-ray scattering (NIXS), x-ray photo emission spectroscopy (XPS), photoemission spectroscopy (PS), and x-ray absorption spectroscopy (XAS).

Examples scripts are stored in the `scripts` folder.


Credits to Petter Saterskog for inspiration and for some of the (key) functionality.


### Get started
- Add the absolute path of the main directory (`impurityModel`) to the `PYTHONPATH` environment variable, such that the Python module in this directory can be found.

- Add the absolute path of the sub directories `impurityModel/scripts` and `impurityModel/plotScripts` to the `PATH` environment variable, such that the Python scripts in these directories can be found.

- Create a directory somewhere and execute one of the scripts in the `impurityModel/scripts` folder. E.g. type:
  ```
  mpirun -n 5 impurityMPI.py 
  ```

#### Output files
Input parameters used are saved and stored in `.npz` format.
Spectra are stored in either one `.h5` file or in many `.npz` files.
Some small size spectra are also stored in `.dat` and `.bin` format, for easy and fast plotting with e.g. gnuplot.
For plotting all generated spectra, type:
```
plotSpectra.py
```
For only plotting the RIXS map, type:
```
plotRIXS.py
```
or plot using gnuplot: 
```
plotRIXS.plt
```

### Optimization notes
MPI is used. 
For finding the ground states, parallelization is done over the product states in the considered basis.
For for the spectra, parallelization is done over eigenstates, except for RIXS where parallelization instead is done over the in-coming photon energies.

A Fortran module exists but is at the moment not used. If one would like to used it, compilation of the source code is necessary:
```
f2py -c -m removecreate removecreate.f90
```

### Documentation
The documentation of this package is found in the directory `docs`.

To update the manual, go to directory `docs` and simply type:

```
make html
```
to generate a html-page.
To instead generate a pdf-file, type:
```
make latex
```
and follow the instructions.

Note:
- package `numpydoc` is required. If missing, e.g. type `conda install numpydoc` 
- If a new module or subpackage is created, this information needs to be added to `docs/index.rst`. 




