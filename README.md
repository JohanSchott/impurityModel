# Impurity model
 
Calculate many-body states of an impurity Anderson model and spectra (e.g. XPS, XAS).

Depends (at the moment) on the open-source code RSPt.

Examples scripts are stored in the `script` folder.

For the Fortran module to work, compilation of the source code is necessary:
```
f2py -c -m removecreate removecreate.f90
```

Credits to Petter Säterskog for inspiration and for some of the (key) functionality.



#### Documentation
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




