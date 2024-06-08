# Impurity model

![CI](https://github.com/JohanSchott/impurityModel/actions/workflows/buildci.yml/badge.svg?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


### Introduction

Calculate many-body states of an impurity Anderson model and a various spectra, e.g. photoemission spectroscopy (PS), x-ray photoemission spectroscopy (XPS), x-ray absorption spectroscopy (XAS), non-resonant inelastic x-ray scattering (NIXS), and resonant inelastic x-ray scattering (RIXS), using the [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm).


<figure>
<div class="row">
  <div class="column">
  <img src="impurityModel/test/referenceOutput/Ni_NiO_50bath/ps.png" alt="Photoemission (PS)" width="150"/>
  <img src="impurityModel/test/referenceOutput/Ni_NiO_50bath/xps.png" alt="X-ray photoemission (XPS)" width="150"/>
  <img src="impurityModel/test/referenceOutput/Ni_NiO_50bath/xas.png" alt="X-ray absorption spectroscopy (XAS)" width="150"/>
  <img src="impurityModel/test/referenceOutput/Ni_NiO_50bath/nixs.png" alt="Non-resonant inelastic x-ray scattering (NIXS)" width="150"/>
  <img src="impurityModel/test/referenceOutput/Ni_NiO_50bath/rixs.png" alt="Resonant inelastic x-ray scattering (RIXS)" width="150"/>  </div>
</div>
<figcaption>Spectra of NiO. Simulated using 50 bath orbitals coupled to the Ni 3d orbitals.</figcaption>
</figure>

### Get started
- Execute the bash-script `install_setup.sh`:
```bash
./install_setup.sh
```
This will create a Python virtual environment and install the required system and Python packages.

- Activate the virtual environment and set the PYTHONPATH:
```bash
source env.sh
```

- To run the unit-tests, type
```bash
pytest
```

- To check linting and static type checking, type
```bash
make check
```

- To perform a simulation, first create a directory somewhere on your computer.
Move to that directory and then execute one of the example scripts in the `scripts` folder. E.g. type:
```bash
path/to/folder/impurityModel/scripts/run_Ni_NiO_Xbath.sh
```
This will start a simulation with 10 bath states and one MPI rank.
To have e.g. 20 bath states instead of 10, instead type:
```bash
path/to/folder/impurityModel/scripts/run_Ni_NiO_Xbath.sh 20
```
To have e.g. 20 bath states and 3 MPI ranks, instead type:
```bash
path/to/folder/impurityModel/scripts/run_Ni_NiO_Xbath.sh 20 3
```
These examples will read an non-interacting Hamiltonian from file.

A simpler non-interacting Hamiltonian can instead be constructed by crystal-field parameters.
This is done for NiO by typing:
```bash
path/to/folder/impurityModel/scripts/run_Ni_NiO_CFparam.sh
```
Although using a crystal-field approach is a bigger approximation, it is convinient when doing fitting to experimental spectra.
But for more accurate simulations it is better to read in a non-interacting Hamiltonian from file, that has been constructed from e.g. DFT or DFT+DMFT simulations.
The non-interacting Hamiltonians read from file by the scripts `run_Ni_NiO_Xbath.sh` and `run_Ni_NiO_Xbath.sh` have been constructed using non-spin polarized DFT calculations.

- The bash-scripts in the `scripts`-folder act as templates and can easily be modified. For example, to set the temperature to 10 Kelvin in `get_spectra.py`, add `--T 10` as input when calling the python-script.

#### Output files
The input parameters to the simulation are saved in `.npz` format.
Calculated spectra are saved to the file `spectra.h5`.
Some small size spectra are also stored in `.dat` and `.bin` format, for easy and fast plotting with e.g. gnuplot.
For plotting all generated spectra (using matplotlib), type:
```bash
python -m impurityModel.plotScripts.plotSpectra
```
For only plotting the RIXS map, type:
```bash
python -m impurityModel.plotScripts.plotRIXS
```
Using Gnuplot, instead type:
```bash
path/to/folder/impurityModel/impurityModel/plotScripts/plotRIXS.plt
```

### Optimization notes

#### Computational speed
MPI is used.
For finding the ground states and calculating the spectra (except for RIXS), parallelization is done over the product states in the many-body basis.
For the RIXS simulations, parallelization is by default first done over product states of the core-hole excited system and then over the in-coming photon energies.

#### RAM memory usage
The memory goes primarly to storing the Hamiltonian in a basis of product states.
This Hamiltonian is stored as a dictionary, with product states, |ps>, as dictionary-keys
and the Hamiltonian acting of each product state, H|ps>, as dictionary-values.
When several ranks are used, the information is distributed over the MPI ranks, such that one rank only stores
some of all the product-state keys. This reduces memory usage for each MPI rank.

A sparse matrix format of the Hamiltonian is used when generating a spectrum.
This sparse matrix variable is also distributed in memory over the MPI ranks.
This is done to reduce memory usage per MPI rank.

A product state with electrons in spin-orbitals with indices e.g. 2 and 5 can be described by the tuple: (2,5).
If the system has 7 spin-orbitals in total, the product state can also be described by the binary-string "0010010".
The product state can also be translated into the base-2 integer 2^(7-1-2) + 2^(7-1-5) = 2^4 + 2^1 = 16+2 = 18.
With many electrons the integer representation is a more memory efficient format.
Bitarray is a class which also can be used to represent a product state.
It is mutable which enables fast modifications (adding and removing electrons), and is used in the current version.
To keep the memory usage down, an imutable bytes class is also used in the current version.

### Tests
Type
```bash
pytest
``` 
and 
```bash
pytest impurityModel/test/test_comparison_with_reference.py
```
to run all python unit tests in the repository.

### Documentation
The documentation of this package is found in the directory `doc`.

To update the manual type:

```bash
make -s -C doc/sphinx clean
make -s -C doc/sphinx html
```
Open the generated `doc/sphinx/generated_doc/html/index.html` in a web browser.

### Contributors

Call for contributions: The impurityModel project welcomes your expertise and enthusiasm!

Contributors:
- Johan Schött (@JohanSchott): Implemented many of the functionalities needed to solve the impurity problem.
- Johan Jönsson (@johanjoensson): Implementented the entire DMFT cycle in the fork https://github.com/johanjoensson/impurityModel. Also developed the related repo: https://github.com/johanjoensson/rspt2spectra
- Felix Sorgenfrei (@fesorg): First author in [publication](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.109.115126) using the impurityModel repo. Has also made the related repo: https://github.com/fesorg/Tutorial-X-ray-from-RSPt
- Patrik Thunström (@patrikthunstrom): Involved in discussions about computational algorithms, reported bugs, and has provided theoretical knowledge and inspiration.
- Petter Säterskog (@PetterSaterskog): Written some of the initial key functionalities.
- Christian Häggström (@kalvdans): Has provided valuable reviews on PRs.
- Mébarek Alouani: Has provided theoretical knowledge and inspiration.
- Olle Eriksson: Has provided theoretical knowledge and inspiration.
- Igor Di Marco (@igordimarco): Has provided theoretical knowledge and inspiration.

Please note that the list and the contribution information are incomplete.


### Publications using impurityModel

[Theory of x-ray absorption spectroscopy for ferrites](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.109.115126)

[Re-Dichalcogenides: Resolving Conflicts of TheirStructure–Property Relationship](https://onlinelibrary.wiley.com/doi/epdf/10.1002/apxr.202200010)

