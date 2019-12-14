#!/usr/bin/env python3
"""
A package dealing with many-body impurity models.

    Examples of functionalities:
        - Calculate spectra, e.g. XAS, XPS, PS.
        - Calculate static expectation values

"""

from . import finite
from . import spectra
from . import average
from . import product_state_representation
from . import create
from . import remove
from . import mpi_comm

