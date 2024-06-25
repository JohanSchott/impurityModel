#!/bin/bash -ex

# Change to script folder.
cd "$( dirname "${BASH_SOURCE[0]}" )"

# System libraries
if [ "$(uname)" == "Darwin" ]; then
    # Mac OS X specific things here
    # If homebrew is not installed, install it by following the
    # instructions on https://brew.sh/
    if ! which gfortran; then
        echo Install gfortran
        brew install gcc
    fi
    if ! which mpirun; then
        echo Install Open-MPI
        brew install open-mpi
    fi
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Only for Debian, Ubuntu, and related Linux distributions
    sudo apt-get update -qq -y
    sudo apt-get install -qq -y --no-install-recommends $(cat requirements-ubuntu.txt)
else
    echo "Operating system not supported, yet"
    exit 1
fi

rm -rf ~/envED
python3 -m venv ~/envED

# Activate virtual environment.
. ~/envED/bin/activate

# Install required python libraries.
pip install --disable-pip-version-check -q -U uv==0.2.15
uv pip install -q -r requirements.in