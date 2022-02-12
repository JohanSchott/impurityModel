#!/bin/bash -ex

# Change to script folder.
cd "$( dirname "${BASH_SOURCE[0]}" )"

# System libraries
if [ "$(uname)" == "Darwin" ]; then
    # Mac OS X specific things here
    if ! which gfortran; then
        echo Install gfortran
        ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
        brew update
        brew doctor
        brew install gcc
        brew install swig
    fi
    if ! which mpirun; then
        echo Install Open-MPI
        curl -O https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz
        mv openmpi-4.1.1.tar.gz /tmp/
        cd /tmp
        tar -xf openmpi-4.1.1.tar.gz
        cd openmpi-4.1.1/
        mkdir /usr/local/openmpi
        ./configure --prefix=/usr/local/openmpi
        make all
        make install
        export PATH=${PATH}:/usr/local/openmpi/bin
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
pip install --disable-pip-version-check -q -U pip==21.2.4
pip install --disable-pip-version-check -q pip-tools==6.1.0
rm -f requirements.txt
pip-compile -q requirements.in
pip install --disable-pip-version-check -q -r requirements.txt
