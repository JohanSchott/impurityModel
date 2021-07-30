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
    sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends $(cat requirements-ubuntu.txt)
else
    echo "Operating system not supported, yet"
    exit 1
fi

python3 -m pip install virtualenv
rm -r ~/envED
python3 -m virtualenv ~/envED

# Activate virtual environment.
. ~/envED/bin/activate

# Install required python libraries.
python -m pip install --upgrade pip==21.1.2
python -m pip install pip-tools==6.1.0
rm requirements.txt
pip-compile requirements.in
pip install -r requirements.txt

