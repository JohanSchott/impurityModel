#!/bin/bash

# Change to script folder.
cd "$( dirname "${BASH_SOURCE[0]}" )"

# Create virtual environment. But only if it does not already exist.
test -d ~/envED || virtualenv -p python3.7 ~/envED

# Activate virtual environment and append to PYTHONPATH.
source env.sh

# Install required python libraries.
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run unit-tests
pytest

