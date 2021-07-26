#!/bin/bash -ex

# Change to script folder.
cd "$( dirname "${BASH_SOURCE[0]}" )"

# Create virtual environment. But only if it does not already exist.
test -d ~/envED || virtualenv -p python3.7 ~/envED

# Activate virtual environment.
source ~/envED/bin/activate

# Install required python libraries.
python -m pip install --upgrade pip==21.1.2
python -m pip install pip-tools==6.1.0
pip-compile
pip install -r requirements.txt

