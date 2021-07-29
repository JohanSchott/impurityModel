#!/bin/bash -ex

which python

source ~/envED/bin/activate

which python

# Script folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH=$DIR

