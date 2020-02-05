#!/bin/bash

source ~/env/bin/activate 

# Script folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Parent folder
DIR="${DIR}/../"
export PYTHONPATH=$PYTHONPATH:$DIR

