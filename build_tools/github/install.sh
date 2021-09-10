#!/bin/bash -e

python setup.py build
if [ ! -z "$MIN_REQUIREMENTS" ]; then
    pip install --progress-bar off --upgrade -e .[min]
else
    pip install --progress-bar off --upgrade -e .[test]
fi
