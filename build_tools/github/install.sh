#!/bin/bash -e

python setup.py build
if [ ! -z "$MIN_REQUIREMENTS" ]; then
    pip install --progress-bar off --upgrade -e .
else
    pip install --progress-bar off --upgrade -e .[dev]
fi
