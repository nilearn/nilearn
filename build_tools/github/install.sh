#!/bin/bash -e

python setup.py build
if [ ! -z "$MIN_REQUIREMENTS" ]; then
    pip install --progress-bar off --upgrade -r -e .
else
    pip install --progress-bar off --upgrade -r -e .[dev]
fi
