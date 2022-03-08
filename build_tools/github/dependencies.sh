#!/bin/bash -e

python -m pip install --progress-bar off --upgrade pip setuptools wheel flake8
if [ ! -z "$MIN_REQUIREMENTS" ]; then
    pip install --progress-bar off --upgrade -r requirements-min.txt
    if [ ! -z "$MATPLOTLIB" ]; then
        pip install --progress-bar off --upgrade matplotlib==3.0
    fi
else
    pip install --progress-bar off --upgrade -r requirements-dev.txt
fi
