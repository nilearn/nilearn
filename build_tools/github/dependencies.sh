#!/bin/bash -e

python -m pip install --progress-bar off --upgrade pip setuptools wheel
if [ ! -z "$MIN_REQUIREMENTS" ]; then
    pip install --progress-bar off --upgrade -r requirements-min.txt
else
    pip install --progress-bar off --upgrade -r requirements-dev.txt
fi

if [[ -n "$FLAKE8" ]]; then
    echo "Installing Flake8";
    pip install flake8
fi
