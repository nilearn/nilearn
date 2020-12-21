#!/bin/bash -e

python -m pip install --progress-bar off --upgrade pip setuptools wheel
if [ ! -z "$MIN_REQUIREMENTS" ]; then
    pip install --progress-bar off --upgrade -r requirements-min.txt
else
    pip install --progress-bar off --upgrade -r requirements-dev.txt
fi

if [ ! -z "$MATPLOTLIB_DEV" ]; then
    pip install git+https://github.com/matplotlib/matplotlib.git
fi
