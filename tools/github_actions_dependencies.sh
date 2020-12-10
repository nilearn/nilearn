#!/bin/bash -ef

python -m pip install --progress-bar off --upgrade setuptools wheel
pip uninstall -yq numpy
pip install --progress-bar off --upgrade -r requirements-min.txt
