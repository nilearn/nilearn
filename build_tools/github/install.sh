#!/bin/bash -e

python setup.py build
if [ ! -z "$MIN_REQUIREMENTS" ]; then
    # Install the oldest supported versions of all required dependencies
    # See setup.cfg for dependency group options
    pip install --progress-bar off --upgrade -e .[min,test]
else
    # Install the newest supported versions of required and testing-related dependencies
    pip install --progress-bar off --upgrade -e .[plotting,test]
fi
