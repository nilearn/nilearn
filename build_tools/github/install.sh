#!/bin/bash -e

python -m build
if [ ! -z "$MIN_REQUIREMENTS" ]; then
    # Install the oldest supported versions of all required dependencies
    # See pyproject.toml for dependency group options
    if [ ! -z "$MATPLOTLIB" ]; then
        # Include plotting dependencies too
        pip install --progress-bar off --upgrade -e .[min,plotting,test]
    else
        pip install --progress-bar off --upgrade -e .[min,test]
    fi
else
    # Install the newest supported versions of required and testing-related dependencies
    pip install --progress-bar off $PIP_FLAGS --upgrade -e .[plotting,plotly,test]
fi
