#!/bin/bash -e

python -m pip install --progress-bar off --upgrade $PIP_FLAGS pip build flake8
