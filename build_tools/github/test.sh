#!/bin/bash -x

python -m pytest --pyargs nilearn/regions/tests/test_parcellations.py --cov=nilearn
