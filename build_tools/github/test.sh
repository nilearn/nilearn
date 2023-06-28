#!/bin/bash -x

python -m pytest --pyargs nilearn --cov=nilearn nilearn/regions/tests/test_parcellations.py
