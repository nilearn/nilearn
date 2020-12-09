#!/bin/bash -ef

echo 'python -m pytest --pyargs nilearn --cov=nilearn'
python -m pytest --pyargs nilearn --cov=nilearn
