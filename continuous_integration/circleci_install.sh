#!/usr/bin/env bash

echo 'export PATH="$HOME/miniconda3/bin:$PATH"'  >> $BASH_ENV
conda create -n testenv -y
source activate testenv
conda install python=3.5.2 numpy scipy scikit-learn matplotlib pandas flake8 lxml nose cython mkl sphinx coverage pillow pandas -yq
conda install nibabel nose-timer -c conda-forge
pip install .

