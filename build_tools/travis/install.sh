#!/bin/bash

set -e

echo "CPU Architecture: $TRAVIS_CPU_ARCH."

deactivate

if [[ $TRAVIS_CPU_ARCH == arm64 ]]; then
    MINICONDA_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
else
    MINICONDA_URL="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
fi

wget $MINICONDA_URL -O miniconda.sh
MINICONDA_PATH=$HOME/miniconda
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
export PATH=$MINICONDA_PATH/bin:$PATH
conda update --yes conda

conda init bash
conda create -n testenv -yq
conda install -n testenv -yq python=3.8 numpy scipy scikit-learn matplotlib pandas pytest joblib nibabel cython requests
source activate testenv
python -m pip install --user --upgrade --progress-bar off pip setuptools
python -m pip install .
