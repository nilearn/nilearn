#!/bin/bash -ef

conda init bash
echo "conda version = $(conda --version)"
conda create -n testenv
conda install -n testenv -yq python=3.8
source activate testenv
python -m pip install --user --upgrade --progress-bar off pip setuptools
# Install the local version of the library, along with both standard and testing-related dependencies
# The `doc` dependency group is included because the build_docs job uses this script.
# See setup.cfg for dependency group options
python -m pip install .[plotting,plotly,test,doc]
