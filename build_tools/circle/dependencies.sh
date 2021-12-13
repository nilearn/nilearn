#!/bin/bash -ef

conda init bash
echo "conda version = $(conda --version)"
conda create -n testenv
conda install -n testenv -yq python=3.8 numpy scipy scikit-learn matplotlib pandas lxml mkl sphinx numpydoc pillow pandas
conda install -n testenv -yq nibabel python-kaleido sphinx-gallery sphinxcontrib-bibtex sphinx-copybutton junit-xml -c conda-forge
conda install -n testenv -c plotly plotly
source activate testenv
python -m pip install --user --upgrade --progress-bar off pip setuptools sphinxext-opengraph memory_profiler
# Install the local version of the library, along with both standard and testing-related dependencies
# See setup.cfg for dependency group options
python -m pip install .[test]
