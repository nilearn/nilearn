#!/bin/bash -ef

conda init bash
echo "conda version = $(conda --version)"
conda create -n testenv
conda install -n testenv -yq python=3.8 numpy scipy scikit-learn matplotlib pandas lxml mkl sphinx numpydoc pillow pandas
conda install -n testenv -yq nibabel python-kaleido sphinx-gallery sphinxcontrib-bibtex sphinx-copybutton junit-xml -c conda-forge
conda install -c plotly plotly
source activate testenv
python -m pip install --user --upgrade --progress-bar off pip setuptools sphinxext-opengraph
python -m pip install .
