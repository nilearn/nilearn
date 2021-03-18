#!/bin/bash -ef

sudo -E apt-get -yq update
sudo -E apt-get -yq --no-install-suggests --no-install-recommends --force-yes install dvipng texlive-latex-base texlive-latex-extra
python -m pip install --user --upgrade --progress-bar off pip setuptools
python -m pip install --user --upgrade --progress-bar off -r requirements-dev.txt -r requirements-build-docs.txt
python -m pip install --user -e .
