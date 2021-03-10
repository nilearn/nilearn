	.. -*- mode: rst -*-

.. image:: https://img.shields.io/pypi/v/nilearn.svg
    :target: https://pypi.org/project/nilearn/
    :alt: Pypi Package

.. image:: https://github.com/nilearn/nilearn/workflows/build/badge.svg?branch=master&event=push
   :target: https://github.com/nilearn/nilearn/actions
   :alt: Github Actions Build Status

.. image:: https://codecov.io/gh/nilearn/nilearn/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/nilearn/nilearn
   :alt: Coverage Status

.. image:: https://dev.azure.com/Parietal/Nilearn/_apis/build/status/nilearn.nilearn?branchName=master
   :target: https://dev.azure.com/Parietal/Nilearn/_apis/build/status/nilearn.nilearn?branchName=master
   :alt: Azure Build Status

nilearn
=======

Nilearn enables approachable and versatile analyses of brain volumes. It provides statistical and machine-learning tools, with instructive documentation & friendly community.

It supports general linear model (GLM) based analysis and leverages the `scikit-learn <http://scikit-learn.org>`_ Python toolbox for multivariate statistics with applications such as predictive modelling, classification, decoding, or connectivity analysis.

Important links
===============

- Official source code repo: https://github.com/nilearn/nilearn/
- HTML documentation (stable release): http://nilearn.github.io/

Dependencies
============

The required dependencies to use the software are:

* Python >= 3.5,
* setuptools
* Numpy >= 1.11
* SciPy >= 0.19
* Scikit-learn >= 0.19
* Joblib >= 0.12
* Nibabel >= 2.0.2

If you are using nilearn plotting functionalities or running the
examples, matplotlib >= 1.5.1 is required.

If you want to run the tests, you need pytest >= 3.9 and pytest-cov for coverage reporting.


Install
=======

First make sure you have installed all the dependencies listed above.
Then you can install nilearn by running the following command in
a command prompt::

    pip install -U --user nilearn

More detailed instructions are available at
http://nilearn.github.io/introduction.html#installation.

Development
===========

Detailed instructions on how to contribute are available at
http://nilearn.github.io/development.html
