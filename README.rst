.. -*- mode: rst -*-

.. image:: https://travis-ci.org/nilearn/nilearn.svg?branch=master
   :target: https://travis-ci.org/nilearn/nilearn
   :alt: Travis Build Status

.. image:: https://codecov.io/gh/nilearn/nilearn/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/nilearn/nilearn

.. image:: https://dev.azure.com/Parietal/Nilearn/_apis/build/status/nilearn.nilearn?branchName=master
    :target: https://dev.azure.com/Parietal/Nilearn/_apis/build/status/nilearn.nilearn?branchName=master

nilearn
=======

Nilearn is a Python module for fast and easy statistical learning on
NeuroImaging data.

It leverages the `scikit-learn <http://scikit-learn.org>`_ Python toolbox for multivariate
statistics with applications such as predictive modelling,
classification, decoding, or connectivity analysis.

This work is made available by a community of people, amongst which
the INRIA Parietal Project Team and the scikit-learn folks, in particular
P. Gervais, A. Abraham, V. Michel, A.
Gramfort, G. Varoquaux, F. Pedregosa, B. Thirion, M. Eickenberg, C. F. Gorgolewski,
D. Bzdok, L. Esteve and B. Cipollini.

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
* Joblib >= 0.11
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
http://nilearn.github.io/contributing.html
