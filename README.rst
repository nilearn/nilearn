.. -*- mode: rst -*-

.. image:: https://travis-ci.org/nilearn/nilearn.svg?branch=master
   :target: https://travis-ci.org/nilearn/nilearn
   :alt: Travis Build Status

.. image:: https://ci.appveyor.com/api/projects/status/github/nilearn/nilearn?branch=master&svg=true
   :target: https://ci.appveyor.com/project/nilearn-ci/nilearn
   :alt: AppVeyor Build Status

.. image:: https://codecov.io/gh/nilearn/nilearn/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/nilearn/nilearn

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

* Python >= 2.7,
* setuptools
* Numpy >= 1.6.1
* SciPy >= 0.9
* Scikit-learn >= 0.14.1
* Nibabel >= 1.2.0

If you are using nilearn plotting functionalities or running the
examples, matplotlib >= 1.1.1 is required.

If you want to run the tests, you need nose >= 1.2.1 and coverage >= 3.6.


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
