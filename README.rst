.. -*- mode: rst -*-

nilearn
=======

NiLearn is a Python module for fast and easy statistical learning on
NeuroImaging data.

It leverages the `scikit-learn <http://scikit-learn.org>`_ Python toolbox for multivariate
statistics with applications such as predictive modelling,
classification, decoding, or connectivity analysis.

This work is made available by the INRIA Parietal Project Team and the
scikit-learn folks, among which P. Gervais, A. Abraham, V. Michel, A.
Gramfort, G. Varoquaux, F. Pedregosa, B. Thirion, M. Eickenberg, C. F. Gorgolewski,
D. Bzdok and L. Estève

Important links
===============

- Official source code repo: https://github.com/nilearn/nilearn/
- HTML documentation (stable release): http://nilearn.github.com/

Dependencies
============

The required dependencies to use the software are:

* Python >= 2.6,
* setuptools
* Numpy >= 1.3
* SciPy >= 0.7
* Scikit-learn >= 0.12.1
* Nibabel >= 1.1.0.
This configuration almost matches the Ubuntu 10.04 LTS release from
April 2010, except for scikit-learn, which must be installed separately.

Running the examples requires matplotlib >= 0.99.1

If you want to run the tests, you need nose >= 1.2.1 and coverage >= 3.6.


Install
=======

The simplest is to use pip. Not that nilearn has been released as an
alpha so you need to use the ``--pre`` command-line parameter::

    pip install -U --pre --user nilearn


Development
===========

Build Status
------------
.. |travis-master| image:: https://travis-ci.org/nilearn/nilearn.svg?branch=master
   :target: https://travis-ci.org/nilearn/nilearn
   :alt: Build Status

|travis-master|

Code
----

GIT
~~~

You can check the latest sources with the command::

    git clone git://github.com/nilearn/nilearn

or if you have write privileges::

    git clone git@github.com:nilearn/nilearn


