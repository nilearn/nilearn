.. -*- mode: rst -*-

.. image:: https://travis-ci.org/nilearn/nilearn.svg?branch=master
   :target: https://travis-ci.org/nilearn/nilearn
   :alt: Build Status

.. image:: https://coveralls.io/repos/nilearn/nilearn/badge.svg?branch=master
   :target: https://coveralls.io/r/nilearn/nilearn

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
D. Bzdok, L. Estève and B. Cipollini.

Important links
===============

- Official source code repo: https://github.com/nilearn/nilearn/
- HTML documentation (stable release): http://nilearn.github.io/

Dependencies
============

The required dependencies to use the software are:

* Python >= 2.6,
* setuptools
* Numpy >= 1.6.1
* SciPy >= 0.9
* Scikit-learn >= 0.13 (Some examples require 0.14 to run)
* Nibabel >= 1.1.0

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

Code
----

GIT
~~~

You can check the latest sources with the command::

    git clone git://github.com/nilearn/nilearn

or if you have write privileges::

    git clone git@github.com:nilearn/nilearn


