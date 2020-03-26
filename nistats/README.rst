.. -*- mode: rst -*-

Nistats
=======

Nistats is a Python module for fast and easy modeling and statistical analysis
of functional Magnetic Resonance Imaging data.

It leverages the `nilearn <http://nilearn.github.io>`_ Python toolbox for
neuroimaging data manipulation (data downloading, visualization, masking).

This work is made available by a
`community <https://github.com/nistats/nistats/graphs/contributors>`_ of 
people, amongst which the INRIA Parietal Project Team and D'esposito lab at 
Berkeley.

It is based on developments initiated in the nipy
`nipy <http://nipy.org/nipy/stable>`_ project.

Important links
===============

- Official source code repo: https://github.com/nistats/nistats/
- HTML documentation (stable release): http://nistats.github.io/

Dependencies
============

The required dependencies to use the software are:

* Python >= 2.7
* setuptools
* Numpy >= 1.11
* SciPy >= 0.17
* Nibabel >= 2.0.2
* Nilearn >= 0.4.0
* Pandas >= 0.18.0
* Sklearn >= 0.18.0

If you are using nilearn plotting functionalities or running the
examples, matplotlib >= 1.5.1 is required.

If you want to run the tests, you need nose >= 1.2.1 and coverage >= 3.7.

If you want to download openneuro datasets Boto3 >= 1.2 is required


Install
=======

The installation instructions are found on the webpage:
https://nistats.github.io/

Development
===========

Code
----

GIT
~~~

You can check the latest sources with the command::

    git clone git://github.com/nistats/nistats

or if you have write privileges::

    git clone git@github.com:nistats/nistats
