.. -*- mode: rst -*-

nistats
=======

Nistats is a Python module for fast and easy modeling and statistical analysis
of functional Magnetic Resonance Imaging data.

It leverages the `nilearn <http://nilearn.github.io>`_ Python toolbox for
neuroimaging data manipulation (data downloading, visualization, masking).

This work is made available by a community of people, amongst which
the INRIA Parietal Project Team and D'esposito lab at Berkeley, in particular
G. Varoquaux, B. Thirion, J.B. Poline and M. Brett.

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
* Numpy >= 1.9.0
* SciPy >= 0.14.0
* Nibabel >= 1.2.0
* Nilearn >= 0.2.0
* Pandas >= 0.17.1
* Sklearn >= 0.15.2
* Patsy >= 0.2.0

If you are using nilearn plotting functionalities or running the
examples, matplotlib >= 1.4.0 is required.

If you want to run the tests, you need nose >= 1.2.1 and coverage >= 3.6.


Install
=======

In order to perform the installation, run the following command from the nistats directory::

    python setup.py install --user


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


