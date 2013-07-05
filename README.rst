.. -*- mode: rst -*-

nilearn
=======

This projects contains a tutorial on how to process functional Magnetic Resonance Imaging (fMRI) data with the scikit-learn.

This work is made available by the INRIA Parietal Project Team and the
scikit-learn folks, among which P. Gervais, A. Abraham, V. Michel, A.
Gramfort, G. Varoquaux, F. Pedregosa and B. Thirion.

Important links
===============

- Official source code repo: https://github.com/nilearn/nilearn/
- HTML documentation (stable release): http://nilearn.github.com/

Dependencies
============

The required dependencies to sue the software are Python >= 2.6,
setuptools, Numpy >= 1.3, SciPy >= 0.7, Scikit-learn >= 0.12.1
This configuration almost matches the Ubuntu 10.04 LTS release from
April 2010, except for scikit-learn, which must be installed separately.

Running the examples requires matplotlib >= 0.99.1

If you want to run the tests, you need recent python-coverage and python-nose.
(resp. 3.6 and 1.2.1).


Install
=======

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

  python setup.py install --user

To install for all users on Unix/Linux::

  python setup.py build
  sudo python setup.py install


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


