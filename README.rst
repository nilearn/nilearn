.. -*- mode: rst -*-

nisl
====

Important links
===============

- Official source code repo: https://github.com/nisl/tutorial/
- HTML documentation (stable release): http://nisl.github.com/

Dependencies
============

The required dependencies to build the software are Python >= 2.6,
setuptools, Numpy >= 1.3, SciPy >= 0.7, Scikit-learn >= 0.10 and a working
C/C++ compiler.
This configuration matches the Ubuntu 10.04 LTS release from April 2010.

To run the tests you will also need nose >= 0.10.


Install
=======

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

  python setup.py install --home

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

    git clone git://github.com/nisl/tutorial

or if you have write privileges::

    git clone git@github.com:nisl/tutorial


Testing
-------
