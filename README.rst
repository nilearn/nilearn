	.. -*- mode: rst -*-

.. image:: https://img.shields.io/pypi/v/nilearn.svg
    :target: https://pypi.org/project/nilearn/
    :alt: Pypi Package

.. image:: https://img.shields.io/pypi/pyversions/nilearn.svg
    :target: https://pypi.org/project/nilearn/
    :alt: PyPI - Python Version

.. image:: https://github.com/nilearn/nilearn/workflows/build/badge.svg?branch=main&event=push
   :target: https://github.com/nilearn/nilearn/actions
   :alt: Github Actions Build Status

.. image:: https://codecov.io/gh/nilearn/nilearn/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/nilearn/nilearn
   :alt: Coverage Status

.. image:: https://dev.azure.com/Parietal/Nilearn/_apis/build/status/nilearn.nilearn?branchName=main
   :target: https://dev.azure.com/Parietal/Nilearn/_apis/build/status/nilearn.nilearn?branchName=main
   :alt: Azure Build Status

nilearn
=======

Nilearn enables approachable and versatile analyses of brain volumes. It provides statistical and machine-learning tools, with instructive documentation & friendly community.

It supports general linear model (GLM) based analysis and leverages the `scikit-learn <http://scikit-learn.org>`_ Python toolbox for multivariate statistics with applications such as predictive modelling, classification, decoding, or connectivity analysis.

Important links
===============

- Official source code repo: https://github.com/nilearn/nilearn/
- HTML documentation (stable release): http://nilearn.github.io/

Office Hours
============

The nilearn team organizes regular online office hours to answer questions, discuss feature requests, or have any Nilearn-related discussions. We try to maintain a frequency of *one hour every two weeks*, usually on Mondays, and make sure that at least one member of the core-developer team is available. These events are held on our on `Discord server <https://discord.gg/bMBhb7w>`_ and are fully open, anyone is welcome to join!

You can check when the next office hours will be held on the Nilearn's website `landing page <https://nilearn.github.io/>`_.

Dependencies
============

The required dependencies to use the software are:

* Python >= 3.6,
* setuptools
* Numpy >= 1.16
* SciPy >= 1.2
* Scikit-learn >= 0.21
* Joblib >= 0.12
* Nibabel >= 2.5
* Pandas >= 0.24

If you are using nilearn plotting functionalities or running the
examples, matplotlib >= 1.5.1 is required.

Some plotting functions in Nilearn support both matplotlib and plotly as plotting engines.
In order to use the plotly engine in these functions, you will need to install both plotly and kaleido, which can both be installed with pip and anaconda.

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
