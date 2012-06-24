==============
Introduction
==============

Machine Learning in NeuroImaging: what and why
===============================================

Machine learning is interested in learning from data empirical rules to
make **predictions**. Two kind of problems appear:

:Supervised learning:

    Supervised learning is interesting in predicting an *output
    variable*, or *target* from data. It maybe be a regression problem
    (predicting a continuous quantity) or a classification problem
    (predicting to which class each observations belongs too).

    In neuroimaging, supervised learning is typically used to relate
    brain images to behavioral or clinical observations.

:Unsupervised learning:

    Unsupervised learning studies the structure of a dataset, for
    instance extracting latent factors, or clusters.

    In neuroimaging, it is typically used to study resting state, or to
    find sub-populations in diseases.

Python and the scikit-learn: a primer
=====================================

.. topic:: What is the scikit-learn?

    `The scikit-learn <http://scikit-learn.org>`_ is a Python library for machine
    learning. Its strong points are:

    - Easy to use and well documented
    - Computationally efficient
    - Provide wide variety standard machine learning methods for non-experts

Installation of the materials useful for this tutorial
--------------------------------------------------------

Installing scientific Python
..............................

The scientific Python tool stack is rich. Installing the different
packages needed one after the other takes a lot of time and is not
recommended. We recommend that you install a complete distribution:

:Windows:
  EPD_ or `PythonXY <http://code.google.com/p/pythonxy/>`_: both of these
  distributions come with the scikit-learn installed

:MacOSX:
  EPD_ is the only full scientific Python distribution for Mac

:Linux:
  While EPD_ is available for Linux, most recent linux distributions come
  with the package that are needed for this tutorial. Ask your system
  administrator to install, using the distribution package manager, the
  following packages:
    - scikit-learn (sometimes called `sklearn`)
    - matplotlib
    - ipython

.. _EPD: http://www.enthought.com/products/epd.php


Nibabel
.......

`Nibabel <http://nipy.sourceforge.net/nibabel/>`_ is an easy to use
reader of NeuroImaging data files. It is not included in scientific
Python distributions but is required for all the parts of the tutorial.
You can install it with the following command::

  $ easy_install -U --user nibabel

Scikit-learn
...............

If scikit-learn is not installed on your computer, and you have a
working install of scientific Python packages (numpy, scipy) and a
C compiler, you can add it to your scientific Python install using::

  $ easy_install -U --user scikit-learn


Finding help
-------------

:Reference material:

    * A quick and gentle introduction to scientific computing with Python can
      be found in the 
      `scipy lecture notes <http://scipy-lectures.github.com/>`_.

    * The documentation of the scikit-learn explains each method with tips on
      practical use and examples: 
      `http://scikit-learn.org/ <http://scikit-learn.org/>`_
      While not specific to neuroimaging, it is often a recommended read.
      Be careful to consult the documentation relative to the version of
      the scikit-learn that you are using.

:Mailing lists:

    * You can find help with neuroimaging in Python (file I/O,
      neuroimaging-specific questions) on the nipy user group:
      https://groups.google.com/forum/?fromgroups#!forum/nipy-user

    * For machine-learning and scikit-learn question, expertise can be
      found on the scikit-learn mailing list:
      https://lists.sourceforge.net/lists/listinfo/scikit-learn-general
