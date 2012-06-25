==============
Introduction
==============

Machine Learning in NeuroImaging: what and why
===============================================

Machine learning is interested in learning from data empirical rules to
make **predictions**. Two kind of problems appear:

:Supervised learning:

    Supervised learning is interesting in predicting an *output
    variable*, or *target* from data. It maybe be a **regression** problem
    (predicting a continuous quantity) or a **classification** problem
    (predicting to which class each observations belongs too).

    In neuroimaging, supervised learning is typically used to relate
    brain images to behavioral or clinical observations.

:Unsupervised learning:

    Unsupervised learning studies the structure of a dataset, for
    instance extracting latent factors, or **clustering**.

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

.. XXX need to write the 10min intro to Python

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

Python for Science quickstart
------------------------------

**Don't panic. Python is easy.**
For a full blown introduction to using Python for science, see the 
`scipy lecture notes <http://scipy-lectures.github.com/>`_.


We will be using `IPython <http://ipython.org>`_, in pylab mode, that
provides an interactive scientific environment. Start it with::

    $ ipython -pylab

It's interactive::

    Welcome to pylab, a matplotlib-based Python environment
    For more information, type 'help(pylab)'.

    In [1]: 1 + 2*3
    Out[1]: 7

.. note:: **Prompt**: Below we'll be using `>>>` to indicate input lines
   If you wish to copy these input lines directly into your *IPython*
   console without manually excluding each `>>>`, you can enable
   `Doctest Mode` with the command ::
   	    %doctest_mode

Scientific computing
.....................

In Python, to get scientific features, you need to import the relevant
libraries:

:Numerical arrays:

  ::

    >>> import numpy as np
    >>> t = np.linspace(1, 10, 2000)  # 2000 points between 1 and 10
    >>> t
    array([  1.        ,   1.00450225,   1.0090045 , ...,   9.9909955 ,
             9.99549775,  10.        ])
    >>> t / 2
    array([ 0.5       ,  0.50225113,  0.50450225, ...,  4.99549775,
            4.99774887,  5.        ])
    >>> np.cos(t) # Operations on arrays are defined in the numpy module
    array([ 0.54030231,  0.53650833,  0.53270348, ..., -0.84393609,
           -0.84151234, -0.83907153])
    >>> t[:3] # In Python indexing is done with [] and starts at zero
    array([ 1.        ,  1.00450225,  1.0090045 ])

  `More documentation...
  <http://scipy-lectures.github.com/intro/numpy/index.html>`_

:Plotting:

 .. figure:: auto_examples/images/plot_python_101_1.png
   :target: auto_examples/plot_python_101.html
   :align: right
   :scale: 30

 :: 

    >>> import pylab as pl
    >>> pl.plot(t, np.cos(t))       # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]


 `More documentation...
 <http://scipy-lectures.github.com/intro/matplotlib/matplotlib.html>`_

:Image processing:

 :: 

    >>> from scipy import ndimage
    >>> t_smooth = ndimage.gaussian_filter(t, sigma=2)

 `More documentation...
 <http://scipy-lectures.github.com/advanced/image_processing/index.html>`_

:Signal processing:

    >>> from scipy import signal
    >>> t_detrended = signal.detrend(t)

 `More documentation...
 <http://scipy-lectures.github.com/intro/scipy.html#signal-processing-scipy-signal>`_

:Much more:

  .. hlist::

     * Simple statistics::

        >>> from scipy import stats

     * Linear algebra::

        >>> from scipy import linalg

  `More documentation...
  <http://scipy-lectures.github.com/intro/scipy.html>`_

Scikit-learn: machine learning
..............................

The core concept in the `scikit-learn <http://scikit-learn.org>`_ is the
estimator object, for instance an SVC (`support vector classifier
<http://scikit-learn.org/stable/modules/svm.html>`_).
It is first created with the relevant parameters::

    >>> from sklearn.svm import SVC
    >>> svc = SVC(kernel='linear', C=1.)

These parameters are detailed in the documentation of
the object: in IPython you can do::

    In [3]: SVC?
    ...
    Parameters
    ----------
    C : float or None, optional (default=None)
        Penalty parameter C of the error term. If None then C is set
        to n_samples.

    kernel : string, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'.
        If none is given, 'rbf' will be used.
    ...

Once the object is created, you can fit it on data, for instance here we
use a hand-written digits datasets, that comes with the scikit-learn::

    >>> from sklearn import datasets
    >>> digits = datasets.load_digits()
    >>> data = digits.data
    >>> labels = digits.target

Let's use all but the last 10 samples to train the SVC::

    >>> svc.fit(data[:-10], labels[:-10])   # doctest: +ELLIPSIS
    SVC(C=1.0, ...)

and try predicting the labels on the left-out data::

    >>> svc.predict(data[-10:])
    array([ 5.,  4.,  8.,  8.,  4.,  9.,  0.,  8.,  9.,  8.])
    >>> labels[-10:]    # The actual labels
    array([5, 4, 8, 8, 4, 9, 0, 8, 9, 8])

To find out more, try the `scikit-learn tutorials
<http://scikit-learn.org/stable/tutorial/index.html>`_.

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
