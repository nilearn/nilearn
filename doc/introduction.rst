=====================================
Introduction: nistats in a nutshell
=====================================

.. contents:: **Contents**
    :local:
    :depth: 1


What is nistats: Bold-fMRI analysis
===========================================================================

.. topic:: **Why use nistats?**

    Because it is awesome.

.. _installation:

Installing nistats
====================

.. raw:: html
   :file: install_doc_component.html

.. _quick_start:

Python for NeuroImaging, a quick start
==========================================

If you don't know Python, **Don't panic. Python is easy**. It is important
to realize that most things you will do in nistats require only a few or a
few dozen lines of Python code.
Here, we give
the basics to help you get started. For a very quick start into the programming
language, you can `learn it online <http://www.learnpython.org/>`_.
For a full-blown introduction to
using Python for science, see the `scipy lecture notes
<http://scipy-lectures.github.io/>`_.


We will be using `IPython <http://ipython.org>`_, which provides an
interactive scientific environment that facilitates (e.g.,
interactive debugging) and improves (e.g., printing of large matrices)
many everyday data-manipulation steps. Start it by typing::

    ipython --matplotlib

This will open an interactive prompt::

    IPython ?.?.? -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    In [1]: 1 + 2 * 3
    Out[1]: 7

.. note::

   The ``--matplotlib`` flag, which configures matplotlib for
   interactive use inside IPython, is available for IPython versions
   from 1.0 onwards. If you are using versions older than this,
   e.g. 0.13, you can use the ``--pylab`` flag instead.

.. topic:: `>>>` **Prompt**

   Below we'll be using `>>>` to indicate input lines. If you wish to copy and
   paste these input lines directly into *IPython*, click on the `>>>` located
   at the top right of the code block to toggle these prompt signs


Your first steps with nistats
------------------------------

First things first, nistats does not have a graphical user interface.
But you will soon realize that you don't really need one.
It is typically used interactively in IPython or in an automated way by Python
code.
Most importantly, nistats functions that process neuroimaging data accept
either a filename (i.e., a string variable) or a `NiftiImage object
<http://nipy.org/nibabel/nibabel_images.html>`_. We call the latter
"niimg-like".


Scientific computing with Python
---------------------------------

In case you plan to become a casual nistats user, note that you will not need
to deal with number and array manipulation directly in Python.
However, if you plan to go beyond that, here are a few pointers.

Basic numerics
...............

:Numerical arrays:

  The numerical data (e.g. matrices) are stored in numpy arrays:

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

  `More documentation ...
  <http://scipy-lectures.github.io/intro/numpy/index.html>`__

:Plotting and figures:

 .. figure:: auto_examples/images/sphx_glr_plot_python_101_001.png
   :target: auto_examples/plot_python_101.html
   :align: right
   :scale: 30

 ::

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, np.cos(t))       # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]


 `More documentation ...
 <http://scipy-lectures.github.io/intro/matplotlib/matplotlib.html>`__

:Image processing:

 ::

    >>> from scipy import ndimage
    >>> t_smooth = ndimage.gaussian_filter(t, sigma=2)

 `More documentation ...
 <http://scipy-lectures.github.io/advanced/image_processing/index.html>`__

:Signal processing:

    >>> from scipy import signal
    >>> t_detrended = signal.detrend(t)

 `More documentation ...
 <http://scipy-lectures.github.io/intro/scipy.html#signal-processing-scipy-signal>`__

:Much more:

  .. hlist::

     * Simple statistics::

        >>> from scipy import stats

     * Linear algebra::

        >>> from scipy import linalg

  `More documentation...
  <http://scipy-lectures.github.io/intro/scipy.html>`__


Finding help
-------------

:Reference material:

    * A quick and gentle introduction to scientific computing with Python can
      be found in the
      `scipy lecture notes <http://scipy-lectures.github.io/>`_.

    * The documentation of scikit-learn explains each method with tips on
      practical use and examples:
      `http://scikit-learn.org/ <http://scikit-learn.org/>`_.
      While not specific to neuroimaging, it is often a recommended read.
      Be careful to consult the documentation of the scikit-learn version
      that you are using.

:Mailing lists and forums:

    * Don't hesitate to ask questions about nistats on `neurostars
      <https://neurostars.org/t/nistats/>`_.

    * You can find help with neuroimaging in Python (file I/O,
      neuroimaging-specific questions) via the nipy user group:
      https://groups.google.com/forum/?fromgroups#!forum/nipy-user

