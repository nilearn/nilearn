============
Introduction
============

Machine Learning in NeuroImaging: what and why
===============================================

Machine learning problems and vocabulary
-----------------------------------------

Machine learning is interested in learning empirical rules from data to
make **predictions**. Two kinds of problems appear:

:Supervised learning:

    :ref:`supervised_learning` is interested in predicting an **output
    variable**, or **target**, `y`, from **data** `X`. Typically, we start
    from labeled data (the **training set**) for which we know the `y`
    for each instance of `X` and train a model; this model is then
    applied to new unlabeled data (the **test set**) to predict the
    labels. It may be:
    
    * a **regression** problem: predicting a continuous quantity such 
      as age
    
    * a **classification** problem: predicting the class each 
      observation belongs to, such as patient or control

    In neuroimaging, supervised learning is typically used to relate
    brain images to behavioral or clinical observations.

:Unsupervised learning:

    :ref:`unsupervised_learning` is concerned with data `X` without any
    labels. It analyzes the structure of a dataset, for instance
    **clustering** or extracting latent factors such as independent
    components.

    In neuroimaging, it is typically used to study resting state, or to
    find sub-populations in diseases.


Why is machine learning relevant to NeuroImaging: a few examples
-----------------------------------------------------------------

:Diagnosis and prognosis:

    Predicting a clinical score from brain imaging with :ref:`supervised
    learning <Supervised_learning>` e.g. `[Mourao-Miranda 2012]
    <http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0029482>`_

:Generalization scores:

    * Information mapping: using the prediction accuracy of a classifier
      to test links between brain images and stimuli. (e.g.
      :ref:`searchlight <searchlight>`) `[Kriegeskorte 2005]
      <http://www.pnas.org/content/103/10/3863.short>`_

    * Transfer learning: measuring how much an estimator trained on a
      task generalizes to another task (e.g. discriminating left from
      right eye movements also discriminates additions from subtractions
      `[Knops 2009]
      <http://www.sciencemag.org/content/324/5934/1583.short>`_)

:Statistical estimation:

    From a statistical point of view, machine learning implements
    statistical estimation of models with a large number of parameters.
    The tricks pulled in machine learning (e.g. regularization) can
    make this estimation possible with a small number of observations
    `[Varoquaux 2012] <http://icml.cc/discuss/2012/688.html>`_. This
    usage of machine learning requires some understanding of the models.

:Data mining:

    Data-driven exploration of brain images. :ref:`unsupervised_learning`
    extracts structure from the data, such as `clusters
    <http://scikit-learn.org/stable/modules/clustering.html>`_ or
    `multivariate decompositions
    <http://scikit-learn.org/stable/modules/decomposition.html>`_
    (*latent factors* such as ICA). This may be useful for implementing
    some form of *density estimation*: learning a probabilistic model of
    the data (e.g. in `[Thirion 2009]
    <http://www.springerlink.com/content/7377x70p5515v778/>`_).

.. _installation:

Installation
=============

.. warning::

   nilearn is an un-released package. It lacks functionality and
   documentation, may have bugs, and will change in the future.
   Nonetheless, we strive to ensure its quality and hope that it may be
   useful in its current state, for technically-savvy users.

Installing the environment
---------------------------

Python scientific stack
........................

We recommend that you **install a complete scientific Python
distribution**, and not download the bare Python. Indeed, the scientific
Python tool stack is rich. Installing the different packages needed one
after is time-comsuming and error prone. error prone.

:Windows and MacOSX:
  We suggest that you install Anaconda_. [1]

:Linux:
  While Anaconda_ is available for Linux, most recent linux
  distributions come with the packages that are needed for this tutorial.
  Ask your system administrator to install, using the distribution
  package manager, the following packages:

    - scikit-learn (sometimes called `sklearn`, or `python-sklearn`)
    - matplotlib (sometimes called `python-matplotlib`)
    - ipython
    - nibabel (sometimes called `python-nibabel`)

.. _EPD: http://www.enthought.com/products/epd.php

.. _Anaconda: https://store.continuum.io/cshop/anaconda/

|

.. [1]

  EPD_,  `PythonXY <http://code.google.com/p/pythonxy/>`_ are other
  options, do make sure to install the full, non-free, EPD and not
  EPD-free.

Nibabel: reading neuroimaging files
....................................

`Nibabel <http://nipy.sourceforge.net/nibabel/>`_ is an easy to use
reader of NeuroImaging data files. It is not included in scientific
Python distributions, but is required for all parts of this tutorial.
You can install it with the following command::

  $ pip install -U --user nibabel

If you are under linux, your package manager might have nibabel, for
instance called `python-nibabel`.

.. warning::

   Note that this is a "shell" command, that you need to type in a
   command prompt, and not a Python command.

Installing nilearn
-------------------

Downloading the development version
....................................

Nilearn is unreleased. We advise that you track it's development using
git.
  
* **Under Windows or Max OSX**, you can easily to that by going to
  https://github.com/nilearn/nilearn and clicking the 'Clone in Desktop'
  button on the lower right of the page. This will install a software
  that will download nilearn and that you can use to update nilearn as we
  move forward.
  
* **Under Linux**, run the following command (as a shell command, not a
  Python command)::

    $ git clone https://github.com/nilearn/nilearn.git

As time goes, you can update you copy of nilearn by doing "git pull" in
this directory.

If you really don't want to use git, you download the latest development
snapshot from the following link and unziping it:
https://github.com/nilearn/nilearn/archive/master.zip


Setting up
..........

In the ``nilearn`` directory created by the previous steps, run (as a
shell command, once again)::

    $ python setup.py install --user    

To test if you have done everything right, open IPython and try the
following, in the Python prompt::

    In [1]: import nilearn

If you do not get any errors, you have installed nilearn correctly.

Python for NeuroImaging: a quick-start
==========================================

If you don't know Python, **Don't panic. Python is easy**. Here, we give
the basics to help you get started. For a full blown introduction to
using Python for science, see the `scipy lecture notes
<http://scipy-lectures.github.io/>`_.


We will be using `IPython <http://ipython.org>`_, in pylab mode, which
provides an interactive scientific environment. Start it with::

    $ ipython --pylab

(depending on your ipython version, you may need to use the ``-pylab``
flag instead).

It's interactive::

    Welcome to pylab, a matplotlib-based Python environment
    For more information, type 'help(pylab)'.

    In [1]: 1 + 2*3
    Out[1]: 7

.. topic:: **`>>>` Prompt**
   
   Below we'll be using `>>>` to indicate input lines. If you wish to copy
   these input lines directly into *IPython*, click on the `>>>` located
   at the top right of the code block

Scientific computing with Python
---------------------------------

Basic numerics
...............

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

  `More documentation ...
  <http://scipy-lectures.github.com/intro/numpy/index.html>`__

:Plotting:

 .. figure:: auto_examples/images/plot_python_101_1.png
   :target: auto_examples/plot_python_101.html
   :align: right
   :scale: 30

 :: 

    >>> import matplotlib.pyplot as plt
    >>> pl.plot(t, np.cos(t))       # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]


 `More documentation ...
 <http://scipy-lectures.github.com/intro/matplotlib/matplotlib.html>`__

:Image processing:

 :: 

    >>> from scipy import ndimage
    >>> t_smooth = ndimage.gaussian_filter(t, sigma=2)

 `More documentation ...
 <http://scipy-lectures.github.com/advanced/image_processing/index.html>`__

:Signal processing:

    >>> from scipy import signal
    >>> t_detrended = signal.detrend(t)

 `More documentation ...
 <http://scipy-lectures.github.com/intro/scipy.html#signal-processing-scipy-signal>`__

:Much more:

  .. hlist::

     * Simple statistics::

        >>> from scipy import stats

     * Linear algebra::

        >>> from scipy import linalg

  `More documentation...
  <http://scipy-lectures.github.com/intro/scipy.html>`__


Scikit-learn: machine learning in Python
.........................................

.. topic:: What is scikit-learn?

    `Scikit-learn <http://scikit-learn.org>`_ is a Python library for machine
    learning. Its strong points are:

    - Easy to use and well documented
    - Computationally efficient
    - Provides a wide variety of standard machine learning methods for non-experts

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
use a hand-written digits dataset, which comes with scikit-learn::

    >>> from sklearn import datasets
    >>> digits = datasets.load_digits()
    >>> data = digits.data
    >>> labels = digits.target

Let's use all but the last 10 samples to train the SVC::

    >>> svc.fit(data[:-10], labels[:-10])   # doctest: +ELLIPSIS
    SVC(C=1.0, ...)

and try predicting the labels on the left-out data::

    >>> svc.predict(data[-10:])
    array([5, 4, 8, 8, 4, 9, 0, 8, 9, 8])
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

    * The documentation of scikit-learn explains each method with tips on
      practical use and examples: 
      `http://scikit-learn.org/ <http://scikit-learn.org/>`_
      While not specific to neuroimaging, it is often a recommended read.
      Be careful to consult the documentation relative to the version of
      scikit-learn that you are using.

:Mailing lists:

    * You can find help with neuroimaging in Python (file I/O,
      neuroimaging-specific questions) via the nipy user group:
      https://groups.google.com/forum/?fromgroups#!forum/nipy-user

    * For machine-learning and scikit-learn questions, expertise can be
      found on the scikit-learn mailing list:
      https://lists.sourceforge.net/lists/listinfo/scikit-learn-general
