=====================================
Introduction: nilearn in a nutshell
=====================================

.. contents:: **Contents**
    :local:
    :depth: 1


What is nilearn: MVPA, decoding, predictive models, functional connectivity
===========================================================================

.. topic:: **Why use nilearn?**

    Nilearn makes it easy to use many advanced **machine learning**,
    **pattern recognition** and **multivariate statistical** technics on
    neuroimaging data for applications such as **MVPA** (Mutli-Voxel
    Pattern Analysis),
    :ref:`decoding <decoding>`,
    :ref:`predictive modelling <decoding>`,
    :ref:`functional connectivity <functional_connectomes>`,
    :ref:`brain parcellations <parcellating_brain>`, 
    :ref:`connectomes <functional_connectomes>`.

    Nilearn can readily be used on :ref:`task fMRI <decoding_tutorial>`,
    :ref:`resting-state <functional_connectomes>`, or 
    :ref:`VBM <example_decoding_plot_oasis_vbm.py>` data.

    For a machine-learning expert, the value of nilearn can be seen as
    domain-specific **feature engineering** construction, to go from
    neuroimaging data to a feature-matrix well suited to statistical
    learning, or vice versa.


Why is machine learning relevant to NeuroImaging: a few examples
-----------------------------------------------------------------

:Diagnosis and prognosis:

    Predicting a clinical score from brain imaging with :ref:`supervised
    learning <decoding>` e.g. `[Mourao-Miranda 2012]
    <http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0029482>`_

:Measuring generalization scores:

    * **Information mapping**: using the prediction accuracy of a classifier
      to test links between brain images and stimuli. (e.g.
      :ref:`searchlight <searchlight>`) `[Kriegeskorte 2005]
      <http://www.pnas.org/content/103/10/3863.short>`_

    * **Transfer learning**: measuring how much an estimator trained on a
      task generalizes to another task (e.g. discriminating left from
      right eye movements also discriminates additions from subtractions
      `[Knops 2009]
      <http://www.sciencemag.org/content/324/5934/1583.short>`_)

:High-dimensional multivariate statistics:

    From a statistical point of view, machine learning implements
    statistical estimation of models with a large number of parameters.
    Tricks pulled in machine learning (e.g. regularization) can
    make this estimation possible with a small number of observations
    `[Varoquaux 2012] <http://icml.cc/discuss/2012/688.html>`_. This
    usage of machine learning requires some understanding of the models.

:Data mining / exploration:

    Data-driven exploration of brain images. For example,
    :ref:`extracting_rsn` or :ref:`parcellating_brain` with clustering.

Glossary: machine learning vocabulary
--------------------------------------

:Supervised learning:

    :ref:`Supervised learning <decoding>` is interested in predicting an
    **output variable**, or **target**, `y`, from **data** `X`.
    Typically, we start from labeled data (the **training set**) for
    which we know the `y` for each instance of `X` and train a model;
    this model is then applied to new unlabeled data (the **test set**)
    to predict the labels. It may be:
    
    * a **regression** problem: predicting a continuous quantity such 
      as age
    
    * a **classification** problem: predicting the class each 
      observation belongs to, such as patient or control

    In neuroimaging, supervised learning is typically used to relate
    brain images to behavioral or clinical observations.

:Unsupervised learning:

    `Unsupervised learning
    <http://scikit-learn.org/stable/unsupervised_learning.html>`_ is
    concerned with data `X` without any labels. It analyzes the structure
    of a dataset, for instance **clustering** or extracting latent
    factors such as with **independent components analysis (ICA)**.

    In neuroimaging, it is typically used to study resting state, or to
    find sub-populations in diseases.

|

.. _installation:

Installing nilearn
====================

Installing the Python scientific environment
----------------------------------------------

We recommend that you **install a complete scientific Python distribution**,
and not download the bare Python. Indeed, the scientific Python tool stack is
rich. Installing the different packages needed is time-consuming and error
prone.

:Windows and MacOSX:
  We suggest you to install 64 bit Anaconda_.

  `Enthought Canopy`_, `PythonXY <http://code.google.com/p/pythonxy/>`_ are
  other options. Enthought Canopy Express, the free version, should cover all
  the required packages.


:Linux:
  While Anaconda_ is available for Linux, most recent linux
  distributions come with the packages that are needed for nilearn.
  Ask your system administrator to install, using the distribution
  package manager, the following packages:

    - scikit-learn (sometimes called `sklearn`, or `python-sklearn`)
    - matplotlib (sometimes called `python-matplotlib`)
    - ipython
    - nibabel (sometimes called `python-nibabel`)

.. _Enthought Canopy: https://store.enthought.com/

.. _Anaconda: https://store.continuum.io/cshop/anaconda/

Installing nilearn
-------------------

The simplest way to install nilearn is to run the following command in
a command prompt::

    pip install -U --user nilearn

.. warning::

   Note that this is a "shell" command, that you need to type in a
   command prompt, and not a Python command.

|

.. _testing_installation:

Testing your installation
.........................

To test if you have done everything right, open IPython and try the
following, in the Python prompt::

    In [1]: import nilearn

If you do not get any errors, you have installed nilearn correctly.

Installing the development version
....................................

**Downloading** As an alternative to using pip, and only in the case if you
want the latest nilearn version you can do so by using git.

* **Under Windows or Max OSX**, you can easily to that by going to
  https://github.com/nilearn/nilearn and clicking the 'Clone in Desktop'
  button on the lower right of the page. This will install a software
  that will download nilearn and that you can use to update nilearn.
  
* **Under Linux**, run the following command (as a shell command, not a
  Python command)::

    git clone https://github.com/nilearn/nilearn.git

As time goes, you can update your copy of nilearn by doing "git pull" in
this directory.

If you really don't want to use git, you download the latest development
snapshot from the following link and unziping it:
https://github.com/nilearn/nilearn/archive/master.zip

**Installing** In the ``nilearn`` directory created by the previous steps, run
(as a shell command, once again)::

    python setup.py install --user

To make sure that the installation went smoothly, you can follow the
same steps as in :ref:`testing_installation`.

.. _quick_start:

Python for NeuroImaging, a quick start
==========================================

If you don't know Python, **Don't panic. Python is easy**. Here, we give
the basics to help you get started. For a very quick start to the
language, you can `learn it online <http://www.learnpython.org/>`_.
For a full blown introduction to
using Python for science, see the `scipy lecture notes
<http://scipy-lectures.github.io/>`_.


We will be using `IPython <http://ipython.org>`_, which provides an
interactive scientific environment. Start it with::

    ipython --matplotlib

which will open an interactive prompt::

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

   Below we'll be using `>>>` to indicate input lines. If you wish to copy
   these input lines directly into *IPython*, click on the `>>>` located
   at the top right of the code block


Your first steps with nilearn
------------------------------

nilearn does not have a graphical user interface, it is used via Python
code. Most important function to process neuroimaging data take as an
input either a filename, or a `NiftiImage object
<http://nipy.org/nibabel/nibabel_images.html>`_, which we call a
"niimg-like".

For instance, suppose that you have a Tmap file saved in a Nifti file
"t_map000.nii", in the directory "/home/user", to visualize it, you will
first import the :ref:`plotting <plotting>` functionality::

    from nilearn import plotting

then you can call the function that creates a "glass brain" by giving it
the file name::

    plotting.plot_glass_brain("/home/user/t_map000.nii")

.. image:: auto_examples/manipulating_visualizing/images/plot_demo_glass_brain_001.png
    :target: auto_examples/manipulating_visualizing/plot_demo_glass_brain.html
    :align: center
    :scale: 60

.. currentmodule:: nilearn

For simple operations on images, there exists many functions, such as in
the :mod:`nilearn.image` module for image manipulation, eg
:func:`image.smooth_img` for smoothing::

    >>> from nilearn import image
    >>> smoothed_img = image.smooth_img("/home/user/t_map000.nii", fwhm=5)

The returned value, `smoothed_img` is a `NiftiImage object
<http://nipy.org/nibabel/nibabel_images.html>`_, and can be either passed
to other nilearn functions operating on niimgs (neuroimaging images), or
saved to disk with::

    >>> smoothed_img.to_filename("/home/user/t_map000_smoothed.nii")

Finally, nilearn deals with Nifti images that come in two flavors: 3D
images, that represent a brain volume, and 4D images, that represent a
series of brain volume. To extract the n-th 3D image in a 4D image, use
:func:`image.index_img` (keep in mind that indexing starts at 0 in
Python)::

    >>> first_volume = image.index_img("/home/user/fmri_volumes.nii", 0)

To do a loop over each volume of a 4D image, use :func:`image.iter_img`::

   >>> for volume in image.iter_img("/home/user/fmri_volumes.nii"):
   ...     smoothed_img = image.smooth_img(volume, fwhm=5)

.. topic:: **Exercise: varying the amount of smoothing**
   :class: green

   Want to sharpen your skills with nilearn? 
   Compute the mean EPI for first subject of the ADHD
   dataset downloaded with :func:`nilearn.datasets.fetch_adhd`, and
   smooth it with an FWHM varying from 0mm to 20mm in increments of 5mm

   **Hints:**

      * Inspect the '.keys()' of the object returned by
        :func:`nilearn.datasets.fetch_adhd`

      * Look at the "reference" section of the documentation: there is a
        function to compute the mean of a 4D image

      * You can do a for loop in Python. You can use the "range" function

      * The solution is found :ref:`here
        <example_manipulating_visualizing_plot_smooth_mean_image.py>`

|

____

Now, if you want readily-made methods to process neuroimaging data, jump
directly to the section you need:

* :ref:`decoding`

* :ref:`functional_connectivity`

|

Scientific computing with Python
---------------------------------

If you are interesting in a casual use of nilearn, you will not need
numerics in Python. To go further, here are a few pointers.

Basic numerics
...............

:Numerical arrays:

  The numerical data (eg matrices) are stored in numpy arrays:

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

 .. figure:: auto_examples/images/plot_python_101_001.png
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


Scikit-learn: machine learning in Python
.........................................

.. topic:: **What is scikit-learn?**

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
      `scipy lecture notes <http://scipy-lectures.github.io/>`_.

    * The documentation of scikit-learn explains each method with tips on
      practical use and examples: 
      `http://scikit-learn.org/ <http://scikit-learn.org/>`_.
      While not specific to neuroimaging, it is often a recommended read.
      Be careful to consult the documentation relative to the version of
      scikit-learn that you are using.

:Mailing lists and forums:

    * Don't hesitate to ask questions about nilearn on `neurostars
      <https://neurostars.org/t/nilearn/>`_.

    * You can find help with neuroimaging in Python (file I/O,
      neuroimaging-specific questions) via the nipy user group:
      https://groups.google.com/forum/?fromgroups#!forum/nipy-user

    * For machine-learning and scikit-learn questions, expertise can be
      found on the scikit-learn mailing list:
      https://lists.sourceforge.net/lists/listinfo/scikit-learn-general
