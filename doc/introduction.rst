.. for doc tests to run with recent NumPy 1.14, we need to set print options
   to older versions. See issue #1593 for more details
    >>> import numpy as np
    >>> from distutils.version import LooseVersion
    >>> if LooseVersion(np.__version__) >= LooseVersion('1.14'):
    ...     np.set_printoptions(legacy='1.13')

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
    **pattern recognition** and **multivariate statistical** techniques on
    neuroimaging data for applications such as **MVPA** (Mutli-Voxel
    Pattern Analysis),
    :ref:`decoding <decoding>`,
    :ref:`predictive modelling <decoding>`,
    :ref:`functional connectivity <functional_connectomes>`,
    :ref:`brain parcellations <parcellating_brain>`,
    :ref:`connectomes <functional_connectomes>`.

    Nilearn can readily be used on :ref:`task fMRI <decoding_intro>`,
    :ref:`resting-state <functional_connectomes>`, or
    :ref:`VBM <sphx_glr_auto_examples_02_decoding_plot_oasis_vbm.py>` data.

    For a machine-learning expert, the value of nilearn can be seen as
    domain-specific **feature engineering** construction, that is, shaping
    neuroimaging data into a feature matrix well suited to statistical
    learning, or vice versa.


Why is machine learning relevant to NeuroImaging? A few examples!
-----------------------------------------------------------------

:Diagnosis and prognosis:

    Predicting a clinical score or even treatment response
    from brain imaging with :ref:`supervised
    learning <decoding>` e.g. `[Mourao-Miranda 2012]
    <http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0029482>`_

:Measuring generalization scores:

    * **Information mapping**: using the prediction accuracy of a classifier
      to characterize relationships between brain images and stimuli. (e.g.
      :ref:`searchlight <searchlight>`) `[Kriegeskorte 2005]
      <http://www.pnas.org/content/103/10/3863.short>`_

    * **Transfer learning**: measuring how much an estimator trained on one
      specific psychological process/task can predict the neural activity
      underlying another specific psychological process/task
      (e.g. discriminating left from
      right eye movements also discriminates additions from subtractions
      `[Knops 2009]
      <http://www.sciencemag.org/content/324/5934/1583.short>`_)

:High-dimensional multivariate statistics:

    From a statistical point of view, machine learning implements
    statistical estimation of models with a large number of parameters.
    Tricks pulled in machine learning (e.g. regularization) can
    make this estimation possible despite the usually
    small number of observations in the neuroimaging domain
    `[Varoquaux 2012] <http://icml.cc/2012/papers/688.pdf>`_. This
    usage of machine learning requires some understanding of the models.

:Data mining / exploration:

    Data-driven exploration of brain images. This includes the extraction of
    the major brain networks from resting-state data ("resting-state networks")
    as well as the discovery of connectionally coherent functional modules
    ("connectivity-based parcellation").
    For example,
    :ref:`extracting_rsn` or :ref:`parcellating_brain` with clustering.

Glossary: machine learning vocabulary
--------------------------------------

:Supervised learning:

    :ref:`Supervised learning <decoding>` is interested in predicting an
    **output variable**, or **target**, `y`, from **data** `X`.
    Typically, we start from labeled data (the **training set**). We need to
    know the `y` for each instance of `X` in order to train the model. Once
    learned, this model is then applied to new unlabeled data (the **test set**)
    to predict the labels (although we actually know them). There are
    essentially two possible goals:

    * a **regression** problem: predicting a continuous variable, such
      as participant age, from the data `X`

    * a **classification** problem: predicting a binary variable that splits
      the observations into two groups, such as patients versus controls

    In neuroimaging research, supervised learning is typically used to
    derive an underlying cognitive process (e.g., emotional versus non-emotional
    theory of mind), a behavioral variable (e.g., reaction time or IQ), or
    diagnosis status (e.g., schizophrenia versus healthy) from brain images.

:Unsupervised learning:

    `Unsupervised learning
    <http://scikit-learn.org/stable/unsupervised_learning.html>`_ is
    concerned with data `X` without any labels. It analyzes the structure
    of a dataset to find coherent underlying structure,
    for instance using **clustering**, or to extract latent
    factors, for instance using **independent components analysis (ICA)**.

    In neuroimaging research, it is typically used to create functional and
    anatomical brain atlases by clustering based on connectivity or to
    extract the main brain networks from resting-state correlations. An
    important option of future research will be the identification of
    potential neurobiological subgroups in psychiatric and neurobiological
    disorders.

|

.. _installation:

Installing nilearn
====================

.. raw:: html
   :file: install_doc_component.html

.. _quick_start:

Python for NeuroImaging, a quick start
==========================================

If you don't know Python, **Don't panic. Python is easy**. It is important
to realize that most things you will do in nilearn require only a few or a
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


Your first steps with nilearn
------------------------------

First things first, nilearn does not have a graphical user interface.
But you will soon realize that you don't really need one.
It is typically used interactively in IPython or in an automated way by Python
code.
Most importantly, nilearn functions that process neuroimaging data accept
either a filename (i.e., a string variable) or a `NiftiImage object
<http://nipy.org/nibabel/nibabel_images.html>`_. We call the latter
"niimg-like".

Suppose for instance that you have a Tmap image saved in the Nifti file
"t_map000.nii" in the directory "/home/user". To visualize that image, you will
first have to import the :ref:`plotting <plotting>` functionality by::

    >>> from nilearn import plotting

Then you can call the function that creates a "glass brain" by giving it
the file name::

    >>> plotting.plot_glass_brain("/home/user/t_map000.nii")   # doctest: +SKIP

.. sidebar:: File name matchings

   The filename could be given as "~/t_map000.nii' as nilearn expands "~" to
   the home directory.
   :ref:`See more on file name matchings <filename_matching>`.


.. image:: auto_examples/01_plotting/images/sphx_glr_plot_demo_glass_brain_001.png
    :target: auto_examples/01_plotting/plot_demo_glass_brain.html
    :align: center
    :scale: 60

.. note::

   There are many other plotting functions. Take your time to have a look
   at the :ref:`different options <plotting>`.

|

.. currentmodule:: nilearn

For simple functions/operations on images, many functions exist, such as in
the :mod:`nilearn.image` module for image manipulation, e.g.
:func:`image.smooth_img` for smoothing::

    >>> from nilearn import image
    >>> smoothed_img = image.smooth_img("/home/user/t_map000.nii", fwhm=5)   # doctest: +SKIP

The returned value `smoothed_img` is a `NiftiImage object
<http://nipy.org/nibabel/nibabel_images.html>`_. It can either be passed
to other nilearn functions operating on niimgs (neuroimaging images) or
saved to disk with::

    >>> smoothed_img.to_filename("/home/user/t_map000_smoothed.nii")   # doctest: +SKIP

Finally, nilearn deals with Nifti images that come in two flavors: 3D
images, which represent a brain volume, and 4D images, which represent a
series of brain volumes. To extract the n-th 3D image from a 4D image, you can
use the :func:`image.index_img` function (keep in mind that array indexing
always starts at 0 in the Python language)::

    >>> first_volume = image.index_img("/home/user/fmri_volumes.nii", 0)   # doctest: +SKIP

To loop over each individual volume of a 4D image, use :func:`image.iter_img`::

   >>> for volume in image.iter_img("/home/user/fmri_volumes.nii"):   # doctest: +SKIP
   ...     smoothed_img = image.smooth_img(volume, fwhm=5)

.. topic:: **Exercise: varying the amount of smoothing**
   :class: green

   Want to sharpen your skills with nilearn?
   Compute the mean EPI for first subject of the ADHD
   dataset downloaded with :func:`nilearn.datasets.fetch_adhd` and
   smooth it with an FWHM varying from 0mm to 20mm in increments of 5mm

   **Hints:**

      * Inspect the '.keys()' of the object returned by
        :func:`nilearn.datasets.fetch_adhd`

      * Look at the "reference" section of the documentation: there is a
        function to compute the mean of a 4D image

      * To perform a for loop in Python, you can use the "range" function

      * The solution can be found :ref:`here
        <sphx_glr_auto_examples_04_manipulating_images_plot_smooth_mean_image.py>`

|


.. topic:: **Warm up examples**

   The two following examples may be useful to get familiar with data
   representation in nilearn:

   * :ref:`sphx_glr_auto_examples_plot_nilearn_101.py`

   * :ref:`sphx_glr_auto_examples_plot_3d_and_4d_niimg.py`

____

Now, if you want out-of-the-box methods to process neuroimaging data, jump
directly to the section you need:

* :ref:`decoding`

* :ref:`functional_connectivity`

|

Scientific computing with Python
---------------------------------

In case you plan to become a casual nilearn user, note that you will not need
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


Scikit-learn: machine learning in Python
.........................................

.. topic:: **What is scikit-learn?**

    `Scikit-learn <http://scikit-learn.org>`_ is a Python library for machine
    learning. Its strong points are:

    - Easy to use and well documented
    - Computationally efficient
    - Provides a wide variety of standard machine learning methods for non-experts

The core concept in `scikit-learn <http://scikit-learn.org>`_ is the
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

Once the object is created, you can fit it on data. For instance, here we
use a hand-written digits dataset, which comes with scikit-learn::

    >>> from sklearn import datasets
    >>> digits = datasets.load_digits()
    >>> data = digits.data
    >>> labels = digits.target

Let's use all but the last 10 samples to train the SVC::

    >>> svc.fit(data[:-10], labels[:-10])   # doctest: +ELLIPSIS
    SVC(C=1.0, ...)

and try predicting the labels on the left-out data::

    >>> svc.predict(data[-10:])     # doctest: +SKIP
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
      Be careful to consult the documentation of the scikit-learn version
      that you are using.

:Mailing lists and forums:

    * Don't hesitate to ask questions about nilearn on `neurostars
      <https://neurostars.org/t/nilearn/>`_.

    * You can find help with neuroimaging in Python (file I/O,
      neuroimaging-specific questions) via the nipy user group:
      https://groups.google.com/forum/?fromgroups#!forum/nipy-user

    * For machine-learning and scikit-learn questions, expertise can be
      found on the scikit-learn mailing list:
      https://mail.python.org/mailman/listinfo/scikit-learn
