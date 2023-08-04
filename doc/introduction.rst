Introduction
============


What is ``nilearn``?
====================

``nilearn`` is a package that makes it easy to use advanced machine learning techniques to analyze data acquired with MRI machines.
In particular, underlying machine learning problems include
:ref:`decoding brain data <decoding>`,
computing :ref:`brain parcellations <parcellating_brain>`,
analyzing :ref:`functional connectivity <functional_connectomes>` and :ref:`connectomes <functional_connectomes>`,
doing multi-voxel pattern analysis (MVPA) or :ref:`predictive modelling <decoding>`.

``nilearn`` can readily be used on :ref:`task fMRI <decoding_intro>`,
:ref:`resting-state <functional_connectomes>`, or
:ref:`voxel-based morphometry (VBM) <sphx_glr_auto_examples_02_decoding_plot_oasis_vbm.py>` data.

For machine learning experts, the value of ``nilearn`` can be seen as
domain-specific **feature engineering** construction, that is, shaping
neuroimaging data into a feature matrix well suited for statistical learning.

.. note::

    It is ok if these terms don't make sense to you yet:
    this guide will walk you through them in a comprehensive manner.


.. _quick_start:


Using ``nilearn`` for the first time
====================================

``nilearn`` is a Python library. If you have never used Python before,
you should probably have a look at a `general introduction about Python <http://www.learnpython.org/>`_
as well as to an `introduction to using Python for science <http://scipy-lectures.github.io/>`_ before diving into ``nilearn``.

First steps with nilearn
------------------------

At this stage, you should have :ref:`installed <quickstart>` ``nilearn`` and opened a Jupyter notebook
or an IPython / Python session.  First, load ``nilearn`` with

.. code-block:: default

    import nilearn

``nilearn`` comes in with some data that are commonly used in neuroimaging.
For instance, it comes with volumic template images of brains such as MNI:

.. code-block:: default

    print(nilearn.datasets.MNI152_FILE_PATH)

Output:

.. code-block:: text
    :class: highlight-primary

    '/home/yasmin/nilearn/nilearn/nilearn/datasets/data/mni_icbm152_t1_tal_nlin_sym_09a_converted.nii.gz'

Let's have a look at this image:

.. code-block:: default

    nilearn.plotting.plot_img(nilearn.datasets.MNI152_FILE_PATH)

.. image:: auto_examples/01_plotting/images/sphx_glr_plot_demo_glass_brain_001.png
    :target: auto_examples/00_tutorials/images/sphx_glr_plot_nilearn_101_001.png
    :align: center
    :scale: 60

Learning with the API references
--------------------------------

In the last command, you just made use of 2 ``nilearn`` modules: :mod:`nilearn.datasets`
and :mod:`nilearn.plotting`.
All modules are described in the :ref:`API references <modules>`.

Oftentimes, if you are already familiar with the problems and vocabulary of MRI analysis,
the module and function names are explicit enough that you should understand what ``nilearn`` does.

.. note:: **Exercise: Varying the amount of smoothing in an image**

   Compute the mean :term:`EPI` for one individual of the brain development
   dataset downloaded with :func:`nilearn.datasets.fetch_development_fmri` and
   smooth it with an :term:`FWHM` varying from 0mm to 20mm in increments of 5mm

   **Intermediate steps:**

   1. Run :func:`nilearn.datasets.fetch_development_fmri` and inspect the ``.keys()`` of the returned object

   2. Check the :mod:`nilearn.image` module in the documentation to find a function to compute the mean of a 4D image

   3. Check the :mod:`nilearn.image` module again to find a function which smoothes images

   4. Plot the computed image for each smoothing value

   A solution can be found :ref:`here <sphx_glr_auto_examples_06_manipulating_images_plot_smooth_mean_image.py>`.

Learning with examples
----------------------

``nilearn`` comes with a lot of :ref:`examples/tutorials <tutorial_examples>`.
Going through them should give you a precise overview of what you can achieve with this package.

For new-comers, we recommend going through the following examples in the suggested order:

.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A simple example showing how to load an existing Nifti file and use basic nilearn functiona...">

.. only:: html

  .. image:: /auto_examples/00_tutorials/images/thumb/sphx_glr_plot_nilearn_101_thumb.png
    :alt: Basic nilearn example: manipulating and looking at data

  :ref:`sphx_glr_auto_examples_00_tutorials_plot_nilearn_101.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Basic nilearn example: manipulating and looking at data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Here we discover how to work with 3D and 4D niimgs.">

.. only:: html

  .. image:: /auto_examples/00_tutorials/images/thumb/sphx_glr_plot_3d_and_4d_niimg_thumb.png
    :alt: 3D and 4D niimgs: handling and visualizing

  :ref:`sphx_glr_auto_examples_00_tutorials_plot_3d_and_4d_niimg.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D and 4D niimgs: handling and visualizing</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Here is a simple tutorial on decoding with nilearn. It reproduces the Haxby 2001 study on a fac...">

.. only:: html

  .. image:: /auto_examples/00_tutorials/images/thumb/sphx_glr_plot_decoding_tutorial_thumb.png
    :alt: A introduction tutorial to fMRI decoding

  :ref:`sphx_glr_auto_examples_00_tutorials_plot_decoding_tutorial.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A introduction tutorial to fMRI decoding</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we use a General Linear Model (:term:`GLM`) to compare the fMRI signal during...">

.. only:: html

  .. image:: /auto_examples/00_tutorials/images/thumb/sphx_glr_plot_single_subject_single_run_thumb.png
    :alt: Intro to GLM Analysis: a single-session, single-subject fMRI dataset

  :ref:`sphx_glr_auto_examples_00_tutorials_plot_single_subject_single_run.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Intro to GLM Analysis: a single-session, single-subject fMRI dataset</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we will project a 3D statistical map onto a cortical mesh using vol_to_surf, d...">

.. only:: html

  .. image:: /auto_examples/01_plotting/images/thumb/sphx_glr_plot_3d_map_to_surface_projection_thumb.png
    :alt: Making a surface plot of a 3D statistical map

  :ref:`sphx_glr_auto_examples_01_plotting_plot_3d_map_to_surface_projection.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Making a surface plot of a 3D statistical map</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows manual steps to create and further modify an ROI spatial mask. They represen...">

.. only:: html

  .. image:: /auto_examples/06_manipulating_images/images/thumb/sphx_glr_plot_roi_extraction_thumb.png
    :alt: Computing a Region of Interest (ROI) mask manually

  :ref:`sphx_glr_auto_examples_06_manipulating_images_plot_roi_extraction.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Computing a Region of Interest (ROI) mask manually</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Here, we will go through a full step-by-step example of fitting a GLM to experimental data and ...">

.. only:: html

  .. image:: /auto_examples/04_glm_first_level/images/thumb/sphx_glr_plot_fiac_analysis_thumb.png
    :alt: Simple example of two-session fMRI model fitting

  :ref:`sphx_glr_auto_examples_04_glm_first_level_plot_fiac_analysis.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Simple example of two-session fMRI model fitting</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example compares different kinds of functional connectivity between regions of interest : ...">

.. only:: html

  .. image:: /auto_examples/03_connectivity/images/thumb/sphx_glr_plot_group_level_connectivity_thumb.png
    :alt: Classification of age groups using functional connectivity

  :ref:`sphx_glr_auto_examples_03_connectivity_plot_group_level_connectivity.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Classification of age groups using functional connectivity</div>
    </div>


.. raw:: html

    </div>


Finding help
------------

On top of this guide, there is a lot of content available outside of ``nilearn``
that could be of interest to new-comers:

1. `An introduction to fMRI <https://www.cs.mtsu.edu/~xyang/fMRIHandBook.pdf>`_ by Russel Poldrack, Jeanette Mumford and Thomas Nichols.

2. (For French readers) `An introduction to cognitive neuroscience <https://psy3018.github.io/intro.html>`_ given at the University of Montréal.

3. The documentation of ``scikit-learn`` explains each method with tips on practical use and examples: :sklearn:`\ `.  While not specific to neuroimaging, it is often a recommended read.

4. (For Python beginners) A quick and gentle introduction to scientific computing with Python with the `scipy lecture notes <http://scipy-lectures.github.io/>`_.
Moreover, you can use ``nilearn`` with `Jupyter <http://jupyter.org>`_ notebooks or
`IPython <http://ipython.org>`_ sessions. They provide an interactive
environment that greatly facilitates debugging and visualisation.


Besides, you can find help on :neurostars:`neurostars <>` for questions
related to ``nilearn`` and to computational neuroscience in general.
Finally, the ``nilearn`` team organizes weekly :ref:`drop-in hours <quickstart>`.
We can also be reached on :nilearn-gh:`github <issues>`
in case you find a bug.


Machine learning applications to Neuroimaging
=============================================

``nilearn`` brings easy-to-use machine learning tools that can be leveraged to solve more complex applications.
The interested reader can dive into the following articles for more content.

We give a non-exhaustive list of such important applications.

**Diagnosis and prognosis**

Predicting a clinical score or even treatment response
from brain imaging with :ref:`supervised
learning <decoding>` e.g. `[Mourao-Miranda 2012]
<http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0029482>`_

**Information mapping**

Using the prediction accuracy of a classifier
to characterize relationships between brain images and stimuli. (e.g.
:ref:`searchlight <searchlight>`) `[Kriegeskorte 2006]
<http://www.pnas.org/content/103/10/3863.short>`_

**Transfer learning**

Measuring how much an estimator trained on one
specific psychological process/task can predict the neural activity
underlying another specific psychological process/task
(e.g. discriminating left from
right eye movements also discriminates additions from subtractions
`[Knops 2009] <http://www.sciencemag.org/content/324/5934/1583.short>`_)

**High-dimensional multivariate statistics**

From a statistical point of view, machine learning implements
statistical estimation of models with a large number of parameters.
Tricks pulled in machine learning (e.g. regularization) can
make this estimation possible despite the usually
small number of observations in the neuroimaging domain
`[Varoquaux 2012] <http://icml.cc/2012/papers/688.pdf>`_. This
usage of machine learning requires some understanding of the models.

**Data mining / exploration**

Data-driven exploration of brain images. This includes the extraction of
the major brain networks from resting-state data ("resting-state networks")
or movie-watching data as well as the discovery of connectionally coherent
functional modules ("connectivity-based parcellation").
For example,
:ref:`extracting_rsn` or :ref:`parcellating_brain` with clustering.
