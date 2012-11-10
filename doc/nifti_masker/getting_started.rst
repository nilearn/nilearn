.. for doctests to run, we need to fetch the dataset
    >>> from nisl import datasets, utils
    >>> from nisl.io import NiftiMasker
    >>> dataset = datasets.fetch_nyu_rest(n_subjects=1)


.. _getting_started:

================================================================================
Getting started
================================================================================

.. currentmodule:: nisl.io.nifti_masker

.. topic:: Steps

   Nisl provides a scikit-learn compliant transformer that helps loading Nifti
   files and provides some necessary preprocessing:

    1. :ref:`data_loading` : load Nifti files and check consistency of data
    2. :ref:`masking` : if a mask is not provided, computes one
    3. :ref:`resampling`
    4. :ref:`temporal_filtering`: detrending, confounds, normalization

.. _data_loading:

Data loading
============

The typical way to load data is to use a fetching function provided by Nisl
which will download a dataset and return a bunch of paths to the dataset
files. Each returned object is dataset specific. In fact there is no generic
dataset type in nisl: each dataset singularities are conserved. The functional
files of the fetched dataset can then be passed directly to the
:class:`NiftiMasker`.
For example :

.. literalinclude:: ../../plot_nifti_simple.py
    :start-after: ### Load nyu_rest dataset #####################################################
    :end-before: ### Compute the mask ##########################################################

We can now take a look at what Nisl loaded. It is only filenames referring to
dataset files on the disk::

  >>> dataset.keys()
  ['anat_skull', 'session', 'anat_anon', 'func']
  >>> dataset.func # doctest: +ELLIPSIS
  ['.../nyu_rest/session1/sub05676/func/lfo.nii.gz']
  >>> dataset.session
  [1]

We can also see that the data fetcher has generated
a session array, this is explained in the advanced tutorial.

.. note:: Custom data

   Please note that there are several ways to pass data to the Nifti Masker.
   You are not limited to the datasets proposed by nisl. Please take a look
   at :ref:`nifti_masker_advanced` to see how to use your own datasets or modify
   proposed dataset prior to passing them to the masker. 

.. _masking:

Masking (fit)
=============

The main functionality of the Nifti Masker is obviously masking. It simply can
apply a mask to your data, or generate one if you want. The great advantage of
using the masker is that it can be easily embedded in a scikit-learn pipeline.

Mask Computing
--------------

If your dataset does not provide a mask, the Nifti masker will compute one
for you. This is done in the `fit` step of the transformer. The generated
mask can be accessed via the `mask_` member and visualized.

.. literalinclude:: ../../plot_nifti_simple.py
    :start-after: ### Compute the mask ##########################################################
    :end-before: ### Visualize the mask ########################################################

.. note:: Transpose
   
   You may notice that we initialized the masker with parameter
   `transpose=True`. This is because we will process our data with ICA that
   processes temporal series. Please see :ref:`ica_rest` page to
   understand what is done here. Understanding that is not required for this
   example.

Mask Visualization
------------------

We can easily get the mask from the Nifti Masker and display it. Here we use an
EPI slice as a background and display the mask as a red overlay.

.. literalinclude:: ../../plot_nifti_simple.py
    :start-after: ### Visualize the mask ########################################################
    :end-before: ### Preprocess data ###########################################################

.. figure:: ../auto_examples/images/plot_nifti_simple_1.png
    :target: ../auto_examples/plot_nifti_simple.html
    :align: center
    :scale: 50%

Running default preprocessing (transform)
=========================================

The transformer is a classic scikit-learn transformer. You can invoke him
thanks to `fit` and `transform` methods or by a single call to `fit_transform`.

.. literalinclude:: ../../plot_nifti_simple.py
    :start-after: ### Preprocess data ###########################################################
    :end-before: ### Run an algorithm ##########################################################

Processing: running ICA
============================

As a processing, we invoke scikit-learn ICA.

.. literalinclude:: ../../plot_nifti_simple.py
    :start-after: ### Run an algorithm ##########################################################
    :end-before: ### Reverse masking ###########################################################

Unmasking (inverse_transform)
=============================

Unmasking data is as easy as masking it! Just call `inverse_transform` on your processed data.

.. literalinclude:: ../../plot_nifti_simple.py
    :start-after: ### Reverse masking ###########################################################
    :end-before: ### Show results ##############################################################

Visualizing results
===================

We visualize the result in a classical way. Here we show the default network.
Please note that to end up with a better ICA map, some processing on the ICA
maps is required: normalizing, centering and thresholding. Please see the ICA
example for a more accurate representation of the maps.

.. literalinclude:: ../../plot_nifti_simple.py
    :start-after: ### Show results ##############################################################
    :end-before: ### The same with a pipeline ##################################################

.. figure:: ../auto_examples/images/plot_nifti_simple_2.png
    :target: ../auto_examples/plot_nifti_simple.html
    :align: center
    :scale: 50%

Using the power of pipeline
===========================

As said before, if the nifti masker is a scikit-learn transformer. As such, it
can be integrated in a scikit-learn pipeline. The previous lines can be
replaces by this more compact version:

.. literalinclude:: ../../plot_nifti_simple.py
    :start-after: ### The same with a pipeline ##################################################

Going further
=============

Nifti masker
------------

Do you want to get the full potential of the Nifti Masker ? Take a look at the
advanced tutorial where you will learn to feed the masker with your own data
and tweak the parameters to get the best result !

ICA
---

Seduced by the ICA algorithm ? See the full example: :ref:`ica_rest`.
