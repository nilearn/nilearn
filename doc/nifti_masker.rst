.. _mri_transformer:

================================================================================
fMRI loading: Loading and preprocessing MRI data
================================================================================

.. topic:: Steps

   Nisl provides a scikit-compliant transformer that helps loading Nifti files
   and provides some necessary preprocessing:

    1. :ref:`data_loading` : load Nifti files and check consistency of data
    2. :ref:`mask_computing` : if a mask is not provided, computes one
    3. :ref:`resampling`
    4. :ref:`masking_and_smoothing`
    5. :ref:`temporal_filtering`: detrending, confounds, normalization

.. _data_loading:

Data loading
============

The Nifti Loader accepts several data formats. You can use the
dataset fetching functions provided by Nisl to directly obtain well formatted
data but it is easy to use your own dataset.

The Nifti Loader requires a 4-dimensional Nifti-like data (ie an object that
owns a ``get_data()`` and a ``get_affine()`` methods). Accepted inputs :

  * Path to a 4-dimensional Nifti image
  * List of paths to 3-dimensional Nifti images
  * 4-dimensional Nifti-like object
  * List of 3-dimensional Nifti-like objects

.. note:: Image affines

   If you provide a sequence of Nifti images, all of them must have the same
   affine.



.. _masking:

Masking
=======

One

.. _resampling:

Resampling
==========

Todo

.. _masking_and_smoothing:

Masking and smoothing
=====================

Todo

.. _temporal_filtering:

Temporal Filtering
==================

Todo
