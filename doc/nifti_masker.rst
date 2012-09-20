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

The Nifti Masker accepts several data formats. You can use the
dataset fetching functions provided by Nisl to directly obtain well formatted
data but it is easy to use your own dataset.

The Nifti Masker requires a 4-dimensional Nifti-like data (ie an object that
owns a ``get_data()`` and a ``get_affine()`` methods). Accepted inputs are:

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

The main functionality of the Nifti Masker is obviously masking. It can
obviously apply a mask to your data, or generate one if you want but, above
all, it allows you to do it simply and integrate it in your scikit-learn
pipeline.

Mask Computing
--------------

As said before, if a mask is not given, the Nifti Masker will try to compute
one. It is *very important* to take a look at the generated mask, to see if it
is suitable for your data and adjust parameters if it is not. See documentation
for a complete list of mask computation parameters.

We will now try to build a mask on a dataset form scratch. Haxby dataset will
be used since it provides a mask that we can use as a reference.



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
