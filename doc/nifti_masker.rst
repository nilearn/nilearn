.. _mri_transformer:

================================================================================
fMRI loading: Loading and preprocessing MRI data
================================================================================

.. topic:: Steps

   Nisl provides a scikit-compliant transformer that helps loading Nifti files
   and provides some necessary preprocessing:

    1. :ref:`data_loading` : load Nifti files and check consistency of data
    2. :ref:`masking` : if a mask is not provided, computes one
    3. :ref:`resampling`
    4. :ref:`temporal_filtering`: detrending, confounds, normalization

.. _data_loading:

Data loading
============

The Nifti Masker accepts several data formats. You can use the
dataset fetching functions provided by Nisl to directly obtain well formatted
data but it is easy to use your own dataset.

Niimg
-----
Niimg (pronounce ni-image) is a common term used in Nisl. It can either
represents:

  * a file path to a Nifti image
  * any object exposing ``get_data()`` and ``get_affine()`` methods (it is
    obviously intended to handle nibabel's Nifti1Image but also user custom
    types if needed).

The Nifti Masker requires a 4-dimensional Nifti-like data. Accepted inputs are:

  * Path to a 4-dimensional Nifti image
  * List of paths to 3-dimensional Nifti images
  * 4-dimensional Nifti-like object
  * List of 3-dimensional Nifti-like objects

.. note:: Image affines

   If you provide a sequence of Nifti images, all of them must have the same
   affine.

The typical way to load data is to use a fetching function which will return
a bunch of path to the dataset files and then pass it directly to the Nifti
masker. For example :


.. literalinclude:: ../plot_nifti_simple.py
    :start-after: ### Load nyu_rest dataset #####################################################
    :end-before: # Visualize the mask ##########################################################

You can take a look at what Nisl loaded. It is only filenames referring to
dataset files on the disk. You can also see that the data fetcher has generated
for you a session array that can be given directly to the Nifti masker.


.. _masking:

Fit step : masking
==================

The main functionality of the Nifti Masker is obviously masking. It simply can
apply a mask to your data, or generate one if you want. The great advantage of
using the masker is that it can be easily embedded in a scikit-learn pipeline.

Mask Computing
--------------

If your dataset does not provide a mask, the Nifti masker will compute one
for you. This is done in the `fit` step of the transformer. The generated
mask can be accessed via the `mask_` member and visualized.

.. literalinclude:: ../plot_nifti_simple.py
    :start-after: ### Load nyu_rest dataset #####################################################
    :end-before: # Visualize the mask ##########################################################


.. _resampling:

Transform step : preprocessings
===============================

Resampling
----------

Nifti Masker offers two ways to resample images:

  * *target_affine*: resample (resize, rotate...) images by providing a new affine
  * *target_shape*: resize images by providing directly a new shape

Resampling can be used for example to reduce processing time of an algorithm by
lowering image resolution.

.. _temporal_filtering:

Temporal Filtering
------------------

All previous filters concern spatial filtering. On the time axis, the Nifti
masker also proposes some filters.

By default, the signal will be normalized. If the dataset provides a confounds
file, it can be applied by providing the path to the file to the masker.
Low pass and High pass filters allows one to remove artefacts.

Detrending removes linear trend along axis from data. It is not activated by
default in the Nifti Masker but it is almost essential.

.. note:: Exercise

   You can, more as a training than as an exercise, try to play with the
   parameters in Nisl examples. Try to disable detrending in haxby decoding
   and run it: does it have a big impact on the results ?


Inverse transform: unmasking data
=================================

Once that your computation is finished, you want to unmask your data to be able
to visualize it. This step is present in almost all the examples provided in
Nisl.


.. literalinclude:: ../plot_haxby_decoding.py
    :start-after: svc = feature_selection.inverse_transform(svc)
    :end-before: # We use a masked array so that the voxels at '-1' are displayed
