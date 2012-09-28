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


