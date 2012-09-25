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


.. literalinclude:: ../plot_rest_clustering.py
    :start-after: ### Load nyu_rest dataset #####################################################
    :end-before: ### Ward ######################################################################

Sometimes, you may want to preprocess data by yourself a little.
In this example, we will restrict Haxby dataset to 150 frames to speed up
computation. To do that, we load the dataset, restrain it to 150 frames and
build a brand new Nifti like object. There is no need to save your data in
a file to pass it to Nifti masker. Simply use your Niimg.


.. literalinclude:: ../plot_haxby_masking.py
    :start-after: from nisl import datasets, io, utils  
    :end-before: # Display helper

.. _masking:

Fit step : masking
==================

The main functionality of the Nifti Masker is obviously masking. It simply can
apply a mask to your data, or generate one if you want. The great advantage of
using the masker is that it can be easily embedded in a scikit-learn pipeline.

Mask Visualization
------------------

Before exploring the subject, we define an helper function to display the
masks. This function will display a background (compose of a mean of epi scans)
and the mask as a red layer over this background.


.. literalinclude:: ../plot_haxby_masking.py
    :start-after: haxby_img = Nifti1Image(haxby_func, haxby_img.get_affine()) 
    :end-before: # Generate mask with default parameters 


Mask Computing
--------------

As said before, if a mask is not given, the Nifti Masker will try to compute
one. It is *very important* to take a look at the generated mask, to see if it
is suitable for your data and adjust parameters if it is not. See documentation
for a complete list of mask computation parameters.

As an example, we will now try to build a mask on a dataset form scratch. Haxby
dataset will be used since it provides a mask that we can use as a reference.

The first the step of the generation is to generate a mask with default
parameters and take a look at it. As an indicator, we can, for example, compare
the mask to original data. Here we count the number of non-zero pixels outside
of the mask.
i

.. literalinclude:: ../plot_haxby_masking.py
    :start-after: # Generate mask with default parameters
    :end-before: # Generate mask with opening

.. figure:: auto_examples/images/plot_haxby_masking_1.png
    :target: auto_examples/plot_haxby_masking.html
    :align: center
    :scale: 50%

With naked eyes, we can see that there is a problem with this mask. In fact,
it does not cover a part of the brain and the outline of the mask is
not very smooth. As we want to enlarge the mask a little bit and make it
smoother, we try to apply opening (*mask_opening=true*).


.. literalinclude:: ../plot_haxby_masking.py
    :start-after: # Generate mask with opening 
    :end-before: # Generate mask with upper cutoff

.. figure:: auto_examples/images/plot_haxby_masking_2.png
    :target: auto_examples/plot_haxby_masking.html
    :align: center
    :scale: 50%

This is not very effective. If we look at 
:class:`nisl.masking.compute_epi_mask` documentation, we spot two interesting
parameters: *lower_cutoff* and *upper_cutoff*. The algorithm seems to ignore
dark (low) values. Without getting into the details of the algorithm, this
means that the threshold is chosen into high values. We can tell the algorithm
to ignore high values by lowering *upper cutoff*. Default value is 0.9, so we
try 0.8.


.. literalinclude:: ../plot_haxby_masking.py
    :start-after: # Generate mask with upper cutoff 

.. figure:: auto_examples/images/plot_haxby_masking_3.png
    :target: auto_examples/plot_haxby_masking.html
    :align: center
    :scale: 50%


The resulting mask seems correct. If we compare it to the original one, they
are very close.

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
