.. _extracting_data:

======================================================
Extracting data: input/output and basic transformation
======================================================

Before applying some complex machine learning algorithm, or perform
sophisticated analysis, the first step is to read data from file and
do some basic transformation on them. Nisl offers several ways to do
this. This part is concerned with only high-level classes (in
modules nisl.io.nifti_masker and nisl.io.nifti_region), description of
low-level functions can be found in the reference documentation.

The philosophy underlying these classes is similar to scikit-learn's
transformers. Objects are initialized with some parameters unrelated
to the data, then the fit() method is called, possibly with some
information related to the data (such as number of images to process).
This step performs some initial computations (e.g. fitting a mask
based on the data). Then transform() is called, with the data as argument.
This method is meant to perform some computation on the dataset itself
(e.g. extracting timeseries from images).

The following parts explain how to use this API to extract voxel or
region signals from fMRI data. Advanced usage of these classes
requires to tweak their parameters. However, some choices have been
made in the design, and not every operation is possible. It is thus
sometimes necessary to go beyond them and use low-level functions to
achieve what is wanted.

.. currentmodule:: nisl.io

.. _nifti_masker:

:class:`NiftiMasker`: loading, masking and filtering
=========================================================

This section describes how to use the :class:`NiftiMasker` class in
more details than the previous description. :class:`NiftiMasker` is a
powerful tool to load images and extract voxel signals in the area
defined by the mask. It is designed to apply some basic preprocessing
steps by default with commonly used default parameters. But it is
*very important* to look at your data to see the effects of the
preprocessings and validate them.

In addition, :class:`NiftiMasker` is a scikit-learn compliant
transformer so that you can directly plug it into a scikit-learn
pipeline.

Custom data loading
--------------------

Sometimes, some custom preprocessing of data is necessary. In this
example, we will restrict Haxby dataset (which contains 1452 frames)
to 150 frames to speed up computation. To do that, we load the
dataset, restrain it to 150 frames and build a brand new Nifti-like
object to give it to the Nifti masker. Though it is possible, there is
no need to save your data in a file to pass it to a NiftiMasker. Simply
use `nibabel` to create a :ref:`Niimg <niimg>` in memory:


.. literalinclude:: ../plot_nifti_advanced.py
    :start-after: from nisl import datasets
    :end-before: # Display helper

.. _masking:

Custom Masking
---------------

In the basic tutorial, we showed how the masker could compute a mask
automatically, and it has done a good job. But, on some datasets, the
default algorithm performs poorly. This is why it is very important to
*always look at how your data look like*.

Mask Visualization
...................

Before exploring the subject, we define a helper function to display
masks. This function will display a background (composed of a mean of
epi scans) and a mask as a red layer over this background.


.. literalinclude:: ../plot_nifti_advanced.py
    :start-after: haxby_img = nibabel.Nifti1Image(haxby_func, haxby_img.get_affine())
    :end-before: # Generate mask with default parameters


Computing the mask
...................

.. currentmodule:: nisl.io

If a mask is not given, :class:`NiftiMasker` will try to compute
one. It is *very important* to take a look at the generated mask, to see if it
is suitable for your data and adjust parameters if it is not. See the
:class:`NiftiMasker` documentation for a complete list of mask computation
parameters.

As an example, we will now try to build a mask based on a dataset from
scratch. The Haxby dataset will be used since it provides a mask that we
can use as a reference.

.. figure:: auto_examples/images/plot_nifti_advanced_1.png
    :target: auto_examples/plot_nifti_advanced.html
    :align: right
    :scale: 50%

The first step is to generate a mask with default parameters and take
a look at it. As an indicator, we can, for example, compare the mask
to original data.

.. literalinclude:: ../plot_nifti_advanced.py
    :start-after: # Generate mask with default parameters
    :end-before: # Generate mask with opening

With naked eyes, we can see that the outline of the mask is not very
smooth. To make it smoother, try applying opening
(*mask_opening=true*).

.. figure:: auto_examples/images/plot_nifti_advanced_2.png
    :target: auto_examples/plot_nifti_advanced.html
    :align: right
    :scale: 50%

.. literalinclude:: ../plot_nifti_advanced.py
    :start-after: # Generate mask with opening
    :end-before: # Generate mask with upper cutoff

Looking at the :class:`NiftiMasker` object, we
see two interesting parameters: *lower_cutoff* and *upper_cutoff*. The
algorithm ignores dark (low) values. We can tell the algorithm to ignore
high values by lowering *upper cutoff*. Default value is 0.9, so we try
0.8 to lower a bit the threshold and get a larger mask.


.. literalinclude:: ../plot_nifti_advanced.py
    :start-after: # Generate mask with upper cutoff
    :end-before: # trended vs detrended

.. figure:: auto_examples/images/plot_nifti_advanced_3.png
    :target: auto_examples/plot_nifti_advanced.html
    :align: center
    :scale: 50%

The resulting mask seems to be correct. Compared to the original one,
it is very close.


Preprocessing
-------------

.. _resampling:

Resampling
..........

:class:`NiftiMasker` offers two ways to resample images:

  * *target_affine*: resample (resize, rotate...) images by providing a new affine
  * *target_shape*: resize images by directly providing a new shape

Resampling can be used for example to reduce processing time by lowering image
resolution.


Smoothing
.........

If smoothing the data prior to converting to voxel signals is required, it can
be performed by :class:`NiftiMasker`. It is achieved by passing the full-width
half maximum (in millimeter) along each axis in the parameter `smoothing_fwhm`.
For an isotropic filtering, passing a scalar is also possible. The underlying
function handles properly the tricky case of non-cubic voxels, by scaling the
given widths appropriately.


.. _temporal_filtering:

Temporal Filtering
..................

All previous filters operate on images, before conversion to voxel signals.
NiftiMasker can also process voxel signals. Here are the possibilities:

- Confound removal. Two ways of removing confounds are provided. Any linear
  trend can be removed by activating the `detrend` option. It is not activated
  by default in NiftiMasker but is almost essential. More complex confounds can
  be removed by passing them to transform(). If the dataset provides a confounds
  file, just pass its path to the masker.

- Linear filtering. Low-pass and high-pass filters allow for removing artifacts.
  Care has been taken to apply this processing to confounds if necessary.

- Normalization. Signals can be normalized (scaled to unit variance) before
  returning them. This is performed by default.

.. note:: **Exercise**
   :class: green

   You can, more as a training than as an exercise, try to play with the
   parameters in Nisl examples. Try to enable detrending in haxby decoding
   and run it: does it have a big impact on the result?


Inverse transform: unmasking data
----------------------------------

Once voxel signals have been processed, the result can be visualized as images
after unmasking (turning voxel signals into a series of images, using the same
mask as for masking). This step is present in almost all the examples provided
in Nisl.


.. literalinclude:: ../plot_haxby_decoding.py
    :start-after: svc = feature_selection.inverse_transform(svc)
    :end-before: # We use a masked array so that the voxels at '-1' are displayed


Extraction of signals from regions:\  :class:`NiftiLabelsMasker`, :class:`NiftiMapsMasker`.
===========================================================================================

The purpose of :class:`NiftiLabelsMasker` and :class:`NiftiMapsMasker` is to
compute signals from regions containing many voxels. They make it easy to get
these signals once you have an atlas or a parcellation.

Regions definition
------------------

Nisl understands two different way of defining regions, which are called
labels and maps, handled respectively by :class:`NiftiLabelsMasker` and
:class:`NiftiMapsMasker`.

- labels: a single region is defined as the set of all the voxels that have a
  common label (usually an integer) in the region definition array. The set of
  region is defined by a single 3D array, containing at each location the label
  of the region the voxel is in. This technique has one big advantage: the
  amount of memory required is independent of the number of regions, allowing
  for representing a large number of regions. On the other hand, there are
  several contraints: regions cannot overlap, and only their support could be
  represented (no weighting).
- maps: a single region is defined as the set of all the voxels that have a
  non-zero weight. A set of regions is thus defined by a set of 3D images (or a
  single 4D image), one 3D image per region. Overlapping regions with weights
  can be represented, but with a storage cost that scales linearly with the
  number of regions. Handling a large number (thousands) of regions will prove
  difficult with this representation.

NiftiMapsMasker Usage
---------------------

Usage of :class:`NiftiMapsMasker` and :class:`NiftiLabelsMasker` is very close,
and very close to the usage of NiftiMasker. Only options specific to
NiftiMapsMasker and NiftiLabelsMasker are described in this section.

Nisl provided a few downloaders to get a brain parcellation. Load the MSDL one:


.. literalinclude:: ../plot_adhd_covariance.py
    :start-after: print("-- Loading raw data ({0:d}) and masking ...".format(subject_n))
    :end-before: print("-- Computing confounds ...")

This atlas defines its regions using maps. The path to the corresponding file
can be found under the "maps" key. Assuming that a confounds file name is in the
variable "confounds", extracting region signals can be performed like this:


.. literalinclude:: ../plot_adhd_covariance.py
   :start-after: print("-- Computing region signals ...")
   :end-before: print("-- Computing covariance matrices ...")

`region_ts` is a numpy.ndarray, containing one signal per column.

One important thing that happens transparently during the execution of
fit_transform() is resampling. Initially, the images and the atlas do not have
the same shape nor the same affine. Getting them to the same format is required
for the signals extraction to work. The keyword argument `resampling_target`
specifies which format everything should be resampled to. In the present case,
"maps" indicates that all images should be resampled to have the same shape and
affine as the msdl atlas. See the reference documentation for every possible
option.

`region_ts` can then be used as input to a scikit-learn transformer. In the
present case, covariance between region signals can be obtained using the graph
lasso algorithm:

.. literalinclude:: ../plot_adhd_covariance.py
   :start-after: print("-- Computing covariance matrices ...")
   :end-before: plot_matrices(estimator.covariance_, -estimator.precision_,


NiftiLabelsMasker Usage
-----------------------

Usage of :class:`NiftiLabelsMasker` is similar to that of
:class:`NiftiMapsMasker`. The main difference is that it requires a labels image
instead of a set of maps as input.

The `background_label` keyword of :class:`NiftiLabelsMasker` deserves some
explanation. The voxels that correspond to the brain in an fMRI image do not
fill the entire image. Consequently, in the labels image, there must be a label
corresponding to "outside" the brain, for which no signal should be extracted.
By default, this label is set to zero in Nisl, and is referred to as
"background". Should some non-zero value occur, it is possible to change the
background value with the `background_label` keyword.
