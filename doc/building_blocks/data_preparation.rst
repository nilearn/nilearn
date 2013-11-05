.. _extracting_data:

=========================================================
Data preparation: input/output and basic transformation
=========================================================

Before applying some complex machine learning algorithm, or perform
sophisticated analysis, the first step is to read data from file and
do some basic transformation on them. Nilearn offers several ways to do
this. This part is concerned with only high-level classes (in
modules :mod:`nilearn.input_data`), the description of
low-level functions can be found in the reference documentation.

The philosophy underlying these classes is similar to `scikit-learn
<http://scikit-learn.org>`_\ 's
transformers. Objects are initialized with some parameters proper to
the transformation (unrelated to the data), then the fit() method
should be called, possibly specifying some data-related
information (such as number of images to process), to perform some
initial computation (e.g. fitting a mask based on the data). Then
transform() can be called, with the data as argument, to perform some
computation on data themselves (e.g. extracting timeseries from images).

The following parts explain how to use this API to extract voxel or
region signals from fMRI data. Advanced usage of these classes
requires to tweak their parameters. However, high-level objects have
been designed to perform common operations. Users who want to make
some specific processing may have to call low-level functions (see
e.g. :mod:`nilearn.signal`, :mod:`nilearn.masking`.)

.. currentmodule:: nilearn.input_data

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

In addition, :class:`NiftiMasker` is a `scikit-learn
<http://scikit-learn.org>`_ compliant
transformer so that you can directly plug it into a `scikit-learn
<http://scikit-learn.org>`_ pipeline.

Custom data loading
--------------------

Sometimes, some custom preprocessing of data is necessary. In this
example, we will restrict Haxby dataset (which contains 1452 frames)
to 150 frames to speed up computation. To do that, we load the dataset
with :func:`fetch_haxby_simple() <nilearn.datasets.fetch_haxby_simple>`,
restrict it to 150 frames and build a brand new Nifti-like object to
give it to the masker. Though it is possible, there is no need to save
your data in a file to pass it to a :class:`NiftiMasker`. Simply use
`nibabel <http://nipy.sourceforge.net/nibabel/>`_ to create a
:ref:`Niimg <niimg>` in memory:


.. literalinclude:: ../plot_nifti_advanced.py
    :start-after: from nilearn import datasets
    :end-before: # Display helper

Custom Masking
---------------

In the basic tutorial, we showed how the masker could compute a mask
automatically, and it has done a good job. But, on some datasets, the
default algorithm performs poorly. This is why it is very important to
**always look at how your data look like**.

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
smooth. To make it less smooth, bypass the opening step
(*mask_opening=0*).

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


.. literalinclude:: ../../plot_nifti_advanced.py
    :start-after: # Generate mask with upper cutoff
    :end-before: # trended vs detrended

.. figure:: ../auto_examples/images/plot_nifti_advanced_3.png
    :target: ../auto_examples/plot_nifti_advanced.html
    :align: center
    :scale: 50%

The resulting mask seems to be correct. Compared to the original one,
it is very close.

.. note::

    The full example described in this section can be found here:
    :doc:`plot_nifti_advanced.py <../auto_examples/plot_nifti_advanced>`.
    This one can be relevant too:
    :doc:`plot_nifti_simple.py <../auto_examples/plot_nifti_simple>`.


Preprocessing
-------------

.. _resampling:

Resampling
..........

:class:`NiftiMasker` makes it possible to resample images. 
       The resampling procedure takes as input the
       *target_affine* to resample (resize, rotate...) images in order
       to match the spatial configuration defined by the new
       affine. Additionally, a *target_shape* can be used to resize
       images (i.e. croping or padding with zeros) to match an
       expected shape.

Resampling can be used for example to reduce processing time by
lowering image resolution. Certain image viewers also require images to be
resampled in order to allow image fusion.


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
:class:`NiftiMasker` can also process voxel signals. Here are the possibilities:

- Confound removal. Two ways of removing confounds are provided. Any linear
  trend can be removed by activating the `detrend` option. It is not activated
  by default in :class:`NiftiMasker` but is almost essential. More complex confounds can
  be removed by passing them to :meth:`NiftiMasker.transform`. If the
  dataset provides a confounds file, just pass its path to the masker.

- Linear filtering. Low-pass and high-pass filters can be used to remove artifacts.
  Care has been taken to apply this processing to confounds if necessary.

- Normalization. Signals can be normalized (scaled to unit variance) before
  returning them. This is performed by default.

.. topic:: **Exercise**

   You can, more as a training than as an exercise, try to play with
   the parameters in :doc:`plot_haxby_simple.py
   <auto_examples/plot_haxby_simple>`. Try to enable detrending
   and run the script: does it have a big impact on the result?


Inverse transform: unmasking data
----------------------------------

Once voxel signals have been processed, the result can be visualized as images
after unmasking (turning voxel signals into a series of images, using the same
mask as for masking). This step is present in almost all the
:doc:`examples <auto_examples/index>` provided in Nilearn.


.. literalinclude:: ../plot_haxby_decoding.py
    :start-after: svc = feature_selection.inverse_transform(svc)
    :end-before: # We use a masked array so that the voxels at '-1' are displayed


.. _region:

Extraction of signals from regions:\  :class:`NiftiLabelsMasker`, :class:`NiftiMapsMasker`.
===========================================================================================

The purpose of :class:`NiftiLabelsMasker` and :class:`NiftiMapsMasker` is to
compute signals from regions containing many voxels. They make it easy to get
these signals once you have an atlas or a parcellation.

Regions definition
------------------

Nilearn understands two different way of defining regions, which are called
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

:class:`NiftiMapsMasker` Usage
------------------------------

.. note::

   The full example in this section can be found here:
   :doc:`plot_adhd_covariance.py <../auto_examples/plot_adhd_covariance>`

Usage of :class:`NiftiMapsMasker` and :class:`NiftiLabelsMasker` is very close,
and very close to the usage of :class:`NiftiMasker`. Only options specific to
:class:`NiftiMapsMasker` and :class:`NiftiLabelsMasker` are described
in this section.

Nilearn provides several downloaders to get a brain parcellation. Load
the `MSDL one
<https://team.inria.fr/parietal/research/spatial_patterns/spatial-patterns-in-resting-state/>`_:


.. literalinclude:: ../plot_adhd_covariance.py
    :start-after: print("-- Fetching datasets ...")
    :end-before: dataset = nilearn.datasets.fetch_adhd()

This atlas defines its regions using maps. The path to the corresponding file
can be found under the "maps" key. Extracting region signals for
several subjects can be performed like this:

.. literalinclude:: ../plot_adhd_covariance.py
   :start-after: atlas = nilearn.datasets.fetch_msdl_atlas()
   :end-before: print("-- Computing group-sparse precision matrices ...")

`region_ts` is a `numpy.ndarray
<http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_,
containing one signal per column. The `subjects` list contains signals
for several subjects. In this example, confounds removal is performed,
by both using a provided list of confounds, and by computing new ones
using :func:`nilearn.image.high_variance_confounds`.


One important thing that happens transparently during the execution of
:meth:`NiftiMasker.fit_transform` is resampling. Initially, the images
and the atlas do not have the same shape nor the same affine. Getting
them to the same format is required for the signals extraction to
work. The keyword argument `resampling_target` specifies which format
everything should be resampled to. In the present case, "maps"
indicates that all images should be resampled to have the same shape
and affine as the `MSDL atlas
<https://team.inria.fr/parietal/research/spatial_patterns/spatial-patterns-in-resting-state/>`_.
See the reference documentation for :class:`NiftiMasker` for every
possible option.

The :class:`NiftiMapsMasker` output can then be used as input to a
`scikit-learn <http://scikit-learn.org>`_ transformer. In the present
case, covariance between region signals can be obtained for each
subject either using the `graph lasso
<http://biostatistics.oxfordjournals.org/content/9/3/432.short>`_
or the `group-sparse covariance <http://arxiv.org/abs/1207.4255>`_
algorithm:

.. literalinclude:: ../plot_adhd_covariance.py
   :start-after: subjects.append(region_ts)
   :end-before: print("-- Displaying results")


NiftiLabelsMasker Usage
-----------------------

Usage of :class:`NiftiLabelsMasker` is similar to that of
:class:`NiftiMapsMasker`. The main difference is that it requires a labels image
instead of a set of maps as input.

The `background_label` keyword of :class:`NiftiLabelsMasker` deserves
some explanation. The voxels that correspond to the brain or a region
of interest in an fMRI image do not fill the entire
image. Consequently, in the labels image, there must be a label
corresponding to "outside" the brain, for which no signal should be
extracted.  By default, this label is set to zero in Nilearn, and is
referred to as "background". Should some non-zero value occur, it is
possible to change the background value with the `background_label`
keyword.
