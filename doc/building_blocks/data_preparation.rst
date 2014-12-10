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
pipeline <http://scikit-learn.org/stable/modules/pipeline.html>`_.

Custom data loading
--------------------

Sometimes, some custom preprocessing of data is necessary. For instance
we can restrict a dataset to the first 100 frames. Below, we load
a resting-state dataset with :func:`fetch_fetch_nyu_rest()
<nilearn.datasets.fetch_nyu_rest>`, restrict it to 100 frames and
build a brand new Nifti-like object to give it to the masker. Though it
is possible, there is no need to save your data in a file to pass it to a
:class:`NiftiMasker`. Simply use `nibabel
<http://nipy.sourceforge.net/nibabel/>`_ to create a :ref:`Niimg <niimg>`
in memory:


.. literalinclude:: ../../examples/manipulating_images_and_visualization/plot_mask_computation.py
    :start-after: Load NYU resting-state dataset
    :end-before: # To display the background

Controlling how the mask is computed from the data
-----------------------------------------------------

In the basic tutorial, we showed how the masker could compute a mask
automatically, and it has done a good job. But, on some datasets, the
default algorithm performs poorly. This is why it is very important to
**always look at how your data look like**.

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

.. figure:: ../auto_examples/manipulating_images_and_visualization/images/plot_mask_computation_2.png
    :target: ../auto_examples/plot_mask_computation.html
    :align: right
    :scale: 50%

The first step is to generate a mask with default parameters and take
a look at it.

.. literalinclude:: ../../examples/manipulating_images_and_visualization/plot_mask_computation.py
    :start-after: # Simple mask extraction from EPI images
    :end-before: # Generate mask with strong opening

____

.. figure:: ../auto_examples/manipulating_images_and_visualization/images/plot_mask_computation_3.png
    :target: ../auto_examples/plot_mask_computation.html
    :align: right
    :scale: 50%

We can make the outline of the mask more by increasing the number of
opening steps (*opening=10*) using the `mask_args` argument of the
:class:`NiftiMasker`.

.. literalinclude:: ../../examples/manipulating_images_and_visualization/plot_mask_computation.py
    :start-after: # Generate mask with strong opening
    :end-before: # Generate mask with a high lower cutoff

____

Looking at the :func:`nilearn.masking.compute_epi_mask` called by the
:class:`NiftiMasker` object, we see two interesting parameters:
*lower_cutoff* and *upper_cutoff*. These set the grey-values bounds in
which the masking algorithm is going to try to find it's threshold (where
0 is the minimum of the image, and 1 the maximum). Here we raise a lot
the lower cutoff, and thus force the masking algorithm to select only
voxels that are very light on the EPI image.

.. figure:: ../auto_examples/manipulating_images_and_visualization/images/plot_mask_computation_4.png
    :target: ../auto_examples/plot_mask_computation.html
    :align: right
    :scale: 50%


.. literalinclude:: ../../examples/manipulating_images_and_visualization/plot_mask_computation.py
    :start-after: # Generate mask with a high lower cutoff
    :end-before: ################################################################################


.. note::

    The full example described in this section can be found here:
    :doc:`plot_mask_computation.py <../auto_examples/manipulating_images_and_visualization/plot_mask_computation>`.
    This one can be relevant too:
    :doc:`plot_nifti_simple.py <../auto_examples/plot_nifti_simple>`.


Common data preparation steps: resampling, smoothing, filtering
----------------------------------------------------------------

.. seealso::

   If you don't want to use the :class:`NiftiMasker` to perform these
   simple operations on data, note that they are
   :ref:`corresponding functions <preprocessing_functions>`.

.. _resampling:

Resampling
..........

:class:`NiftiMasker` and many similar classes enable resampling images.
       The resampling procedure takes as input the
       *target_affine* to resample (resize, rotate...) images in order
       to match the spatial configuration defined by the new
       affine. Additionally, a *target_shape* can be used to resize
       images (i.e. croping or padding with zeros) to match an
       expected shape.

Resampling can be used for example to reduce processing time by
lowering image resolution. Certain image viewers also require images to be
resampled to display overlays.

Automatic computation of offset and bounding box can be performed by
specifying a 3x3 matrix instead of the 4x4 affine, in which case nilearn
computes automatically the translation part of the affine.

.. image:: ../auto_examples/manipulating_images_and_visualization/images/plot_affine_transformation_2.png
    :target: ../auto_examples/plot_affine_transformation.html
    :scale: 36%
.. image:: ../auto_examples/manipulating_images_and_visualization/images/plot_affine_transformation_4.png
    :target: ../auto_examples/plot_affine_transformation.html
    :scale: 36%
.. image:: ../auto_examples/manipulating_images_and_visualization/images/plot_affine_transformation_3.png
    :target: ../auto_examples/plot_affine_transformation.html
    :scale: 36%


.. topic:: **Special case: resampling to a given voxel size**

   Specifying a 3x3 matrix that is diagonal as a target_affine fixes the
   voxel size. For instance to resample to 3x3x3 mm voxels::

    >>> import numpy as np
    >>> target_affine = np.diag((3, 3, 3))

|

.. seealso::

   :func:`nilearn.image.resample_img`


Smoothing
.........

If smoothing the data prior to converting to voxel signals is required, it can
be performed by :class:`NiftiMasker`. It is achieved by passing the full-width
half maximum (in millimeter) along each axis in the parameter `smoothing_fwhm`.
For an isotropic filtering, passing a scalar is also possible. The underlying
function handles properly the tricky case of non-cubic voxels, by scaling the
given widths appropriately.

.. seealso::

   :func:`nilearn.image.smooth_img`


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
   the parameters in :ref:`example_plot_haxby_simple.py`. Try to enable detrending
   and run the script: does it have a big impact on the result?


.. seealso::

   :func:`nilearn.signal.clean`


Inverse transform: unmasking data
----------------------------------

Once voxel signals have been processed, the result can be visualized as
images after unmasking (turning voxel signals into a series of images,
using the same mask as for masking). This step is present in almost all
the :ref:`examples <examples-index>` provided in Nilearn. Below is
an excerpt of :ref:`the example performing Anova-SVM on the Haxby data
<example_decoding_plot_haxby_anova_svm.py>`):

.. literalinclude:: ../../examples/decoding/plot_haxby_anova_svm.py
    :start-after: ### Look at the SVC's discriminating weights
    :end-before: ### Create the figure


.. _region:

Extraction of signals from regions:\  :class:`NiftiLabelsMasker`, :class:`NiftiMapsMasker`.
===========================================================================================

The purpose of :class:`NiftiLabelsMasker` and :class:`NiftiMapsMasker` is to
compute signals from regions containing many voxels. They make it easy to get
these signals once you have an atlas or a parcellation.

Regions definition
------------------

Nilearn understands two different ways of defining regions, which are called
labels and maps, handled respectively by :class:`NiftiLabelsMasker` and
:class:`NiftiMapsMasker`.

- labels: a single region is defined as the set of all the voxels that have a
  common label (usually an integer) in the region definition array. The set of
  regions is defined by a single 3D array, containing at each location the label
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
   :doc:`plot_adhd_covariance.py <../auto_examples/connectivity/plot_adhd_covariance>`

Usage of :class:`NiftiMapsMasker` is very close to the usage of
:class:`NiftiMasker`. Only options specific to
:class:`NiftiMapsMasker` are described in this section.

Nilearn provides several downloaders to get a brain parcellation. Load
the `MSDL one
<https://team.inria.fr/parietal/research/spatial_patterns/spatial-patterns-in-resting-state/>`_:


.. literalinclude:: ../../examples/connectivity/plot_adhd_covariance.py
    :start-after: print("-- Fetching datasets ...")
    :end-before: dataset = nilearn.datasets.fetch_adhd()

This atlas defines its regions using maps. The path to the corresponding file
can be found under the "maps" key. Extracting region signals for
several subjects can be performed like this:

.. literalinclude:: ../../examples/connectivity/plot_adhd_covariance.py
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

.. literalinclude:: ../../examples/connectivity/plot_adhd_covariance.py
   :start-after: subjects.append(region_ts)
   :end-before: print("-- Displaying results")


:class:`NiftiLabelsMasker` Usage
---------------------------------

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
