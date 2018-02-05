.. _masker_objects:

=====================================================================
From neuroimaging volumes to data matrices: the masker objects
=====================================================================

This chapter introduces the maskers: objects that go from
neuroimaging volumes, on the disk or in memory, to data matrices, eg of
time series.

.. contents:: **Chapters contents**
    :local:
    :depth: 1


The concept of "masker" objects
===============================

In any analysis, the first step is to load the data.
It is often convenient to apply some basic data
transformations and to turn the data in a 2D (samples x features) matrix,
where the samples could be different time points, and the features derived
from different voxels (e.g., restrict analysis to the ventral visual stream),
regions of interest (e.g., extract local signals from spheres/cubes), or
pre-specified networks (e.g., look at data from all voxels of a set of
network nodes). Think of masker objects as swiss-army knifes for shaping
the raw neuroimaging data in 3D space into the units of observation
relevant for the research questions at hand.


.. |niimgs| image:: ../images/niimgs.jpg
    :scale: 50%

.. |arrays| image:: ../images/feature_array.jpg
    :scale: 35%

.. |arrow| raw:: html

   <span style="padding: .5em; font-size: 400%">&rarr;</span>

.. centered:: |niimgs|  |arrow|  |arrays|



"masker" objects (found in modules :mod:`nilearn.input_data`)
simplify these "data folding" steps that often preceed the
statistical analysis.

Note that the masker objects may not cover all the image transformations
for specific tasks. Users who want to make some specific processing may
have to call :ref:`specific functions <preprocessing_functions>`
(modules :mod:`nilearn.signal`, :mod:`nilearn.masking`).

|

.. topic:: **Advanced: Design philosophy of "Maskers"**

    The design of these classes is similar to `scikit-learn
    <http://scikit-learn.org>`_\ 's transformers. First, objects are
    initialized with some parameters guiding the transformation
    (unrelated to the data). Then the `fit()` method should be called,
    possibly specifying some data-related information (such as number of
    images to process), to perform some initial computation (e.g.,
    fitting a mask based on the data). Finally, `transform()` can be
    called, with the data as argument, to perform some computation on
    data themselves (e.g., extracting time series from images).


.. currentmodule:: nilearn.input_data

.. _nifti_masker:

:class:`NiftiMasker`: applying a mask to load time-series
==========================================================

:class:`NiftiMasker` is a powerful tool to load images and
extract voxel signals in the area defined by the mask.
It applies some basic preprocessing
steps with commonly used parameters as defaults.
But it is *very important* to look at your data to see the effects
of the preprocessings and validate them.

.. topic:: **Advanced: scikit-learn Pipelines**

    :class:`NiftiMasker` is a `scikit-learn
    <http://scikit-learn.org>`_ compliant
    transformer so that you can directly plug it into a `scikit-learn
    pipeline <http://scikit-learn.org/stable/modules/pipeline.html>`_.


Custom data loading: loading only the first 100 time points
------------------------------------------------------------

Suppose we want to restrict a dataset to the first 100 frames. Below, we load
a resting-state dataset with :func:`fetch_adhd()
<nilearn.datasets.fetch_adhd>`, restrict it to 100 frames and
build a new niimg object that we can give to the masker. Although
possible, there is no need to save your data to a file to pass it to a
:class:`NiftiMasker`. Simply use :func:`nilearn.image.index_img` to apply a
slice and create a :ref:`Niimg <niimg>` in memory:


.. literalinclude:: ../../examples/04_manipulating_images/plot_mask_computation.py
    :start-after: Load ADHD resting-state dataset
    :end-before: # To display the background

Controlling how the mask is computed from the data
--------------------------------------------------

In this section, we show how the masker object can compute a mask
automatically for subsequent statistical analysis.
On some datasets, the default algorithm may however perform poorly.
This is why it is very important to
**always look at your data** before and after feature
engineering using masker objects.

.. note::

    The full example described in this section can be found here:
    :doc:`plot_mask_computation.py <../auto_examples/04_manipulating_images/plot_mask_computation>`.
    It is also related to this example:
    :doc:`plot_nifti_simple.py <../auto_examples/04_manipulating_images/plot_nifti_simple>`.


Visualizing the computed mask
..............................

If a mask is not specified as an argument, :class:`NiftiMasker` will try to
compute one from the provided neuroimaging data.
It is *very important* to verify the quality of the generated mask by visualization.
This allows to see whether it is suitable for your data and intended analyses.
Alternatively, the mask computation parameters can still be modified.
See the :class:`NiftiMasker` documentation for a complete list of
mask computation parameters.

As a first example, we will now automatically build a mask from a dataset.
We will here use the Haxby dataset because it provides the original mask that
we can compare the data-derived mask against.

Generate a mask with default parameters and visualize it (it is in the
`mask_img_` attribute of the masker):

.. literalinclude:: ../../examples/04_manipulating_images/plot_mask_computation.py
    :start-after: # Simple mask extraction from EPI images
    :end-before: # Generate mask with strong opening


.. figure:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_mask_computation_002.png
    :target: ../auto_examples/04_manipulating_images/plot_mask_computation.html
    :scale: 50%

Changing mask parameters: opening, cutoff
..........................................

We can then fine-tune the outline of the mask by increasing the number of
opening steps (`opening=10`) using the `mask_args` argument of the
:class:`NiftiMasker`. This effectively performs erosion and dilation operations
on the outer voxel layers of the mask, which can for example remove remaining
skull parts in the image.

.. literalinclude:: ../../examples/04_manipulating_images/plot_mask_computation.py
    :start-after: # Generate mask with strong opening
    :end-before: # Generate mask with a high lower cutoff


.. figure:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_mask_computation_003.png
    :target: ../auto_examples/04_manipulating_images/plot_mask_computation.html
    :scale: 50%


Looking at the :func:`nilearn.masking.compute_epi_mask` called by the
:class:`NiftiMasker` object, we see two interesting parameters:
`lower_cutoff` and `upper_cutoff`. These set the grey-value bounds in
which the masking algorithm will search for its threshold
(0 being the minimum of the image and 1 the maximum). We will here increase
the lower cutoff to enforce selection of those
voxels that appear as bright in the EPI image.


.. literalinclude:: ../../examples/04_manipulating_images/plot_mask_computation.py
    :start-after: # Generate mask with a high lower cutoff
    :end-before: ###############################################################################


.. figure:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_mask_computation_004.png
    :target: ../auto_examples/04_manipulating_images/plot_mask_computation.html
    :scale: 50%

.. _masker_preprocessing_steps:

Common data preparation steps: smoothing, filtering, resampling
----------------------------------------------------------------

:class:`NiftiMasker` comes with many parameters that enable data
preparation::

   >>> from nilearn import input_data
   >>> masker = input_data.NiftiMasker()
   >>> masker
   NiftiMasker(detrend=False, high_pass=None, low_pass=None, mask_args=None,
         mask_img=None, mask_strategy='background',
         memory=Memory(cachedir=None), memory_level=1, sample_mask=None,
         sessions=None, smoothing_fwhm=None, standardize=False, t_r=None,
         target_affine=None, target_shape=None, verbose=0)

The meaning of each parameter is described in the documentation of
:class:`NiftiMasker` (click on the name :class:`NiftiMasker`), here we
comment on the most important.

.. seealso::

   If you do not want to use the :class:`NiftiMasker` to perform these
   simple operations on data, note that they can also be manually
   accessed in nilearn such as in
   :ref:`corresponding functions <preprocessing_functions>`.

Smoothing
.........

:class:`NiftiMasker` can apply Gaussian spatial smoothing to the
neuroimaging data, useful to fight noise or for inter-individual
differences in neuroanatomy. It is achieved by specifying the full-width
half maximum (FWHM; in millimeter scale) with the `smoothing_fwhm`
parameter. Anisotropic filtering is also possible by passing 3 scalars
``(x, y, z)``, the FWHM along the x, y, and z direction.

The underlying function handles properly non-cubic voxels by scaling the
given widths appropriately.

.. seealso::

   :func:`nilearn.image.smooth_img`

.. _temporal_filtering:

Temporal Filtering and confound removal
........................................

:class:`NiftiMasker` can also improve aspects of temporal data
properties, before conversion to voxel signals.

- **Standardization**. Parameter ``standardize``: Signals can be
  standardized (scaled to unit variance). 

- **Frequency filtering**. Low-pass and high-pass filters can be used to
  remove artifacts. Parameters: ``high_pass`` and ``low_pass``, specified
  in Hz (note that you must specific the sampling rate in seconds with
  the ``t_r`` parameter: ``loss_pass=.5, t_r=2.1``).

- **Confound removal**. Two ways of removing confounds are provided: simple
  detrending or using prespecified confounds, such as behavioral or movement 
  information.

  * Linear trends can be removed by activating the `detrend` parameter.
    This accounts for slow (as opposed to abrupt or transient) changes
    in voxel values along a series of brain images that are unrelated to the
    signal of interest (e.g., the neural correlates of cognitive tasks).
    It is not activated by default in :class:`NiftiMasker` but is recommended
    in almost all scenarios.
    
  * More complex confounds, measured during the acquision, can be removed
    by passing them to :meth:`NiftiMasker.transform`. If the dataset
    provides a confounds file, just pass its path to the masker.

.. topic:: **Exercise**
   :class: green

   You can, more as a training than as an exercise, try to play with
   the parameters in
   :ref:`sphx_glr_auto_examples_plot_decoding_tutorial.py`.
   Try to enable detrending and run the script:
   does it have a big impact on the result?


.. seealso::

   :func:`nilearn.signal.clean`




Resampling: resizing and changing resolutions of images
.......................................................

:class:`NiftiMasker` and many similar classes enable resampling
(recasting of images into different resolutions and transformations of
brain voxel data). Two parameters control resampling:

* `target_affine` to resample (resize, rotate...) images in order to match
  the spatial configuration defined by the new affine (i.e., matrix
  transforming from voxel space into world space).

* Additionally, a `target_shape` can be used to resize images
  (i.e., cropping or padding with zeros) to match an expected data
  image dimensions (shape composed of x, y, and z).

How to combine these parameter to obtain the specific resampling desired
is explained in details in :ref:`resampling`.

.. seealso::

   :func:`nilearn.image.resample_img`, :func:`nilearn.image.resample_to_img`

.. _unmasking_step:

Inverse transform: unmasking data
---------------------------------

Once voxel signals have been processed, the result can be visualized as
images after unmasking (masked-reduced data transformed back into
the original whole-brain space). This step is present in almost all
the :ref:`examples <examples-index>` provided in nilearn. Below you will find
an excerpt of :ref:`the example performing Anova-SVM on the Haxby data
<sphx_glr_auto_examples_02_decoding_plot_haxby_anova_svm.py>`):

.. literalinclude:: ../../examples/02_decoding/plot_haxby_anova_svm.py
    :start-after: # Look at the SVC's discriminating weights
    :end-before: # Create the figure

|

.. topic:: **Examples to better understand the NiftiMasker**

   * :ref:`sphx_glr_auto_examples_04_manipulating_images_plot_nifti_simple.py`

   * :ref:`sphx_glr_auto_examples_04_manipulating_images_plot_mask_computation.py`

|

.. _region:

Extraction of signals from regions:\  :class:`NiftiLabelsMasker`, :class:`NiftiMapsMasker`
==========================================================================================

The purpose of :class:`NiftiLabelsMasker` and :class:`NiftiMapsMasker` is to
compute signals from regions containing many voxels. They make it easy to get
these signals once you have an atlas or a parcellation into brain regions.

Regions definition
------------------

Nilearn understands two different ways of defining regions, which are called
labels and maps, handled by :class:`NiftiLabelsMasker` and
:class:`NiftiMapsMasker`, respectively.

- labels: a single region is defined as the set of all the voxels that have a
  common label (e.g., anatomical brain region definitions as integers)
  in the region definition array. The set of
  regions is defined by a single 3D array, containing a voxel-wise
  dictionary of label numbers that denote what
  region a given voxel belongs to. This technique has a big advantage: the
  required memory load is independent of the number of regions, allowing
  for a large number of regions. On the other hand, there are
  several disadvantages: regions cannot spatially overlap
  and are represented in a binary present/nonpresent coding (no weighting).

- maps: a single region is defined as the set of all the voxels that have a
  non-zero weight. A set of regions is thus defined by a set of 3D images (or a
  single 4D image), one 3D image per region (as opposed to all regions in a
  single 3D image such as for labels, cf. above).
  While these defined weighted regions can exhibit spatial
  overlap (as opposed to labels), storage cost scales linearly with the
  number of regions. Handling a large number (e.g., thousands)
  of regions will prove difficult with this data transformation of
  whole-brain voxel data into weighted region-wise data.

.. note::

   These usage are illustrated in the section :ref:`functional_connectomes`.

:class:`NiftiLabelsMasker` Usage
--------------------------------

Usage of :class:`NiftiLabelsMasker` is similar to that of
:class:`NiftiMapsMasker`. The main difference is that it requires a labels image
instead of a set of maps as input.

The `background_label` keyword of :class:`NiftiLabelsMasker` deserves
some explanation. The voxels that correspond to the brain or a region
of interest in an fMRI image do not fill the entire image.
Consequently, in the labels image, there must be a label value that corresponds
to "outside" the brain (for which no signal should be extracted).
By default, this label is set to zero in nilearn (refered to as "background").
Should some non-zero value encoding be necessary, it is possible
to change the background value with the `background_label` keyword.

.. topic:: **Examples**

    * :ref:`sphx_glr_auto_examples_03_connectivity_plot_signal_extraction.py`

:class:`NiftiMapsMasker` Usage
------------------------------

This atlas defines its regions using maps. The path to the corresponding
file is given in the `maps_img` argument.

One important thing that happens transparently during the execution of
:meth:`NiftiMasker.fit_transform` is resampling. Initially, the images
and the atlas do typically not have the same shape nor the same affine.
Casting them into the same format is required for successful signal extraction
The keyword argument `resampling_target` specifies which format
(i.e., dimensions and affine) the data should be resampled to.
See the reference documentation for :class:`NiftiMapsMasker` for every
possible option.

.. topic:: **Examples**

   * :ref:`sphx_glr_auto_examples_03_connectivity_plot_probabilistic_atlas_extraction.py`

Extraction of signals from seeds:\  :class:`NiftiSpheresMasker`
===============================================================

The purpose of :class:`NiftiSpheresMasker` is to compute signals from
seeds containing voxels in spheres. It makes it easy to get these signals once
you have a list of coordinates.
A single seed is a sphere defined by the radius (in millimeters) and the
coordinates (typically MNI or TAL) of its center.

Using :class:`NiftiSpheresMasker` needs to define a list of coordinates.
`seeds` argument takes a list of 3D coordinates (tuples) of the spheres centers,
they should be in the same space as the images.
Seeds can overlap spatially and are represented in a binary present/nonpresent
coding (no weighting).
Below is an example of a coordinates list of four seeds from the default mode network::

  >>> dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (0, 50, -5)]

`radius` is an optional argument that takes a real value in millimeters.
If no value is given for the `radius` argument, the single voxel at the given
seed position is used.

.. topic:: **Examples**

  * :ref:`sphx_glr_auto_examples_03_connectivity_plot_adhd_spheres.py`
