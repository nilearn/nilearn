.. _data_manipulation:

===============================================================
fMRI data manipulation: input/output, masking, visualization...
===============================================================

.. _downloading_data:

Downloading example datasets
============================

.. currentmodule:: nisl.datasets

This tutorial package embeds tools to download and load datasets. They
can be imported from :mod:`nisl.datasets`::

    >>> from nisl import datasets
    >>> haxby_files = datasets.fetch_haxby_simple()
    >>> # The structures contains paths to haxby dataset files:
    >>> haxby_files.keys() # doctest: +SKIP
    ['data', 'session_target', 'mask', 'conditions_target']
    >>> import nibabel
    >>> haxby_data = nibabel.load(haxby_files.func)
    >>> haxby_data.get_data().shape # 1452 time points and a spatial size of 40x64x64
    (40, 64, 64, 1452)

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fetch_haxby
   fetch_haxby_simple
   fetch_nyu_rest
   fetch_adhd

The data are downloaded only once and stored locally, in one of the
following directories (in order of priority):

  * the folder specified by `data_dir` parameter in the fetching function
    if it is specified
  * the environment variable `NISL_DATA` if it exists
  * the `nisl_data` folder in the current directory
   
Note that you can copy that folder across computers to avoid
downloading the data twice.

Understanding MRI data 
=======================

Nifti or analyze files
-----------------------

.. topic:: NIfTI and Analyse file structures

    `NifTi <http://nifti.nimh.nih.gov/>`_ files (or Analyze files) are
    the standard way of sharing data in neuroimaging. We may be
    interested in the following three main components:

    :data: 
        raw scans bundled in a numpy array: `data = img.get_data()`
    :affine: 
        gives the correspondance between voxel index and spatial location: 
        `affine = img.get_affine()`
    :header: 
        informations about the data (slice duration...):
        `header = img.get_header()`


Neuroimaging data can be loaded simply thanks to nibabel_. Once the file is
downloaded, a single line is needed to load it.

.. literalinclude:: ../plot_visualization.py
     :start-after: # Fetch data ################################################################
     :end-before: # Visualization #############################################################

.. topic:: Dataset formatting: data shape

    We can find two main representations for MRI scans:

    - a big 4D matrix representing 3D MRI along time, stored in a big 4D
      NifTi file. 
      `FSL <http://www.fmrib.ox.ac.uk/fsl/>`_ users tend to 
      prefer this format.
    - several 3D matrices representing each volume (time point) of the 
      session, stored in set of 3D Nifti or analyse files. 
      `SPM <http://www.fil.ion.ucl.ac.uk/spm/>`_ users tend
      to prefer this format.

.. _niimg:

Niimg-like objects
-------------------

**Niimg:** Niimg (pronounce ni-image) is a common term used in Nisl. A
Niimg-like object can either be:

  * a file path to a Nifti or Analyse image
  * any object exposing ``get_data()`` and ``get_affine()`` methods, for
    instance a ``Nifti1Image`` from nibabel_.

**Niimg-4D:** Similarly, some functions require 4-dimensional Nifti-like
data, which we call Niimgs, or Niimg-4D. Accepted inputs are then:

  * A path to a 4-dimensional Nifti image
  * List of paths to 3-dimensional Nifti images
  * 4-dimensional Nifti-like object
  * List of 3-dimensional Nifti-like objects

.. note:: **Image affines**

   If you provide a sequence of Nifti images, all of them must have the same
   affine.


Visualizing brain images
========================

Once that NIfTI data are loaded, visualization is simply the display of the
desired slice (the first three dimensions) at a desired time point (fourth
dimension). For *haxby*, data is rotated so we have to turn each image
counter-clockwise.

.. literalinclude:: ../plot_visualization.py
     :start-after: # Visualization #############################################################
     :end-before: # Extracting a brain mask ###################################################

.. figure:: auto_examples/images/plot_visualization_1.png
    :target: auto_examples/plot_visualization.html
    :align: center
    :scale: 60


.. currentmodule:: nisl.io

.. _nifti_masker:

The :class:`NiftiMasker`: loading, masking and filtering
=========================================================

In this section gives some details and show how to custom data loading.
For this, we rely on the :class:`NiftiMasker` class. Advanced usage of
this class uses its parameters to tweak algorithms. Sometimes, you will
have to go beyond it and put your hands in the code to achieve what you
want.

The :class:`NiftiMasker` is a power tool to *1)* load data easily, *2)*
preprocess it and then *3)* send it directly into a scikit-learn pipeline. It
is designed to apply some basic preprocessing steps by default with commonly
used default parameters. But it is *very important* to look at your data to see
the effects of these preprocessings and validate them.

In addition, the :class:`NiftiMasker` is a scikit-learn compliant
transformer so that you can directly plug it into a scikit-learn
pipeline. This feature can be seen in :ref:`nifti_masker_advanced`.

Custom data loading
--------------------

Sometimes, you may want to preprocess data by yourself.
In this example, we will restrict Haxby dataset to 150 frames to speed up
computation. To do that, we load the dataset, restrain it to 150 frames and
build a brand new Nifti like object to give it to the Nifti masker. There
is no need to save your data in a file to pass it to Nifti masker.
Simply use nibabel_ to create a :ref:`Niimg <niimg>` in memory:


.. literalinclude:: ../plot_nifti_advanced.py
    :start-after: from nisl import datasets
    :end-before: # Display helper

.. _masking:

Custom Masking
---------------

In the basic tutorial, we showed how the masker could compute a mask
automatically: the result was quite impressive. But, on some datasets, the
default algorithm performs poorly. That is why it is very important to
*always look at how your data look like*.

Mask Visualization
...................

Before exploring the subject, we define a helper function to display the
masks. This function will display a background (composed of a mean of epi scans)
and the mask as a red layer over this background.


.. literalinclude:: ../plot_nifti_advanced.py
    :start-after: haxby_img = nibabel.Nifti1Image(haxby_func, haxby_img.get_affine()) 
    :end-before: # Generate mask with default parameters 


Computing the mask
...................

.. currentmodule:: nisl.io

If a mask is not given, the :class:`NiftiMasker` will try to compute
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

The first the step of the generation is to generate a mask with default
parameters and take a look at it. As an indicator, we can, for example, compare
the mask to original data.

.. literalinclude:: ../plot_nifti_advanced.py
    :start-after: # Generate mask with default parameters
    :end-before: # Generate mask with opening

With naked eyes, we can see that the outline of the mask is not very
smooth. To make it smoother, we try to apply opening
(*mask_opening=true*).

.. figure:: auto_examples/images/plot_nifti_advanced_2.png
    :target: auto_examples/plot_nifti_advanced.html
    :align: right
    :scale: 50%

.. literalinclude:: ../plot_nifti_advanced.py
    :start-after: # Generate mask with opening 
    :end-before: # Generate mask with upper cutoff

If we look at the :class:`NiftiMasker` object, we
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

The resulting mask seems to be correct. If we compare it to the
original one, they are very close.


Preprocessings
--------------

.. _resampling:

Resampling
...........

:class:`NiftiMasker` offers two ways to resample images:

  * *target_affine*: resample (resize, rotate...) images by providing a new affine
  * *target_shape*: resize images by directly providing a new shape

Resampling can be used for example to reduce processing time of an algorithm by
lowering image resolution.

.. _temporal_filtering:

Temporal Filtering
...................

All previous filters concern spatial filtering. On the time axis, the Nifti
masker also proposes some filters.

By default, the signal will be normalized. If the dataset provides a confounds
file, it can be applied by providing the path to the file to the masker.
Low-pass and high-pass filters allow one to remove artifacts.

Detrending removes any linear trend along specified axis from data. It
is not activated by default in the Nifti Masker but is almost
essential.

.. note:: **Exercise**
   :class: green

   You can, more as a training than as an exercise, try to play with the
   parameters in Nisl examples. Try to enable detrending in haxby decoding
   and run it: does it have a big impact on the results ?


Inverse transform: unmasking data
----------------------------------

Once your computation is finished, you want to unmask your data to be able
to visualize it. This step is present in almost all the examples provided in
Nisl.


.. literalinclude:: ../plot_haxby_decoding.py
    :start-after: svc = feature_selection.inverse_transform(svc)
    :end-before: # We use a masked array so that the voxels at '-1' are displayed

Masking the data manually
==========================

Extracting a brain mask
------------------------

If we do not have a mask of the relevant regions available, a brain mask
can be easily extracted from the fMRI data using the
:func:`nisl.masking.compute_epi_mask` function:

.. currentmodule:: nisl.masking

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_epi_mask

.. literalinclude:: ../plot_visualization.py
     :start-after: # Extracting a brain mask ###################################################
     :end-before: # Applying the mask #########################################################

.. figure:: auto_examples/images/plot_visualization_2.png
    :target: auto_examples/plot_visualization.html
    :align: center
    :scale: 50

.. _mask_4d_2_3d:

From 4D to 2D arrays
--------------------

fMRI data is usually represented as a 4D block of data: 3 spatial
dimensions and one of time. In practice, we are most often only
interested in working only on the time-series of the voxels in the
brain. It is thus convenient to apply a brain mask and go from a 4D
array to a 2D array, `voxel` **x** `time`, as depicted below:

.. only:: html

    .. image:: images/masking.jpg
        :align: center
        :width: 100%

.. only:: latex

    .. image:: images/masking.jpg
        :align: center

.. literalinclude:: ../plot_visualization.py
     :start-after: # Applying the mask #########################################################


.. figure:: auto_examples/images/plot_visualization_3.png
    :target: auto_examples/plot_visualization.html
    :align: center
    :scale: 50

.. _preprocessing_functions:

Preprocessing functions
========================

.. currentmodule:: nisl.io.nifti_masker

The :class:`NiftiMasker` automatically calls some preprocessing
functions that are available if you want to set up your own
preprocessing procedure:

.. currentmodule:: nisl

* Resampling: :func:`nisl.resampling.resample_img`
* Masking:

  * compute: :func:`nisl.masking.compute_epi_mask`
  * compute for multiple sessions/subjects: :func:`nisl.masking.compute_multi_epi_mask`
  * apply: :func:`nisl.masking.apply_mask`
  * intersect several masks (useful for multi sessions/subjects): :func:`nisl.masking.intersect_masks`
  * unmasking: :func:`nisl.masking.unmask`

* Cleaning signals: :func:`nisl.signals.clean`

.. _nibabel: http://nipy.sourceforge.net/nibabel/
