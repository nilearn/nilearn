.. _data_manipulation:

=================================================================
MRI data manipulation: input/output, masking, ROIs, smoothing...
=================================================================

This chapter presents the structure of brain image data and tools to
manipulation them.


.. contents:: **Chapters contents**
    :local:
    :depth: 1



.. _loading_data:

Loading data
============

.. currentmodule:: nilearn.datasets

.. _datasets:

Fetching datasets
-----------------

Nilearn package embeds a dataset fetching utility to download reference
datasets and atlases. Dataset fetching functions can be imported from
:mod:`nilearn.datasets`::

    >>> from nilearn import datasets
    >>> haxby_files = datasets.fetch_haxby(n_subjects=1)

They return a structure that contains the different file names::

    >>> # The different files
    >>> print haxby_files.keys()
    ['mask_house_little', 'anat', 'mask_house', 'mask_face', 'func', 'session_target', 'mask_vt', 'mask_face_little']
    >>> #  Path to first functional file
    >>> print haxby_files.func[0] # doctest: +ELLIPSIS
    /.../nilearn_data/haxby2001/subj1/bold.nii.gz

|

For a list of all the data fetching functions in nilearn, see :ref:`datasets_ref`.

The data are downloaded only once and stored locally, in one of the
following directories (in order of priority):

  * the folder specified by `data_dir` parameter in the fetching function
    if it is specified
  * the global environment variable `NILEARN_SHARED_DATA` if it exists
  * the user environment variable `NILEARN_DATA` if it exists
  * the `nilearn_data` folder in the user home folder

Two different environment variables are provided to distinguish a global dataset
repository that may be read-only from a user-level one.
Note that you can copy that folder across computers to avoid
downloading the data twice.


Loading your own data
----------------------

Using your own experiment in nilearn is as simple as declaring a list of
your files ::

    # dataset folder contains subject1.nii and subject2.nii
    my_data = ['dataset/subject1.nii', 'dataset/subject2.nii']

Python also provides helpers to work with filepaths. In particular,
:func:`glob.glob` is useful to
list many files with a "wild-card": \*.nii

.. warning::
   The result of :func:`glob.glob` is not sorted. For neuroimaging, you
   should always sort the output of glob using the :func:`sorted`
   function.

::

   >>> # dataset folder contains subject1.nii and subject2.nii
   >>> import glob
   >>> sorted(glob.glob('dataset/subject*.nii')) # doctest: +SKIP
   ['dataset/subject1.nii', 'dataset/subject2.nii']


Understanding Neuroimaging data
===============================

Nifti and Analyze files
------------------------

.. topic:: **NIfTI and Analyze file structures**

    `NifTi <http://nifti.nimh.nih.gov/>`_ files (or Analyze files) are
    the standard way of sharing data in neuroimaging. We may be
    interested in the following three main components:

     :data:
         raw scans bundled in a numpy array: ``data = img.get_data()``
     :affine:
         gives the correspondance between voxel index and spatial location:
         ``affine = img.get_affine()``
     :header:
         informations about the data (slice duration...):
         ``header = img.get_header()``


Neuroimaging data can be loaded simply thanks to nibabel_. Once the file is
downloaded, a single line is needed to load it.

.. literalinclude:: ../../examples/manipulating_visualizing/plot_visualization.py
     :start-after: ### Load an fMRI file #########################################################
     :end-before: ### Visualization #############################################################

.. topic:: **Dataset formatting: data shape**

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

Often, nilearn functions take as input parameters what we call
"Niimg-like objects:

**Niimg:** A Niimg-like object can either be:

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

Text files: phenotype or behavior
----------------------------------

Phenotypic or behavioral data are often provided as text or CSV
(Comma Separated Values) file. They
can be loaded with `numpy.genfromtxt` but you may have to specify some options
(typically `skip_header` ignores column titles if needed).

For the Haxby datasets, we can load the categories of the images
presented to the subject::

    >>> from nilearn import datasets
    >>> haxby_files = datasets.fetch_haxby(n_subjects=1)
    >>> import numpy as np
    >>> labels = np.genfromtxt(haxby_files.session_target[0], skip_header=1,
    ...                        usecols=[0], dtype=basestring)
    >>> print np.unique(labels)
    ['bottle' 'cat' 'chair' 'face' 'house' 'rest' 'scissors' 'scrambledpix'
     'shoe']

|

Masking data manually
=====================

Extracting a brain mask
------------------------

If we do not have a mask of the relevant regions available, a brain mask
can be easily extracted from the fMRI data using the
:func:`nilearn.masking.compute_epi_mask` function:

.. currentmodule:: nilearn.masking

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_epi_mask

.. figure:: ../auto_examples/manipulating_visualizing/images/plot_visualization_2.png
    :target: ../auto_examples/manipulating_visualizing/plot_visualization.html
    :align: right
    :scale: 50%

.. literalinclude:: ../../examples/manipulating_visualizing/plot_visualization.py
     :start-after: ### Extracting a brain mask ###################################################
     :end-before: ### Applying the mask #########################################################

.. _mask_4d_2_3d:

From 4D to 2D arrays
--------------------

fMRI data is usually represented as a 4D block of data: 3 spatial
dimensions and one of time. In practice, we are most often only
interested in working only on the time-series of the voxels in the
brain. It is thus convenient to apply a brain mask and go from a 4D
array to a 2D array, `voxel` **x** `time`, as depicted below:

.. image:: ../images/masking.jpg
    :align: center
    :width: 100%


.. literalinclude:: ../../examples/manipulating_visualizing/plot_visualization.py
     :start-after: ### Applying the mask #########################################################
     :end-before: ### Find voxels of interest ###################################################

.. figure:: ../auto_examples/manipulating_visualizing/images/plot_visualization_3.png
    :target: ../auto_examples/manipulating_visualizing/plot_visualization.html
    :align: center
    :scale: 50

.. _preprocessing_functions:

Functions for data preparation steps
=====================================

.. currentmodule:: nilearn.input_data

The :class:`NiftiMasker` automatically does some important data preparation
steps. These steps are also available as simple functions if you want to
set up your own data preparation procedure:

.. currentmodule:: nilearn

* Resampling: :func:`nilearn.image.resample_img`. See the example
  :ref:`example_manipulating_visualizing_plot_affine_transformation.py` to
  see the effect of affine transforms on data and bounding boxes.
* Computing the mean of images (in the time of 4th direction):
  :func:`nilearn.image.mean_img`
* Swapping voxels of both hemisphere:
  :func:`nilearn.image.swap_img_hemispheres`
* Smoothing: :func:`nilearn.image.smooth_img`
* Masking:

  * compute from EPI images: :func:`nilearn.masking.compute_epi_mask`
  * compute from images with a flat background:
    :func:`nilearn.masking.compute_background_mask`
  * compute for multiple sessions/subjects:
    :func:`nilearn.masking.compute_multi_epi_mask`
    :func:`nilearn.masking.compute_multi_background_mask`
  * apply: :func:`nilearn.masking.apply_mask`
  * intersect several masks (useful for multi sessions/subjects): :func:`nilearn.masking.intersect_masks`
  * unmasking: :func:`nilearn.masking.unmask`

* Cleaning signals: :func:`nilearn.signal.clean`


Image operations: creating a ROI mask manually
===============================================

This section shows manual steps to create and finally control an ROI
mask. They are a good example of using basic image manipulation on Nifti
images.

Smoothing
---------

Functional MRI data has a low signal-to-noise ratio. When using simple methods
that are not robust to noise, it is useful to smooth the data. Smoothing is
usually applied using a Gaussian function with 4mm to 8mm full-width at
half-maximum. The function :func:`nilearn.image.smooth_img` accounts for potential
anisotropy in the image affine. As many nilearn functions, it can also
use file names as input parameters.


.. literalinclude:: ../../examples/manipulating_visualizing/plot_roi_extraction.py
    :start-after: # Smooth the data
    :end-before: # Run a T-test for face and houses

.. figure:: ../auto_examples/manipulating_visualizing/images/plot_roi_extraction_1.png
    :target: ../auto_examples/manipulating_visualizing/plot_roi_extraction.html
    :align: center
    :scale: 50%

Selecting features
------------------

Functional MRI data are high dimensional compared to the number of samples
(usually 50000 voxels for 1000 samples). In this setting, machine learning
algorithm can perform poorly. However, a simple statistical test can help
reducing the number of voxels.

The Student's t-test (:func:`scipy.stats.ttest_ind`) performs a simple statistical test that determines if two
distributions are statistically different. It can be used to compare voxel
timeseries in two different conditions (when houses or faces are shown in our
case). If the timeserie distribution is similar in the two conditions, then the
voxel is not very interesting to discriminate the condition.

This test returns p-values that represents probabilities that the two
timeseries are drawn from the same distribution. The lower is the p-value, the
more discriminative is the voxel.

.. literalinclude:: ../../examples/manipulating_visualizing/plot_roi_extraction.py
    :start-after: # Run a T-test for face and houses
    :end-before: ### Build a mask ##############################################################

.. figure:: ../auto_examples/manipulating_visualizing/images/plot_roi_extraction_2.png
    :target: ../auto_examples/manipulating_visualizing/plot_roi_extraction.html
    :align: center
    :scale: 50%

This feature selection method is available in the scikit-learn where it has been
extended to several classes, using the
:func:`sklearn.feature_selection.f_classif` function.

Thresholding
------------

Higher p-values are kept as voxels of interest. Applying a threshold to an array
is easy thanks to numpy indexing a la Matlab.

.. literalinclude:: ../../examples/manipulating_visualizing/plot_roi_extraction.py
    :start-after: # Thresholding
    :end-before: # Binarization and intersection with VT mask

.. figure:: ../auto_examples/manipulating_visualizing/images/plot_roi_extraction_3.png
    :target: ../auto_examples/manipulating_visualizing/plot_roi_extraction.html
    :align: center
    :scale: 50%

Mask intersection
-----------------

We now want to restrict our study to the ventral temporal area. The
corresponding mask is provided in `haxby.mask_vt`. We want to compute the
intersection of this mask with our mask. The first step is to load it with
nibabel's :func:`nibabel.load`. We then use a logical "and"
-- :func:`numpy.logical_and` -- to keep only voxels
that are selected in both masks.

.. literalinclude:: ../../examples/manipulating_visualizing/plot_roi_extraction.py
    :start-after: # Binarization and intersection with VT mask
    :end-before: # Dilation

.. figure:: ../auto_examples/manipulating_visualizing/images/plot_roi_extraction_4.png
    :target: ../auto_examples/manipulating_visualizing/plot_roi_extraction.html
    :align: center
    :scale: 50%

Mask dilation
--------------

We observe that our voxels are a bit scattered across the brain. To obtain more
compact shape, we use a `morphological dilation <http://en.wikipedia.org/wiki/Dilation_(morphology)>`_. This is a common step to be sure
not to forget voxels located on the edge of a ROI.

.. literalinclude:: ../../examples/manipulating_visualizing/plot_roi_extraction.py
    :start-after: # Dilation
    :end-before: # Identification of connected components

.. figure:: ../auto_examples/manipulating_visualizing/images/plot_roi_extraction_5.png
    :target: ../auto_examples/manipulating_visualizing/plot_roi_extraction.html
    :align: center
    :scale: 50%

Extracting connected components
-------------------------------

Scipy function :func:`scipy.ndimage.label` identifies connected
components in our final mask: it assigns a separate integer label to each
one of them.

.. literalinclude:: ../../examples/manipulating_visualizing/plot_roi_extraction.py
    :start-after: # Identification of connected components
    :end-before: # use the new ROIs to extract data maps in both ROIs

.. figure:: ../auto_examples/manipulating_visualizing/images/plot_roi_extraction_6.png
    :target: ../auto_examples/manipulating_visualizing/plot_roi_extraction.html
    :align: center
    :scale: 50%

Saving the result
-----------------

The final result is saved using nibabel for further consultation with a software
like FSLview for example.

.. literalinclude:: ../../examples/manipulating_visualizing/plot_roi_extraction.py
    :start-after: # save the ROI 'atlas' to a single output nifti

.. _nibabel: http://nipy.sourceforge.net/nibabel/
