.. _data_manipulation:

=====================================================================
Manipulating brain volume: input/output, masking, ROIs, smoothing...
=====================================================================

This chapter introduces the data structure of brain images and tools to
manipulation these.


.. contents:: **Chapters contents**
    :local:
    :depth: 1



.. _loading_data:

Loading data
============

.. currentmodule:: nilearn.datasets

.. _datasets:

Fetching open datasets
----------------------

The nilearn package provides a dataset fetching utility that
automatically downloads reference
datasets and atlases. Dataset fetching functions can be imported from
:mod:`nilearn.datasets`::

    >>> from nilearn import datasets
    >>> haxby_dataset = datasets.fetch_haxby(n_subjects=1)  # doctest: +SKIP

They return a data structure that contains different pieces of
information on the retrieved dataset, including the
file names on hard disk::

    >>> # The different files
    >>> print(sorted(list(haxby_dataset.keys())))  # doctest: +SKIP
    ['anat', 'description', 'func', 'mask', 'mask_face', 'mask_face_little',
    'mask_house', 'mask_house_little', 'mask_vt', 'session_target']
    >>> # Path to first functional file
    >>> print(haxby_dataset.func[0])  # doctest: +SKIP
    /.../nilearn_data/haxby2001/subj1/bold.nii.gz

Explanation and further resources of the dataset at hand can be retrieved as
follows::

    >>> print(haxby_dataset.description)  # doctest: +SKIP
    Haxby 2001 results


    Notes
    -----
    Results from a classical fMRI study that...

|


For a list of all the data fetching functions in nilearn, see :ref:`datasets_ref`.

Besides convenient downloading of openly accessible reference datasets
including important meta-data (e.g., stimulus characteristics and
participant information for confound removal), the fetching functions
perform data downloads only once and return the locally saved data upon
any later function calls.
The locally stored data can be found in one of the
following directories (in order of priority, if present):

  * default system paths used by third party software that may already
    provide the data (e.g., the Harvard-Oxford atlas
    is provided by the FSL software suite)
  * the folder specified by `data_dir` parameter in the fetching function
  * the global environment variable `NILEARN_SHARED_DATA`
  * the user environment variable `NILEARN_DATA`
  * the `nilearn_data` folder in the user home folder

Two different environment variables are provided to distinguish a global dataset
repository that may be read-only at the user-level.
Note that you can copy that folder to another user's computers to avoid
the initial dataset download on the first fetching call.


Loading your own data
---------------------

Using your own data images in nilearn is as simple as creating a list of
file name strings ::

    # dataset folder contains subject1.nii and subject2.nii
    my_data = ['dataset/subject1.nii', 'dataset/subject2.nii']

Nilearn also provides a "wildcard" pattern to list many files with one
expression:

::

    >>> # dataset folder contains subject_01.nii to subject_03.nii
    >>> # dataset/subject_*.nii is a glob expression matching all filenames.
    >>> # Example with a smoothing process:
    >>> from nilearn.image import smooth_img
    >>> result_img = smooth_img("dataset/subject_*") # doctest: +SKIP

.. topic:: **Python globbing**

    For more complicated use cases, Python also provides functions to work
    with file paths, in particular, :func:`glob.glob`.

    .. warning::

        Unlike nilearn's path expansion, the result of :func:`glob.glob` is
        not sorted and depending on the computer you are running they
        might not be in alphabetic order. We advise you to rely on
        nilearn's path expansion.

Understanding neuroimaging data
===============================

Nifti and Analyze files
-----------------------

.. topic:: **NIfTI and Analyze file structures**

    `NifTi <http://nifti.nimh.nih.gov/>`_ files (or Analyze files) are
    the standard way of sharing data in neuroimaging research.
    Three main components are:

     :data:
         raw scans in form of a numpy array: ``data = img.get_data()``
     :affine:
         returns the transformation matrix that maps
         from voxel indices of the numpy array to actual real-world
         locations of the brain:
         ``affine = img.get_affine()``
     :header:
         low-level informations about the data (slice duration, etc.):
         ``header = img.get_header()``


Neuroimaging data can be loaded in a simple way thanks to nibabel_.
A Nifti file on disk can be loaded with a single line.

.. literalinclude:: ../../examples/01_plotting/plot_visualization.py
     :start-after: # Fetch data
     :end-before: # Visualization

.. topic:: **Dataset formatting: data shape**

    It is important to appreciate two main representations for
    storing and accessing more than one Nifti images, that is sets
    of MRI scans:

    - a big 4D matrix representing (3D MRI + 1D for time), stored in a single
      Nifti file.
      `FSL <http://www.fmrib.ox.ac.uk/fsl/>`_ users tend to
      prefer this format.
    - several 3D matrices representing each time point (single 3D volume) of the
      session, stored in set of 3D Nifti or analyse files.
      `SPM <http://www.fil.ion.ucl.ac.uk/spm/>`_ users tend
      to prefer this format.

.. _niimg:

Niimg-like objects
-------------------

As a baseline, nilearn functions take as input argument what we call
"Niimg-like objects":

**Niimg:** A Niimg-like object can be one of the following:

  * A string variable with a file path to a Nifti or Analyse image
  * Any object exposing ``get_data()`` and ``get_affine()`` methods, typically
    a ``Nifti1Image`` from nibabel_.

**Niimg-4D:** Similarly, some functions require 4D Nifti-like
data, which we call Niimgs or Niimg-4D. Accepted input arguments are:

  * A path to a 4D Nifti image
  * List of paths to 3D Nifti images
  * 4D Nifti-like object
  * List of 3D Nifti-like objects

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
    >>> haxby_dataset = datasets.fetch_haxby(n_subjects=1)  # doctest: +SKIP
    >>> import numpy as np
    >>> labels = np.recfromcsv(haxby_dataset.session_target[0], delimiter=" ")  # doctest: +SKIP
    >>> stimuli = labels['labels']  # doctest: +SKIP
    >>> print(np.unique(stimuli))  # doctest: +SKIP
    ['bottle' 'cat' 'chair' 'face' 'house' 'rest' 'scissors' 'scrambledpix'
     'shoe']

|

Masking data manually
=====================

Extracting a brain mask
------------------------

If we do not have a spatial mask of the target regions, a brain mask
can be easily extracted from the fMRI data by the
:func:`nilearn.masking.compute_epi_mask` function:

.. figure:: ../auto_examples/01_plotting/images/sphx_glr_plot_visualization_002.png
    :target: ../auto_examples/01_plotting/plot_visualization.html
    :align: right
    :scale: 50%

.. literalinclude:: ../../examples/01_plotting/plot_visualization.py
     :start-after: # Extracting a brain mask
     :end-before: # Applying the mask to extract the corresponding time series


.. _mask_4d_2_3d:

From 4D Nifti images to 2D data arrays
--------------------------------------

fMRI data is usually represented as a 4D block of data: 3 spatial
dimensions and one time dimension. In practice, we are usually
interested in working on the voxel time-series in the
brain. It is thus convenient to apply a brain mask in order to convert the
4D brain images representation into a restructured 2D data representation,
`voxel` **x** `time`, as depicted below:

.. image:: ../images/masking.jpg
    :align: center
    :width: 100%


.. literalinclude:: ../../examples/01_plotting/plot_visualization.py
     :start-after: # Applying the mask to extract the corresponding time series
     :end-before: # Find voxels of interest

.. figure:: ../auto_examples/01_plotting/images/sphx_glr_plot_visualization_003.png
    :target: ../auto_examples/01_plotting/plot_visualization.html
    :align: center
    :scale: 50

.. _preprocessing_functions:

Functions for data preparation steps
=====================================

.. currentmodule:: nilearn.input_data

The :class:`NiftiMasker` can automatically perform important data preparation
steps. These steps are also available as independent functions if you want to
set up your own data preparation procedure:

.. currentmodule:: nilearn

* Resampling: :func:`nilearn.image.resample_img`. See the example
  :ref:`sphx_glr_auto_examples_04_manipulating_images_plot_affine_transformation.py` to
  see the effect of affine transforms on data and bounding boxes.
* Computing the mean of images (along the time/4th dimension):
  :func:`nilearn.image.mean_img`
* Swapping voxels of both hemisphere (e.g., useful to homogenize masks
  inter-hemispherically):
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

* Cleaning signals (e.g., linear detrending, standardization,
  confound removal, low/high pass filtering): :func:`nilearn.signal.clean`


Image operations: creating a ROI mask manually
===============================================

This section shows manual steps to create and further modify a ROI
(region of interest) spatial mask. They represent a means for "data folding",
that is, extracting and later analyzing data from a subset of voxels rather
than the entire brain images. As a convenient side effect, this can help
alleviate the curse of dimensionality (i.e., statistical problems that
arise in the context of high-dimensional input variables).

Smoothing
---------

Functional MRI data have a low signal-to-noise ratio (yet much better
than EEG or MEG measurements).
When using simple methods
that are not robust to noise, it is useful to apply a spatial filtering
kernel on the data. Such data smoothing is
usually applied using a Gaussian function with 4mm to 12mm full-width at
half-maximum (this is where the FWHM comes from).
The function :func:`nilearn.image.smooth_img` accounts for potential
anisotropy in the image affine (i.e., non-identical voxel size in all
the three dimensions). Analogous to the majority of nilearn functions,
it can also use file names as input parameters.


.. literalinclude:: ../../examples/04_manipulating_images/plot_roi_extraction.py
    :start-after: # Smooth the data
    :end-before: # Run a T-test for face and houses

.. figure:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_roi_extraction_001.png
    :target: ../auto_examples/04_manipulating_images/plot_roi_extraction.html
    :align: center
    :scale: 50%

Selecting features
------------------

Functional MRI data can be considered "high dimensional" given the
p-versus-n ratio (e.g., p=~50,000-200,000 voxels for n=1000 samples).
In this setting, machine-learning
algorithms can perform poorly (i.e., curse-of-dimensionality problem).
However, simple means from the realms of classical statistics can help
reducing the number of voxels.

The Student's t-test (:func:`scipy.stats.ttest_ind`) is an established
method to determine whether two
distributions are statistically different. It can be used to compare voxel
time-series from two different experimental conditions
(e.g., when houses or faces are shown to individuals during brain scanning).
If the time-series distribution is similar in the two conditions, then the
voxel is not very interesting to discriminate the condition.

This test returns p-values that represent probabilities that the two
time-series had been drawn from the same distribution. The lower is the p-value, the
more discriminative is the voxel in distinguishing the two conditions.

.. literalinclude:: ../../examples/04_manipulating_images/plot_roi_extraction.py
    :start-after: # Run a T-test for face and houses
    :end-before: # Build a mask from this statistical map

.. figure:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_roi_extraction_002.png
    :target: ../auto_examples/04_manipulating_images/plot_roi_extraction.html
    :align: center
    :scale: 50%

This feature selection method is available in the scikit-learn Python
package, where it has been
extended to several classes, using the
:func:`sklearn.feature_selection.f_classif` function.

Thresholding
------------

Voxels with better p-values are kept as voxels of interest.
Applying a threshold to an array
is easy thanks to numpy indexing à la Matlab.

.. literalinclude:: ../../examples/04_manipulating_images/plot_roi_extraction.py
    :start-after: # Thresholding
    :end-before: # Binarization and intersection with VT mask

.. figure:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_roi_extraction_003.png
    :target: ../auto_examples/04_manipulating_images/plot_roi_extraction.html
    :align: center
    :scale: 50%

Mask intersection
-----------------

We now want to restrict our investigation to the ventral temporal area. The
corresponding spatial mask is provided in `haxby.mask_vt`.
We want to compute the
intersection of this provided mask with our self-computed mask.
The first step is to load it with
nibabel's **nibabel.load**. We can then use a logical "and" operation
-- **numpy.logical_and** -- to keep only voxels
that have been selected in both masks. In neuroimaging jargon, this is
called an "AND conjunction."

.. literalinclude:: ../../examples/04_manipulating_images/plot_roi_extraction.py
    :start-after: # Binarization and intersection with VT mask
    :end-before: # Dilation

.. figure:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_roi_extraction_004.png
    :target: ../auto_examples/04_manipulating_images/plot_roi_extraction.html
    :align: center
    :scale: 50%

Mask dilation
-------------

Tresholded functional brain images often contain scattered voxels
across the brain.
To consolidate such brain images towards more
compact shapes, we use a `morphological dilation <http://en.wikipedia.org/wiki/Dilation_(morphology)>`_. This is a common step to be sure
not to forget voxels located on the edge of a ROI.
Put differently, such operations can fill "holes" in masked voxel
representations.

.. literalinclude:: ../../examples/04_manipulating_images/plot_roi_extraction.py
    :start-after: # Dilation
    :end-before: # Identification of connected components

.. figure:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_roi_extraction_005.png
    :target: ../auto_examples/04_manipulating_images/plot_roi_extraction.html
    :align: center
    :scale: 50%

Extracting connected components
-------------------------------

The function **scipy.ndimage.label** from the scipy Python library
identifies immediately neighboring
voxels in our voxels mask. It assigns a separate integer label to each
one of them.

.. literalinclude:: ../../examples/04_manipulating_images/plot_roi_extraction.py
    :start-after: # Identification of connected components
    :end-before: # Use the new ROIs to extract data maps in both ROIs

.. figure:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_roi_extraction_006.png
    :target: ../auto_examples/04_manipulating_images/plot_roi_extraction.html
    :align: center
    :scale: 50%

Saving the result
-----------------

The final voxel mask is saved using nibabel for further inspection
with a software such as FSLView.

.. literalinclude:: ../../examples/04_manipulating_images/plot_roi_extraction.py
    :start-after: # save the ROI 'atlas' to a single output Nifti

.. _nibabel: http://nipy.sourceforge.net/nibabel/
