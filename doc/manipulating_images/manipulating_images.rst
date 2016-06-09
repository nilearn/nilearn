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

.. seealso::

    For a list of all the data fetching functions in nilearn, see
    :ref:`datasets_ref`.

.. topic:: **nilearn_data: Where is the downloaded data stored?**

    The fetching functions download the reference datasets to the disk.
    They save it locally for future use, in one of the
    following directories (in order of priority, if present):

     * the folder specified by `data_dir` parameter in the fetching function
     * the global environment variable `NILEARN_SHARED_DATA`
     * the user environment variable `NILEARN_DATA`
     * the `nilearn_data` folder in the user home folder

    The two different environment variables (NILEARN_SHARED_DATA and
    NILEARN_DATA) are provided for multi-user systems, to distinguish a
    global dataset repository that may be read-only at the user-level.
    Note that you can copy that folder to another user's computers to
    avoid the initial dataset download on the first fetching call.


Loading your own data: filenames or Nibabel objects
----------------------------------------------------

Function in nilearn can take filenames or nibabel in-memory object.

Using your own data images in nilearn is as simple as giving the file
name, or list of file name strings ::

    >>> # dataset folder contains subject1.nii and subject2.nii
    >>> from nilearn.image import smooth_img
    >>> result_img = smooth_img(['dataset/subject1.nii', 'dataset/subject2.nii']) # doctest: +SKIP

``result_img`` is a 4D in-memory image, containing the data of both
subjects.

**Filename matching** Nilearn also provides a "wildcard" pattern to list
many files with one expression:

 * **Matching multiple files**: suppose the dataset folder contains
   subject_01.nii to subject_03.nii ``dataset/subject_*.nii`` is a glob
   expression matching all filenames::

    >>> # Example with a smoothing process:
    >>> from nilearn.image import smooth_img
    >>> result_img = smooth_img("dataset/subject_*.nii") # doctest: +SKIP

 * **Expanding the home directory** ``~`` is expanded to your home
   directory::

    >>> result_img = smooth_img("~/dataset/subject_01.nii") # doctest: +SKIP

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

Nifti and Analyze data
-----------------------

For volumetric data, nilearn works with data stored as in the Nifti
structure (via the nibabel_ package).

The `NifTi <http://nifti.nimh.nih.gov/>`_ data structure (also used in
Analyze files) is the standard way of sharing data in neuroimaging
research. Three main components are:

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

If you need to load the data without using nilearn, read the nibabel_
documentation.


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

Nilearn functions take as input argument what we call "Niimg-like
objects":

**Niimg:** A Niimg-like object can be one of the following:

  * A string with a file path to a Nifti or Analyse image
  * Any object exposing ``get_data()`` and ``get_affine()`` methods, typically
    a ``Nifti1Image`` from nibabel_.

**Niimg-4D:** Similarly, some functions require 4D Nifti-like
data, which we call Niimgs or Niimg-4D. Accepted input arguments are:

  * A path to a 4D Nifti image
  * List of paths to 3D Nifti images
  * 4D Nifti-like object
  * List of 3D Nifti-like objects

.. topic:: **Image affines**

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

.. topic:: **Reading CSV with pandas**

    `Pandas <http://pandas.pydata.org/>`_ is a powerful package to read
    data from CSV files and manipulate them.

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

* Resampling: :func:`nilearn.image.resample_img` and :func:`nilearn.image.resample_to_img`.
  See the following examples:

  * affine transforms on data and bounding boxes: :ref:`sphx_glr_auto_examples_04_manipulating_images_plot_affine_transformation.py`,
  * resample an image to a template reference image: :ref:`sphx_glr_auto_examples_04_manipulating_images_plot_resample_to_template.py`.
 
* Computing the mean of images (along the time/4th dimension):
  :func:`nilearn.image.mean_img`
* Applying numpy functions on an image or a list of images:
  :func:`nilearn.image.math_img`
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

.. _resampling:

Resampling images
=================

Resampling one image to match another
-------------------------------------

:func:`nilearn.image.resample_to_img` resamples an image to a reference
image.

.. topic:: **Example**

    * :ref:`sphx_glr_auto_examples_04_manipulating_images_plot_resample_to_template.py`

.. image:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_resample_to_template_001.png
    :target: ../auto_examples/04_manipulating_images/plot_resample_to_template.html
    :scale: 55%

Resampling specific target affine, shape, or resolution
--------------------------------------------------------

The resampling procedure takes as input the
`target_affine` to resample (resize, rotate...) images in order to match
the spatial configuration defined by the new affine (i.e., matrix
transforming from voxel space into world space).

Additionally, a `target_shape` can be used to resize images
(i.e., cropping or padding with zeros) to match an expected data
image dimensions (shape composed of x, y, and z).

As a common use case, resampling can be a viable means to
downsample image quality on purpose to increase processing speed
and lower memory consumption of an analysis pipeline.
In fact, certain image viewers (e.g., FSLView) also require images to be
resampled to display overlays.

On an advanced note, automatic computation of offset and bounding box
can be performed by specifying a 3x3 matrix instead of the 4x4 affine.
In this case, nilearn computes automatically the translation part
of the transformation matrix (i.e., affine).

.. image:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_affine_transformation_002.png
    :target: ../auto_examples/04_manipulating_images/plot_affine_transformation.html
    :scale: 33%
.. image:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_affine_transformation_004.png
    :target: ../auto_examples/04_manipulating_images/plot_affine_transformation.html
    :scale: 33%
.. image:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_affine_transformation_003.png
    :target: ../auto_examples/04_manipulating_images/plot_affine_transformation.html
    :scale: 33%


.. topic:: **Special case: resampling to a given voxel size**

   Specifying a 3x3 matrix that is diagonal as a target_affine fixes the
   voxel size. For instance to resample to 3x3x3 mm voxels::

    >>> import numpy as np
    >>> target_affine = np.diag((3, 3, 3))


Image operations: creating a ROI mask manually
===============================================

Computing Regions of Interest (ROI) mask by ourselves requires a chain of image
operations to do from the input data to a mask over specific targets of
interest. The complete operations are listed below:

 * Fetching datasets: We use Haxby datasets and its experiments. The whole datasets
   can be fetched using a function :func:`nilearn.datasets.fetch_haxby`.
   See the documentation for more details about the Haxby datasets and its experiments.

 * Smoothing: Before building a statistical test, we do simple pre-processing step called
   image smoothing on functional images using function :func:`nilearn.image.smooth_img`
   with parameter given as fwhm=6.

 * Selecting features: Given the smoothed functional data, we select two features of
   interest with face and house experimental conditions. The method we use is a simple
   Student's t-test with scipy function :func:`scipy.stats.ttest_ind`.

 * Thresholding: Now, we threshold the statistical map to have better representation of
   voxels of interest.

 * Mask intersection and dilation: Post-processing the results with simple
   morphological operations, mask intersection and dilation. Here, we use the
   mask from the experiments and our results to select only those voxels which
   are common in both masks.

 * On the other hand, we again do `morphological dilation
   # <http://en.wikipedia.org/wiki/Dilation_(morphology)>`_ called Mask Dilation
   to more compact blobs. The function is used from
   :func:`scipy.ndimage.binary_dilation`.

 * Extracting connected components: We end with splitting the connected ROIs into two
   separate regions (ROIs), one in each hemisphere. The function **scipy.ndimage.label**
   from the scipy Python library is used in this setting.

 * Saving the result: The final voxel mask is saved using `nibabel.save` for further
   inspection with a software such as FSLView.

.. _nibabel: http://nipy.sourceforge.net/nibabel/

.. topic:: **Code**

    A complete script of above steps with full description can be found :ref:`here
    <sphx_glr_auto_examples_04_manipulating_images_plot_roi_extraction.py>`.

.. seealso::

     * :ref:`Automatic region extraction on 4D atlas images
       <sphx_glr_auto_examples_04_manipulating_images_plot_extract_rois_smith_atlas.py>`.
