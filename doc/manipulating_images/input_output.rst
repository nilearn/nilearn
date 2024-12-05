.. _extracting_data:

=====================================================
Input and output: neuroimaging data representation
=====================================================

|

.. currentmodule:: nilearn.image

.. _loading_data:

Inputing data: file names or image objects
===========================================

File names and objects, 3D and 4D images
-----------------------------------------

All Nilearn functions accept file names as arguments:

.. code-block:: python

     from nilearn import image
     smoothed_img = image.smooth_img('/home/user/t_map001.nii')

Nilearn can operate on either file names or `NiftiImage objects
<https://nipy.org/nibabel/nibabel_images.html>`_. The later represent the
data loaded in memory. In the example above, the
function :func:`smooth_img` returns a Nifti1Image object, which can then
be readily passed to other nilearn functions.

In nilearn, we often use the term *"niimg"* as abbreviation that denotes
either a file name or a `NiftiImage object
<https://nipy.org/nibabel/nibabel_images.html>`_.

Niimgs can be 3D or 4D. A 4D niimg may for instance represent a time
series of 3D images. It can be **a list of file names**, if these contain
3D information:

.. code-block:: python

     # dataset folder contains subject1.nii and subject2.nii
     from nilearn.image import smooth_img
     result_img = smooth_img(['dataset/subject1.nii', 'dataset/subject2.nii'])

``result_img`` is a 4D in-memory image, containing the data of both
subjects.


.. _filename_matching:

File name matching: "globbing" and user path expansion
------------------------------------------------------

You can specify files with *wildcard* matching patterns (as in Unix
shell):

 * **Matching multiple files**: suppose the dataset folder contains
   subject_01.nii, subject_03.nii, and subject_03.nii;
   ``dataset/subject_*.nii`` is a glob expression matching all filenames:

 .. code-block:: python

    # Example with a smoothing process:
    from nilearn.image import smooth_img
    result_img = smooth_img("dataset/subject_*.nii")

 Note that the resulting is a 4D image.

 * **Expanding the home directory** ``~`` is expanded to your home
   directory:

 .. code-block:: python

    result_img = smooth_img("~/dataset/subject_01.nii")

 Using ``~`` rather than specifying the details of the path is good
 practice, as it will make it more likely that your script work on
 different computers.


.. topic:: **Python globbing**

    For more complicated use cases, Python also provides functions to work
    with file paths, in particular, :func:`glob.glob`.

    .. warning::

        Unlike nilearn's path expansion, the result of :func:`glob.glob` is
        not sorted and, depending on the computer you are running, they
        might not be in alphabetic order. We advise you to rely on
        nilearn's path expansion.

    To load data with globbing, we suggest that you use
    :func:`nilearn.image.load_img`.


.. currentmodule:: nilearn.datasets

.. _datasets:

Fetching open datasets from Internet
=====================================

Nilearn provides dataset fetching function that
automatically downloads reference
datasets and atlases. They can be imported from
:mod:`nilearn.datasets`:

.. code-block:: python

     from nilearn import datasets
     haxby_dataset = datasets.fetch_haxby()

They return a data structure that contains different pieces of
information on the retrieved dataset, including the
file names on hard disk:

.. code-block:: python

     # The different files
     print(sorted(list(haxby_dataset.keys())))
     # ['anat', 'description', 'func', 'mask', 'mask_face', 'mask_face_little',
     # 'mask_house', 'mask_house_little', 'mask_vt', 'session_target']
     # Path to first functional file
     print(haxby_dataset.func[0])
     # /.../nilearn_data/haxby2001/subj1/bold.nii.gz

Explanation and further resources of the dataset at hand can be retrieved as
follows:

.. code-block:: python

     print(haxby_dataset.description)
     # Haxby 2001 results


     # Notes
     # -----
     # Results from a classical fMRI study that...

|

.. seealso::

    For a list of all the data fetching functions in nilearn, see
    :ref:`datasets_ref`.

|

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

    You can check in which directory nilearn will store the data with the
    function :func:`nilearn.datasets.get_data_dirs`.


|

Understanding neuroimaging data
===============================

Nifti and Analyze data
-----------------------

For volumetric data, nilearn works with data stored as in the Nifti
structure (via the nibabel_ package).

The `NifTi <https://nifti.nimh.nih.gov/>`_ data structure (also used in
Analyze files) is the standard way of sharing data in neuroimaging
research. Three main components are:

:data:
    raw scans in form of a numpy array: ``data = nilearn.image.get_data(img)``
:affine:
    returns the transformation matrix that maps
    from :term:`voxel` indices of the numpy array to actual real-world
    locations of the brain:
    ``affine = img.affine``
:header:
    low-level information about the data (slice duration, etc.):
    ``header = img.header``

If you need to load the data without using nilearn, read the nibabel_
documentation.

Note: For older versions of nibabel_, affine and header can be retrieved
with ``get_affine()`` and ``get_header()``.

.. warning:: if you create images directly with nibabel_, beware of int64
             images. the default integer type used by Numpy is (signed) 64-bit.
             Several popular neuroimaging tools do not handle int64 Nifti
             images, so if you build Nifti images directly from Numpy arrays it
             is recommended to specify a smaller integer type, for example::

               np.array([1, 2000, 7], dtype="int32")



.. topic:: **Dataset formatting: data shape**

    It is important to appreciate two main representations for
    storing and accessing more than one Nifti images, that is sets
    of MRI scans:

    - a big 4D matrix representing (3D MRI + 1D for time), stored in a single
      Nifti file.
      `FSL <https://fsl.fmrib.ox.ac.uk/fsl/docs/>`_ users tend to
      prefer this format.
    - several 3D matrices representing each time point (single 3D volume) of the
      run, stored in set of 3D Nifti or analyze files.
      `SPM <https://www.fil.ion.ucl.ac.uk/spm/>`_ users tend
      to prefer this format.

.. _niimg:

Niimg-like objects
-------------------

Nilearn functions take as input argument what we call "Niimg-like
objects":

**Niimg:** A Niimg-like object can be one of the following:

  * A string or pathlib.Path object with a file path to a Nifti or Analyze image
  * An ``SpatialImage`` from nibabel, ie an object exposing ``get_fdata()``
    method and ``affine`` attribute, typically a ``Nifti1Image`` from nibabel_.

**Niimg-4D:** Similarly, some functions require 4D Nifti-like
data, which we call Niimgs or Niimg-4D. Accepted input arguments are:

  * A path to a 4D Nifti image
  * List of paths to 3D Nifti images
  * 4D Nifti-like object
  * List of 3D Nifti-like objects

.. topic:: **Image affines**

   If you provide a sequence of Nifti images, all of them must have the same
   affine.

.. topic:: **Decreasing memory used when loading Nifti images**

   When Nifti images are stored compressed (.nii.gz), loading them directly
   consumes more memory. As a result, large 4D images may
   raise "MemoryError", especially on smaller computers and when using Nilearn
   routines that require intensive 4D matrix operations. One step to improve
   the situation may be to decompress the data onto disk as an initial step.
   If multiple images are loaded into memory sequentially, another solution may
   be to `uncache <https://nipy.org/nibabel/images_and_memory.html#using-uncache>`_ one before loading and performing operations on another.

Text files: phenotype or behavior
----------------------------------

Phenotypic or behavioral data are often provided as text or CSV
(Comma Separated Values) file. They
can be loaded with ``pd.read_csv`` but you may have to specify some options
(typically ``sep`` if fields aren't delimited with a comma).

For the Haxby datasets, we can load the categories of the images
presented to the subject:

.. code-block:: python

     from nilearn import datasets
     haxby_dataset = datasets.fetch_haxby()
     import pandas as pd
     labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
     stimuli = labels['labels']
     print(stimuli.unique())
     # ['bottle' 'cat' 'chair' 'face' 'house' 'rest' 'scissors' 'scrambledpix' 'shoe']

.. topic:: **Reading CSV with pandas**

    `Pandas <https://pandas.pydata.org/>`_ is a powerful package to read
    data from CSV files and manipulate them.

|

.. _nibabel: https://nipy.org/nibabel/
