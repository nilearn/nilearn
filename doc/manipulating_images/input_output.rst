.. _extracting_data:

=====================================================
Input and output: neuroimaging data representation
=====================================================

.. contents:: **Contents**
    :local:
    :depth: 1

|

.. currentmodule:: nilearn.image

.. _loading_data:

Inputing data: file names or image objects
===========================================

File names and objects, 3D and 4D images
-----------------------------------------

All Nilearn functions accept file names as arguments::

    >>> from nilearn import image
    >>> smoothed_img = image.smooth_img('/home/user/t_map001.nii')  # doctest: +SKIP

Nilearn can operate on either file names or `NiftiImage objects
<http://nipy.org/nibabel/nibabel_images.html>`_. The later represent the
data loaded in memory. In the example above, the
function :func:`smooth_img` returns a Nifti1Image object, which can then
be readily passed to other nilearn functions.

In nilearn, we often use the term *"niimg"* as abbreviation that denotes
either a file name or a `NiftiImage object
<http://nipy.org/nibabel/nibabel_images.html>`_.

Niimgs can be 3D or 4D. A 4D niimg may for instance represent a time
series of 3D images. It can be **a list of file names**, if these contain
3D information::

    >>> # dataset folder contains subject1.nii and subject2.nii
    >>> from nilearn.image import smooth_img
    >>> result_img = smooth_img(['dataset/subject1.nii', 'dataset/subject2.nii']) # doctest: +SKIP

``result_img`` is a 4D in-memory image, containing the data of both
subjects.


.. _filename_matching:

File name matching: "globbing" and user path expansion
------------------------------------------------------

You can specify files with *wildcard* matching patterns (as in Unix
shell):

 * **Matching multiple files**: suppose the dataset folder contains
   subject_01.nii, subject_03.nii, and subject_03.nii;
   ``dataset/subject_*.nii`` is a glob expression matching all filenames::

    >>> # Example with a smoothing process:
    >>> from nilearn.image import smooth_img
    >>> result_img = smooth_img("dataset/subject_*.nii") # doctest: +SKIP

   Note that the resulting is a 4D image.

 * **Expanding the home directory** ``~`` is expanded to your home
   directory::

    >>> result_img = smooth_img("~/dataset/subject_01.nii") # doctest: +SKIP

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
:mod:`nilearn.datasets`::

    >>> from nilearn import datasets
    >>> haxby_dataset = datasets.fetch_haxby()  # doctest: +SKIP

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

The `NifTi <http://nifti.nimh.nih.gov/>`_ data structure (also used in
Analyze files) is the standard way of sharing data in neuroimaging
research. Three main components are:

:data:
    raw scans in form of a numpy array: ``data = img.get_data()``
:affine:
    returns the transformation matrix that maps
    from voxel indices of the numpy array to actual real-world
    locations of the brain:
    ``affine = img.affine``
:header:
    low-level informations about the data (slice duration, etc.):
    ``header = img.header``

If you need to load the data without using nilearn, read the nibabel_
documentation.

Note: For older versions of nibabel_, affine and header can be retrieved
with ``get_affine()`` and ``get_header()``.


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
  * An ``SpatialImage`` from nibabel, ie an object exposing ``get_data()``
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

Text files: phenotype or behavior
----------------------------------

Phenotypic or behavioral data are often provided as text or CSV
(Comma Separated Values) file. They
can be loaded with `pd.read_csv` but you may have to specify some options
(typically `sep` if fields aren't delimited with a comma).

For the Haxby datasets, we can load the categories of the images
presented to the subject::

    >>> from nilearn import datasets
    >>> haxby_dataset = datasets.fetch_haxby()  # doctest: +SKIP
    >>> import pandas as pd  # doctest: +SKIP
    >>> labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ")  # doctest: +SKIP
    >>> stimuli = labels['labels']  # doctest: +SKIP
    >>> print(stimuli.unique())  # doctest: +SKIP
    ['bottle' 'cat' 'chair' 'face' 'house' 'rest' 'scissors' 'scrambledpix'
     'shoe']

.. topic:: **Reading CSV with pandas**

    `Pandas <http://pandas.pydata.org/>`_ is a powerful package to read
    data from CSV files and manipulate them.

|

.. _nibabel: http://nipy.sourceforge.net/nibabel/
