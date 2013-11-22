.. _data_manipulation:

===============================================================
fMRI data manipulation: input/output, masking, visualization...
===============================================================

.. _downloading_data:

Downloading example datasets
============================

.. currentmodule:: nilearn.datasets

This tutorial package embeds tools to download and load datasets. They
can be imported from :mod:`nilearn.datasets`::

    >>> from nilearn import datasets
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
  * the environment variable `NILEARN_DATA` if it exists
  * the `nilearn_data` folder in the current directory
   
Note that you can copy that folder across computers to avoid
downloading the data twice.

Understanding MRI data 
=======================

Nifti and analyze files
------------------------

.. topic:: **NIfTI and Analyse file structures**

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

.. literalinclude:: ../../plot_visualization.py
     :start-after: # Fetch data ################################################################
     :end-before: # Visualization #############################################################

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

**Niimg:** Niimg (pronounce ni-image) is a common term used in Nilearn. A
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

.. _visualizing:

Visualizing brain images
========================

Once that NIfTI data are loaded, visualization is simply the display of the
desired slice (the first three dimensions) at a desired time point (fourth
dimension). For *haxby*, data is rotated so we have to turn each image
counter-clockwise.

.. literalinclude:: ../plot_visualization.py
     :start-after: # Visualization #############################################################
     :end-before: # Extracting a brain mask ###################################################

.. figure:: ../auto_examples/images/plot_visualization_1.png
    :target: ../auto_examples/plot_visualization.html
    :align: center
    :scale: 60


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

.. literalinclude:: ../plot_visualization.py
     :start-after: # Extracting a brain mask ###################################################
     :end-before: # Applying the mask #########################################################

.. figure:: ../auto_examples/images/plot_visualization_2.png
    :target: ../auto_examples/plot_visualization.html
    :align: center
    :scale: 50%

.. _mask_4d_2_3d:

From 4D to 2D arrays
--------------------

fMRI data is usually represented as a 4D block of data: 3 spatial
dimensions and one of time. In practice, we are most often only
interested in working only on the time-series of the voxels in the
brain. It is thus convenient to apply a brain mask and go from a 4D
array to a 2D array, `voxel` **x** `time`, as depicted below:

.. image:: images/masking.jpg
    :align: center
    :width: 100%


.. literalinclude:: ../plot_visualization.py
     :start-after: # Applying the mask #########################################################


.. figure:: ../auto_examples/images/plot_visualization_3.png
    :target: ../auto_examples/plot_visualization.html
    :align: center
    :scale: 50

.. _preprocessing_functions:

Preprocessing functions
========================

.. currentmodule:: nilearn.input_data.nifti_masker

The :class:`NiftiMasker` automatically calls some preprocessing
functions that are available if you want to set up your own
preprocessing procedure:

.. currentmodule:: nilearn

* Resampling: :func:`nilearn.image.resample_img`
* Masking:

  * compute: :func:`nilearn.masking.compute_epi_mask`
  * compute for multiple sessions/subjects: :func:`nilearn.masking.compute_multi_epi_mask`
  * apply: :func:`nilearn.masking.apply_mask`
  * intersect several masks (useful for multi sessions/subjects): :func:`nilearn.masking.intersect_masks`
  * unmasking: :func:`nilearn.masking.unmask`

* Cleaning signals: :func:`nilearn.signal.clean`

.. _nibabel: http://nipy.sourceforge.net/nibabel/
