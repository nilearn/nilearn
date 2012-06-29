.. _datasets:

========================================================
Basic dataset manipulation: loading and visualisation 
========================================================

Downloading the tutorial data
==============================

.. currentmodule:: nisl.datasets

This tutorial package embeds tools to download and load datasets. They
can be imported from ``nisl.datasets``::

    >>> from nisl import datasets
    >>> haxby_data = datasets.fetch_haxby()
    >>> # The data is then already loaded as numpy arrays:
    >>> haxby_data.keys() # doctest: +SKIP
    ['files', 'session', 'target', 'target_strings', 'data', 'affine', 'mask']
    >>> haxby_data.data.shape # 1452 time points and a spatial size of 40x64x64
    (40, 64, 64, 1452)

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fetch_haxby
   fetch_nyu_rest

The data are downloaded only once and stored locally in the `nisl_data`
folder.

Loading Nifti or analyze files
===============================

.. topic:: NIfTI and Analyse files

    `NifTi <http://nifti.nimh.nih.gov/>`_ files (or Analyze files) are 
    the standard way of sharing data in
    neuroimaging. We may be interested in the following
    three main components:

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

Visualizing brain images
============================

Once that NIfTI data is loaded, visualization is simply the display of the
desired slice (the first three dimensions) at a desired time point (fourth
dimension). For *haxby*, data is rotated so we have to turn each image
counter clockwise.

.. literalinclude:: ../plot_visualization.py
     :start-after: # Visualization #############################################################
     :end-before: # Extracting a brain mask ###################################################

.. figure:: auto_examples/images/plot_visualization_1.png
    :target: auto_examples/plot_visualization.html
    :align: center
    :scale: 60


Masking the data
=================

Extracting a brain mask
------------------------

If we do not have a mask of the relevant regions available, a brain mask
can be easily extracted from the fMRI data using the
:func:`nisl.masking.compute_mask` function:

.. currentmodule:: nisl.masking

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_mask

.. literalinclude:: ../plot_visualization.py
     :start-after: # Extracting a brain mask ###################################################
     :end-before: # Applying the mask #########################################################

.. figure:: auto_examples/images/plot_visualization_2.png
    :target: auto_examples/plot_visualization.html
    :align: center
    :scale: 50

.. _mask_4d_2_3d:

From 4D to 2D arrays
----------------------

FMRI data is naturally represented as a 4D block of data: 3 spatial
dimensions and time. In practice, we are most often only interested in
working only on the time-series of the voxels in the brain. It is
convenient to apply a brain mask and go from a 4D array to a 2D array,
`voxel` **x** `time`, as depicted below:

.. only:: html

    .. image:: masking.jpg
        :align: center
        :width: 100%

.. only:: latex

    .. image:: masking.jpg
        :align: center

.. literalinclude:: ../plot_visualization.py
     :start-after: # Applying the mask #########################################################


.. figure:: auto_examples/images/plot_visualization_3.png
    :target: auto_examples/plot_visualization.html
    :align: center
    :scale: 50

.. _nibabel: http://nipy.sourceforge.net/nibabel/
