.. _datasets:

========================================================
Basic dataset manipulation: loading and visualisation 
========================================================

Dataset formatting
==================

Even though standard format exists, some people prefer to stick with custom
data formatting. That is why special processing is sometimes required.

We can find two main representations for MRI scans:

- a big 4D matrix representing 3D MRI along time
- several 3D matrices representing each volume (time point) of the session

These scans can be grouped by sessions and by subjects.

.. XXX: need to discuss masking

Some downloading utilities
===========================

.. currentmodule:: nisl.datasets

The ``nisl.datasets`` package embeds tools to fetch and load datasets. It comes
with a set of several datasets that are not always formatted the same way.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fetch_haxby
   fetch_nyu_rest
   fetch_star_plus

NIfTI and Analyse files
=========================

Specifications
---------------

NIfTI files (or Analyze files) are the standard way of sharing data in
neuroimaging. For our purposes, we may are interested in the following
three main components:

- *data*: raw scans bundled in a numpy array
- *affine*: allows to switch between voxel index and spatial location
- *header*: informations about the data (slice duration...)

Loading Nifti or analyze files
===============================

NIfTI data can be loaded simply thanks to nibabel_. Once the file is
downloaded, a single line is needed to load it.

.. literalinclude:: ../plot_visualization.py
     :start-after: # Fetch data ################################################################
     :end-before: # Visualization #############################################################


Visualizing brain images
============================

Once that NIfTI data is loaded, visualization is simply the display of the
desired slice. For haxby, data is rotated so we have to turn each image
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


From 4D to 2D arrays
----------------------

FMRI data is naturally represented as a 4D block of data: 3 spatial
dimensions and time. In practice, we are most often only interested in
working only on the time-series of the voxels in the brain. It is
convenient to apply a brain mask and go from a 4D array to a 2D array,
`voxel x time`, as depicted below:

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
