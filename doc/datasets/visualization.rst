.. _datasets:

=================================
Dataset loading and visualisation 
=================================

.. currentmodule:: nisl.datasets

The ``nisl.datasets`` package embeds tools to fetch and load datasets. It comes
with a set of several datasets that are not always formatted the same way.

Dataset formatting
==================

Even though standard format exists, some people prefer to stick with custom
data formatting. That is why special processing is sometimes required.

We can find two main representations for MRI scans:

- a big 4D matrix representing 3D MRI along time
- several 3D matrices representing each slice of the session

These scans can be grouped by sessions and by subjects.

NIfTI : the haxby dataset
-------------------------

Specifications
``````````````
All information relative to this data format can be found on their official
website : http://nifti.nimh.nih.gov/nifti-1/

A NIfTI file contains three main components :
- *data*: raw scans bundled in a numpy array
- *affine*: allows to switch between voxel index and spatial location
- *header*: informations about the data (slice duration...)

Loading
```````
NIfTI data can be loaded simply thanks to *nibabel*. Once the file is
downloaded, a single line is needed to load it.

Visualization
`````````````
Once that NIfTI data is loaded, visualization is simply the display of the
desired slice. For haxby, data is rotated so we have to turn each image
counter clockwise.

Matlab : the kamitani dataset
-----------------------------

Specifications
``````````````
There is no standard format for Matlab data. Yet we fall back on some
standard-like formatting most of the time.

Loading
```````
Matlab files can be loaded in Python thanks to SciPy. After that, one has to
refer to the dataset documentation to know how data is formatted.

