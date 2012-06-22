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

.. literalinclude:: ../../plot_visualization.py
     :start-after: # Haxby: Fetch data
     :end-before: # Haxby: Visualization


Visualization
`````````````
Once that NIfTI data is loaded, visualization is simply the display of the
desired slice. For haxby, data is rotated so we have to turn each image
counter clockwise.

.. literalinclude:: ../../plot_visualization.py
     :start-after: # Haxby: Visualization
     :end-before: ### Matlab: kamitani ##########################################################

.. figure:: ../auto_examples/images/plot_visualization_1.png
        :target: ../auto_examples/plot_visualization.html
        :align: center


Load a matlab dataset
---------------------

Specifications
``````````````
There is no standard format for Matlab data. Yet we fall back on some
standard-like formatting most of the time.

Loading
```````
Matlab files can be loaded in Python thanks to SciPy. After that, one has to
refer to the dataset documentation to know how data is formatted. Given that we
deal with non native Python data, some adaptations are needed. In the
particular case of Matlab structures, one can see that raw data is wrapped
several times, requiring usage of several flat/squeeze methods to fall back
on a more classical formatting.

.. literalinclude:: ../../plot_visualization.py
     :start-after: # Kamitani: Fetch data
     :end-before: # Kamitani: take the data of the first scan of the first session


Visualisation
`````````````
As said before, the way to visualize data depends on its formatting. In the
Kamitani dataset, scans are already masked and flattened so we have to go
back through this process to get a 3D representation, which is not trivial.

Three matrices are needed to reconstruct a full 3D scan:

- *data* contains flattened arrays for each scan
- *xyz* are MNI coordinates of a full scan voxels
- *volInd* makes the link between the index in the *data* array and
  the corresponding MNI coordinate in *xyz*

Thanks to *xyz* and *volInd*, we can build a map of MNI coordinates given the
index of the voxel in the *data* array. And then it is easy to build an
original 3D scan from the data.

.. literalinclude:: ../../plot_visualization.py
     :start-after: # Kamitani: take the data of the first scan of the first session


.. figure:: ../auto_examples/images/plot_visualization_2.png
        :target: ../auto_examples/plot_visualization.html
        :align: center

