.. _aal_atlas:

AAL atlas
=========

Access
------
See :func:`nilearn.datasets.fetch_atlas_aal`.

Notes
-----
This atlas is the result of an automated anatomical :term:`parcellation`
of the spatially normalized single-subject high-resolution T1 volume
provided by the Montreal Neurological Institute (MNI)
(:footcite:t:`Collins1998`).

Using this :term:`parcellation` method, three procedures to perform the automated anatomical labeling
of functional studies are proposed:
(1) labeling of an extremum defined by a set of coordinates,
(2) percentage of voxels belonging to each of the AVOI intersected by a sphere centered by a set of coordinates, and
(3) percentage of voxels belonging to each of the AVOI intersected by an activated cluster.

For the AAL version SPM 5, 8, and 12, the map image (data.maps) contains 117 unique integer values
that define the parcellation.
However, these values are not consecutive integers from 0 to 116, as is usually the case in Nilearn.
Therefore, they should not be interpreted as indices for the list of label names.
In contrast, the total number of parcellations in AAL 3v2 is 167.
The 3v2 atlas contains 171 unique integer values that define the parcellation.
These values are consecutive integers from 0 to 170,
except for the anterior cingulate cortex (35, 36) and thalamus (81, 82), which are left empty in AAL 3v2.
In addition, the region IDs are provided as strings, so it is necessary to cast them to integers when indexing.

For example, with version SPM 5, 8 and 12, to get the name of the region
corresponding to the region ID 5021 in the image, you should do:

.. code-block:: python

    # This should print 'Lingual_L'
    data.labels[data.indices.index("5021")]

Conversely, to get the region ID corresponding to the label
"Precentral_L", you should do:

.. code-block:: python

    # This should print '2001'
    data.indices[data.labels.index("Precentral_L")]

For more information on this atlas,
see :footcite:t:`AAL_atlas`,
and :footcite:t:`Tzourio-Mazoyer2002`.

Content
-------
    :"regions": str. path to nifti file containing regions.
    :"labels": dict. labels dictionary with their region id as key and name as value

References
----------

.. footbibliography::

For more information on this dataset's structure, see
https://www.gin.cnrs.fr/en/tools/aal/

License
-------
unknown
