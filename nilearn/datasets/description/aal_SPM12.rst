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
http://www.gin.cnrs.fr/AAL-217?lang=en

License
-------
unknown
