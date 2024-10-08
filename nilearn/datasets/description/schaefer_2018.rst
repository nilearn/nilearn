.. _schaefer_atlas:

Schaefer 2018 atlas
===================

Access
------
See :func:`nilearn.datasets.fetch_atlas_schaefer_2018`.

Notes
-----
This atlas (:footcite:t:`schaefer_atlas`) provides a labeling of cortical voxels in the MNI152
space, see :footcite:t:`Schaefer2017`.
Each ROI is annotated with a network from the :term:`parcellation`
(7- or 17-network solution; see :footcite:t:`Yeo2011`).

Different versions of the atlas are available, varying in
- number of rois (100 to 1000),
- network annotation (7 or 17)
- spatial resolution of the atlas (1 or 2 mm)

Content
-------
    :'maps': 3D Nifti image, values are indices in the list of labels.
    :'labels': ROI labels including Yeo-network annotation.
    :'description': A short description of the atlas and some references.

References
----------

.. footbibliography::

License
-------
MIT
