.. _schaefer_2018_atlas:

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

Release v0.14.3 of the Schaefer 2018 parcellation is used by
default. Versions prior to v0.14.3 are known to contain erroneous region
label names. For more details, see
https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/Updates/Update_20190916_README.md


Content
-------
.. nilearn_dataset_content:: schaefer_2018_atlas


References
----------
.. footbibliography::

License
-------
.. nilearn_dataset_license:: schaefer_2018_atlas
