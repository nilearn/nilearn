.. _harvard_oxford_atlas:

Harvard Oxford atlas
====================

Access
------
See :func:`nilearn.datasets.fetch_atlas_harvard_oxford`.

Notes
-----
Probabilistic atlases covering 48 cortical and 21 subcortical structural areas,
derived from structural data and segmentations kindly
provided by the Harvard Center for Morphometric Analysis.

T1-weighted images of 21 healthy male and 16 healthy female subjects (ages 18-50)
were individually segmented by the CMA using semi-automated tools developed in-house.
The T1-weighted images were affine-registered to MNI152 space using FLIRT (FSL),
and the transforms then applied to the individual labels.
Finally, these were combined across subjects to form population probability maps for each label.

See also :footcite:t:`Makris2006`, :footcite:t:`Desikan2006`,
:footcite:t:`Frazier2005`, :footcite:t:`Goldstein2007`.

Content
-------
.. nilearn_dataset_license:: harvard_oxford_atlas

References
----------
.. footbibliography::

License
-------
.. nilearn_dataset_license:: harvard_oxford_atlas
