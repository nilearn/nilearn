.. _craddock_2012_atlas:

Craddock 2012 atlas
===================

Access
------
See :func:`nilearn.datasets.fetch_atlas_craddock_2012`.

Notes
-----
Atlas from (:footcite:t:`Craddock2012`).

Collection of regions of interest (ROI) that have been generated from applying
spatially constrained clustering on :term:`resting-state` data.

Several clustering statistics are used to compare methodological trade-offs
as well as to determine an adequate number of clusters.
The proposed functional and random parcellations perform equivalently for most of the metrics evaluated.
The online release also contains the scripts to derive these ROI atlases
by using spatially constrained Ncut spectral clustering.

See :footcite:t:`Craddock2012` and :footcite:t:`nitrcClusterROI`
for more information on this :term:`parcellation`.

For more information on this dataset's structure,
see https://www.nitrc.org/projects/cluster_roi/

Content
-------
.. nilearn_dataset_content:: craddock_2012_atlas

References
----------
.. footbibliography::

License
-------
.. nilearn_dataset_license:: craddock_2012_atlas
