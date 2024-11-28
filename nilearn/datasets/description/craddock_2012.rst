.. _craddock_atlas:

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
as well as determine an adequate number of clusters. The proposed functional
and random parcellations perform equivalently for most of the metrics evaluated.
The online release also contains the scripts to derive these ROI atlases
by using spatially constrained Ncut spectral clustering.

See also :footcite:t:`nitrcClusterROI`

Content
-------
    :'random': result of random clustering for comparison
    :'scorr_2level': :term:`parcellation`  results when emphasizing spatial homogeneity
    :'scorr_mean': group-mean :term:`parcellation` results when emphasizing spatial homogeneity
    :'tcorr_2level': :term:`parcellation` results when emphasizing temporal homogeneity
    :'tcorr_mean': group-mean :term:`parcellation` results when emphasizing temporal homogeneity

References
----------

.. footbibliography::

For more information on this dataset's structure,
see https://www.nitrc.org/projects/cluster_roi/

License
-------
Creative Commons Attribution Non-commercial Share Alike.
See :footcite:t:`CreativeCommons` for more information.
