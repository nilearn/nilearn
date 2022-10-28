.. currentmodule:: nilearn

.. include:: names.rst

0.9.3.dev
=========

NEW
---

Fixes
-----

Enhancements
------------

- :func:`~glm.first_level.first_level_from_bids` now takes an optional ``sub_labels`` argument and warns users of given subject labels that are not present in the dataset (:gh:`3351` by `Kevin Sitek`_).

Changes
-------

- Private functions ``nilearn.regions.rena_clustering.weighted_connectivity_graph`` and ``nilearn.regions.rena_clustering.nearest_neighbor_grouping`` have been renamed with a leading "_", while function :func:`~regions.recursive_neighbor_agglomeration` has been added to the public API (:gh:`3347` by `Ahmad Chamma`_).
