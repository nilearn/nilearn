.. currentmodule:: nilearn

.. include:: names.rst

0.9.3.dev
=========

NEW
---

Fixes
-----

- Change calculation of TR in :func:`~.glm.first_level.compute_regressor` to be more precise (:gh:`3362` by `Anne-Sophie Kieslinger`_)

Enhancements
------------

Changes
-------

- Private functions ``nilearn.regions.rena_clustering.weighted_connectivity_graph`` and ``nilearn.regions.rena_clustering.nearest_neighbor_grouping`` have been renamed with a leading "_", while function :func:`~regions.recursive_neighbor_agglomeration` has been added to the public API (:gh:`3347` by `Ahmad Chamma`_).
