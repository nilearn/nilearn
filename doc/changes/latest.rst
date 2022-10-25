.. currentmodule:: nilearn

.. include:: names.rst

0.9.3.dev
=========

NEW
---

Fixes
-----

- Change the order of applying ``sample_masks`` in :func:`~.signal.clean` based on different filtering options (:gh:`3385` by `Hao-Ting Wang`). 
- When using cosine filter and ``sample_masks`` is used, :func:`~.signal.clean` generates the cosine descrete regressors using the full time seiries (:gh:`3385` by `Hao-Ting Wang`). 

Enhancements
------------

- :func:`nilearn.signal.clean` imputes scrubbed volumes (defined through ``sample_masks``) with cubic spline function before applying butterworth filter (:gh:`3385` by `Hao-Ting Wang`). 

Changes
-------

- Private functions ``nilearn.regions.rena_clustering.weighted_connectivity_graph`` and ``nilearn.regions.rena_clustering.nearest_neighbor_grouping`` have been renamed with a leading "_", while function :func:`~regions.recursive_neighbor_agglomeration` has been added to the public API (:gh:`3347` by `Ahmad Chamma`_).
