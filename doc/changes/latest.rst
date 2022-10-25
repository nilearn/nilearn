.. currentmodule:: nilearn

.. include:: names.rst

0.9.3.dev
=========

NEW
---

- New classes :class:`~maskers.MultiNiftiLabelsMasker` and :class:`~maskers.MultiNiftiMapsMasker` create maskers to extract signals from a list of subjects with 4D images using parallelization (:gh:`3237` by `Yasmin Mzayek`_).

Fixes
-----

- Regressor names can now be invalid identifiers but will raise an error with :meth:`~glm.first_level.FirstLevelModel.compute_contrast` if combined to make an invalid expression (:gh:`3374` by `Yasmin Mzayek`_).
- Fix :func:`~plotting.plot_connectome` which was raising a ``ValueError`` when ``vmax < 0`` (:gh:`3390` by `Paul Bogdan`_).

Enhancements
------------

Changes
-------

- Private functions ``nilearn.regions.rena_clustering.weighted_connectivity_graph`` and ``nilearn.regions.rena_clustering.nearest_neighbor_grouping`` have been renamed with a leading "_", while function :func:`~regions.recursive_neighbor_agglomeration` has been added to the public API (:gh:`3347` by `Ahmad Chamma`_).
