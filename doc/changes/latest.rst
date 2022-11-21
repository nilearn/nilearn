.. currentmodule:: nilearn

.. include:: names.rst

0.9.3.dev
=========

NEW
---

- New classes :class:`~maskers.MultiNiftiLabelsMasker` and :class:`~maskers.MultiNiftiMapsMasker` create maskers to extract signals from a list of subjects with 4D images using parallelization (:gh:`3237` by `Yasmin Mzayek`_).

Fixes
-----

- Fix off-by-one error when setting ticks in :func:`~plotting.plot_surf` (:gh:`3105` by `Dimitri Papadopoulos Orfanos`_).
- Regressor names can now be invalid identifiers but will raise an error with :meth:`~glm.first_level.FirstLevelModel.compute_contrast` if combined to make an invalid expression (:gh:`3374` by `Yasmin Mzayek`_).
- Fix :func:`~plotting.plot_connectome` which was raising a ``ValueError`` when ``vmax < 0`` (:gh:`3390` by `Paul Bogdan`_).
- Change the order of applying ``sample_masks`` in :func:`~signal.clean` based on different filtering options (:gh:`3385` by `Hao-Ting Wang`_). 
- When using cosine filter and ``sample_masks`` is used, :func:`~signal.clean` generates the cosine discrete regressors using the full time series (:gh:`3385` by `Hao-Ting Wang`_).
- Description of the ``opening`` parameter added to ``compute_multi_epi_mask`` docs (:gh:`3412` by `Natasha Clarke`_).

Enhancements
------------

- :func:`~signal.clean` imputes scrubbed volumes (defined through ``sample_masks``) with cubic spline function before applying butterworth filter (:gh:`3385` by `Hao-Ting Wang`_). 

Changes
-------

- Private functions ``nilearn.regions.rena_clustering.weighted_connectivity_graph`` and ``nilearn.regions.rena_clustering.nearest_neighbor_grouping`` have been renamed with a leading "_", while function :func:`~regions.recursive_neighbor_agglomeration` has been added to the public API (:gh:`3347` by `Ahmad Chamma`_).
