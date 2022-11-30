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
- Fix display of colorbar in matrix plots. Colorbar was overlapping in :func:`~matplotlib.pyplot.subplots` due to a hardcoded adjustment value in the subplot (:gh:`3403` by `Raphael Meudec`_).

Enhancements
------------

- :func:`~signal.clean` imputes scrubbed volumes (defined through ``sample_masks``) with cubic spline function before applying butterworth filter (:gh:`3385` by `Hao-Ting Wang`_). 
- As part of making the User Guide more user-friendly, the introduction was reworked (:gh:`3380` by `Alexis Thual`_)

Changes
-------

- Private functions ``nilearn.regions.rena_clustering.weighted_connectivity_graph`` and ``nilearn.regions.rena_clustering.nearest_neighbor_grouping`` have been renamed with a leading "_", while function :func:`~regions.recursive_neighbor_agglomeration` has been added to the public API (:gh:`3347` by `Ahmad Chamma`_).
- Numpy deprecated type aliases are replaced by equivalent builtin types (:gh:`3422` by `Yasmin Mzayek`_).
- Function ``nilearn.masking.compute_multi_gray_matter_mask`` has been removed
  since it has been deprecated and replaced by :func:`~masking.compute_multi_brain_mask` (:gh:`3427` by `Yasmin Mzayek`_).
- :mod:`~nilearn.glm` will no longer warn that the module is experimental (:gh:`3424` by `Yasmin Mzayek`_).
