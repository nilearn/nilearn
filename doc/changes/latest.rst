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

Changes
-------

- Function :func:`~plotting.plot_carpet` argument ``cmap`` now respects behaviour specified by docs and changes the color of the carpet_plot. Changing the label colors is now delegated to a new variable ``cmap_labels`` (:gh:`3209` by `Daniel Gomez`_).
- Function :func:`~datasets.fetch_surf_fsaverage` no longer supports the previously deprecated option ``fsaverage5_sphere`` (:gh:`3229` by `Taylor Salo`_).
- Classes :class:`~glm.regression.RegressionResults`, :class:`~glm.regression.SimpleRegressionResults`,
  :class:`~glm.regression.OLSModel`, and :class:`~glm.model.LikelihoodModelResults` no longer support deprecated shortened attribute names,
  including ``df_resid``, ``wdesign``, ``wresid``, ``norm_resid``, ``resid``, and ``wY`` (:gh:`3229` by `Taylor Salo`_).
- Function :func:`~datasets.fetch_openneuro_dataset_index` is now deprecated in favor of the new :func:`~datasets.fetch_ds000030_urls` function (:gh:`3216` by `Taylor Salo`_).
- 64-bit integers in Nifti files: some tools such as FSL, SPM and AFNI cannot
  handle Nifti images containing 64-bit integers. To avoid compatibility issues,
  it is best to avoid writing such images and in the future trying to create
  them with `nibabel` without explicitly specifying a data type will result in
  an error. See details in this issue:
  https://github.com/nipy/nibabel/issues/1046 and this PR:
  https://github.com/nipy/nibabel/pull/1082. To avoid this,
  :func:`~image.new_img_like` now warns when given int64 arrays and converts
  them to int32 when possible (ie when it would not result in an overflow).
  Moreover, any atlas fetcher that returned int64 images now produces images
  containing smaller ints. (:gh:`3227` by `Jerome Dockes`_)
- Refactors fmriprep confound loading such that that the parsing of the
  relevant image file and the loading of the confounds are done in
  separate steps (:gh:`3274` by `David G Ellis`_).
- Private submodules, functions, and classes from the :mod:`~nilearn.decomposition` module now start with a "_" character to make it clear that they are not part of the public API (:gh:`3141` by `Nicolas Gensollen`_).
- Convert references in ``nilearn/glm/regression.py`` and ``nilearn/glm/thresholding.py`` to use footcite/footbibliography (:gh:`3302` by `Ahmad Chamma`_).
- Boolean input data in :func:`~image.new_img_like` now defaults to `np.uint8` instead of `np.int8` (:gh:`3286` by `Yasmin Mzayek`_).
- The current behavior of maskers' ``transform`` on 3D niimg inputs, in which a 2D array is returned, is deprecated, and 1D arrays will be returned starting in version ``0.12`` (:gh:`3322` by `Taylor Salo`_).
- Private functions ``nilearn.regions.rena_clustering.weighted_connectivity_graph`` and ``nilearn.regions.rena_clustering.nearest_neighbor_grouping`` have been renamed with a leading "_", while function :func:`~regions.rena_clustering.recursive_neighbor_agglomeration` has been added to the public API (:gh:`3347` by `Ahmad Chamma`_).
