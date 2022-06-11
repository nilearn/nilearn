.. currentmodule:: nilearn

.. include:: names.rst

0.9.2.dev
=========

NEW
---

- :func:`~interfaces.bids.save_glm_to_bids` has been added, which writes model outputs to disk according to BIDS convention (:gh:`2715` by `Taylor Salo`_).
- :func:`~mass_univariate.permuted_ols` and :func:`~glm.second_level.non_parametric_inference` now support :term:`TFCE` statistic (:gh:`3196` by `Taylor Salo`_).
- :func:`~mass_univariate.permuted_ols` and :func:`~glm.second_level.non_parametric_inference` now support cluster-level Family-wise error correction (:gh:`3181` by `Taylor Salo`_).

Fixes
-----

- Convert references in ``nilearn/mass_univariate/permuted_least_squares.py`` to use bibtex format (:gh:`3222` by `Yasmin Mzayek`_).
- :func:`~plotting.plot_roi` failed before when used with the "contours" view type and passing a list of cut coordinates in display mode "x", "y" or "z"; this has been corrected (:gh:`3241` by `Jerome Dockes`_).
- :func:`~plotting.plot_markers` can now plot a single scatter point (:gh:`3255` by `Caglar Cakan`_).
- Fix title display for :func:`~plotting.plot_surf_stat_map`. The ``title`` argument does not set the figure title anymore but the axis title. (:gh:`3220` by `Raphael Meudec`).
- :func:`~surface.load_surf_mesh` loaded FreeSurfer specific surface files (e.g. `.pial`) with a shift in the coordinates. This is fixed by adding the c_ras coordinates to the mesh coordinates (:gh:`3235` by `Yasmin Mzayek`_).

Enhancements
------------

- Add `sample_masks` to :meth:`~glm.first_level.FirstLevelModel.fit` for censoring time points (:gh:`3193` by `Hao-Ting Wang`_).
- Function :func:`~glm.first_level.run_glm` and class :class:`~glm.first_level.FirstLevelModel` now accept a ``random_state`` parameter, which allows users to seed the ``KMeans`` cluster model used to estimate AR coefficients. (:gh:`3185` by `Sami Jawhar`_).

Changes
-------

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
