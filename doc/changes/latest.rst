.. currentmodule:: nilearn

.. include:: names.rst

0.9.2.dev
=========

NEW
---

- :func:`~mass_univariate.permuted_ols` and :func:`~glm.second_level.non_parametric_inference` now support cluster-level Family-wise error correction (:gh:`3181` by `Taylor Salo`_).

Fixes
-----

- Convert references in ``nilearn/mass_univariate/permuted_least_squares.py`` to use bibtex format (:gh:`3222` by `Yasmin Mzayek`_).
- :func:`~plotting.plot_roi` failed before when used with the "contours" view type and passing a list of cut coordinates in display mode "x", "y" or "z"; this has been corrected (:gh:`3241` by `Jerome Dockes`_).
- :func:`~plotting.plot_markers` can now plot a single scatter point (:gh:`3253` by `Caglar Cakan`_).

Enhancements
------------

- Add `sample_masks` to :meth:`~glm.first_level.FirstLevelModel.fit` for censoring time points (:gh:`3193` by `Hao-Ting Wang`_).

Changes
-------

- Function :func:`~datasets.fetch_surf_fsaverage` no longer supports the previously deprecated option ``fsaverage5_sphere`` (:gh:`3229` by `Taylor Salo`_).
- Classes :class:`~glm.regression.RegressionResults`, :class:`~glm.regression.SimpleRegressionResults`,
  :class:`~glm.regression.OLSModel`, and :class:`~glm.model.LikelihoodModelResults` no longer support deprecated shortened attribute names,
  including ``df_resid``, ``wdesign``, ``wresid``, ``norm_resid``, ``resid``, and ``wY`` (:gh:`3229` by `Taylor Salo`_).
- Function :func:`~datasets.fetch_openneuro_dataset_index` is now deprecated in favor of the new :func:`~datasets.fetch_ds000030_urls` function (:gh:`3216` by `Taylor Salo`_).
