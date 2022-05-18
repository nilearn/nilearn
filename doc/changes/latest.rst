.. currentmodule:: nilearn

.. include:: names.rst

0.9.2.dev
=========

NEW
---

Fixes
-----

- Convert references in ``nilearn/mass_univariate/permuted_least_squares.py`` to use bibtex format (:gh:`3222` by `Yasmin Mzayek`_).

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
