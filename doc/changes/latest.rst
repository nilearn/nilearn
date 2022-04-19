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

- Function :func:`~datasets.fetch_openneuro_dataset_index` is now deprecated in favor of the new :func:`~datasets.fetch_ds000030_urls` function (:gh:`3216` by `Taylor Salo`_).
