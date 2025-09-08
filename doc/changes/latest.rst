.. currentmodule:: nilearn

.. include:: names.rst

0.13.0.dev
==========

HIGHLIGHTS
----------

.. warning::

 | **Support for Python 3.9 has been dropped.**
 | **We recommend upgrading to Python 3.12 or above.**
 |
 | **Minimum supported versions of the following packages have been bumped up:**
 | - matplotlib -- 3.8.0

NEW
---

Fixes
-----

Enhancements
------------

Changes
-------

- :bdg-danger:`Deprecation` Systematically use ``imgs`` instead of ``X`` or ``img`` in the methods ``fit()``, ``transform()`` and ``fit_transform()`` for the Nilearn maskers and their derivative classes (:gh:`5624` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The attribute ``nifti_maps_masker_`` was removed from :class:`~decomposition.CanICA` and :class:`~decomposition.DictLearning`. Use ``maps_masker_`` instead. (:gh:`5626` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``output_type`` of :func:`~mass_univariate.permuted_ols` was changed from ``legacy`` to ``dict``. The parameter ``output_type`` will be removed in Nilearn >= 0.15.0 (:gh:`5631` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``contrast_type`` of  :func:`~glm.compute_contrast` and :class:`~glm.Contrast` has been replaced by ``stat_type`` (:gh:`5630` by `Rémi Gau`_).
