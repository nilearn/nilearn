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

- :bdg-danger:`Deprecation` The attribute ``nifti_maps_masker_`` was removed from :class:`~decomposition.CanICA` and :class:`~decomposition.DictLearning`. Use ``maps_masker_`` instead. (:gh:`5626` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The default for the parameter ``homogeneity`` of :func:`~datasets.fetch_atlas_craddock_2012` was changed from ``None`` to ``'spatial'``. The only allowed values for this parameter must now be one of ``'spatial'``, ``'temporal'`` or ``'random'``. The fetcher now only returns a single map under the key ``maps`` for the requested ``homogeneity`` and ``grp_mean`` (:gh:`5640` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The default for the parameter ``dimension`` of :func:`~datasets.fetch_atlas_smith_2009` was changed from ``None`` to ``10``. The only allowed values for this parameter must now be one of ``10``, ``20`` or ``70``. The fetcher now only returns a single map under the key ``maps`` for the requested ``dimension`` and ``resting`` (:gh:`5640` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``legacy_output`` of :func:`~datasets.fetch_language_localizer_demo_dataset` has been removed. The fetcher now always returns a Scikit-Learn Bunch (:gh:`5640` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The function ``nilearn.datasets.fetch_bids_langloc_dataset`` has been removed. Use :func:`~datasets.fetch_language_localizer_demo_dataset` instead. (:gh:`5640` by `Rémi Gau`_).
