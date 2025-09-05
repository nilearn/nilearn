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

- :bdg-danger:`Deprecation`  In the ``scrubbing`` strategy used by :func:`~interfaces.fmriprep.load_confounds`, the default value of ``fd_threshold`` was changed from ``0.2`` to ``0.5`` and that of ``std_dvars_threshold`` was changed from ``3.0`` to ``1.5`` to better match the values used by fmriprep (:gh:`5633` by `Rémi Gau`_).
