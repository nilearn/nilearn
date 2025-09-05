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

- :bdg-danger:`Deprecation` The parameter ``tr`` was replaced by ``t_r`` in :func:`~glm.first_level.glover_dispersion_derivative`, :func:`~glm.first_level.glover_hrf`, :func:`~glm.first_level.glover_time_derivative`, :func:`~glm.first_level.spm_dispersion_derivative`, :func:`~glm.first_level.spm_hrf` and :func:`~glm.first_level.spm_time_derivative`  (:gh:`5601` by `Rémi Gau`_).
