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
