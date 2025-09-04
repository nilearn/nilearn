.. currentmodule:: nilearn

.. include:: names.rst

0.13.0.dev
==========

NEW
---

Fixes
-----

Enhancements
------------

Changes
-------

- :bdg-danger:`Deprecation` Extra key-words arguments (``kwargs``) have been removed from the constructor of all the Nifti maskers. Any extra-parameters to pass to the call to :func:`~image.clean` done by ``transform`` must be done via the parameter ``clean_args`` (:gh:`5628` by `Rémi Gau`_).
