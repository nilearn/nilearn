.. currentmodule:: nilearn

.. include:: names.rst

0.10.2.dev
==========

NEW
---

Fixes
-----

Enhancements
------------

Changes
-------

- :bdg-danger:`Deprecation` Empty region signals resulting from applying `mask_img` in :class:`~maskers.NiftiMapsMasker` will no longer be kept in release ``0.15``. Meanwhile, use `keep_masked_maps` parameter when initializing the "NiftiMapsMasker" object to enable/disable this behavior. (:gh:`3732` by `Mohammad Torabi`_).