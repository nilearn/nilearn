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

:bdg-danger:`Deprecation` Empty region signals resulting from applying `mask_img` in :class:`~maskers.NiftiLabelsMasker` will no longer be kept in release 0.15. Meanwhile, use `keep_masked_labels` parameter when initializing the :class:`~maskers.NiftiLabelsMasker` object to enable/disable this behavior. (:gh:`3722` by `Mohammad Torabi`_).