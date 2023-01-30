.. currentmodule:: nilearn

.. include:: names.rst

0.10.1.dev
==========

NEW
---

Fixes
-----

- Restore :func:`~image.resample_img` compatibility with all :class:`nibabel.spatialimages.SpatialImage` objects (:gh:`3462` by `Mathias Goncalves`_).

Enhancements
------------

- Addition to docs to note that :meth:`~maskers.BaseMasker.inverse_transform` only performs spatial unmasking (:gh:`3445` by `Robert Williamson`_).

Changes
-------
