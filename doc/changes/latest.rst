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

- :func:`~signal.clean` imputes scrubbed volumes (defined through ``sample_masks``) with cubic spline function before applying butterworth filter (:gh:`3385` by `Hao-Ting Wang`_). 
- As part of making the User Guide more user-friendly, the introduction was reworked (:gh:`3380` by `Alexis Thual`_)
- Added instructions for maintainers to make sure LaTeX dependencies are installed before building and deploying the stable docs (:gh:`3426` by `Yasmin Mzayek`_).
- Addition to docs to note that :meth:`~maskers.BaseMasker.inverse_transform` only performs spatial unmasking (:gh: `3445` by `Robert Williamson`_).

Changes
-------
