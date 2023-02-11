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

- Surface plotting methods now accept the ``bg_map_rescale`` parameter, which, among other things, allows to use curvature sign as a background map (:gh:`3173` by `Alexis Thual`_).
- Addition to docs to note that :meth:`~maskers.BaseMasker.inverse_transform` only performs spatial unmasking (:gh:`3445` by `Robert Williamson`_).
- Give users control over Butterworth filter (:func:`~signal.butterworth`) parameters in :func:`~signal.clean` and Masker objects as kwargs (:gh:`3478` by `Taylor Salo`_).

Changes
-------

- The documentation for :func:`~image.threshold_img` has been improved, with more information about which voxels are set to 0 and which ones keep their original values (:gh:`3485` by `Rémi Gau`_).
