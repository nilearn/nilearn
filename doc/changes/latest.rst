.. currentmodule:: nilearn

.. include:: names.rst

0.10.1.dev
==========

NEW
---

Fixes
-----

- Restore :func:`~image.resample_img` compatibility with all :class:`nibabel.spatialimages.SpatialImage` objects (:gh:`3462` by `Mathias Goncalves`_).

- :func:`~nilearn.glm.second_level.non_parametric_inference` now supports confounding variates when they are available in the input design matrix :func:`~nilearn.mass_univariate.permuted_ols` (:gh:`3465` by `Jelle Roelof Dalenberg`_).

- :func:`~nilearn.mass_univariate.permuted_ols` now checks if confounding variates contain a intercept and raises an warning when multiple intercepts are defined across target and confounding variates (:gh:`3465` by `Jelle Roelof Dalenberg`_).

Enhancements
------------

- Addition to docs to note that :meth:`~maskers.BaseMasker.inverse_transform` only performs spatial unmasking (:gh:`3445` by `Robert Williamson`_).
- Give users control over Butterworth filter (:func:`~signal.butterworth`) parameters in :func:`~signal.clean` and Masker objects as kwargs (:gh:`3478` by `Taylor Salo`_).
- Allow users to output label maps from :func:`~reporting.get_clusters_table` (:gh:`3477` by `Steven Meisler`_).

Changes
-------

- The documentation for :func:`~image.threshold_img` has been improved, with more information about which voxels are set to 0 and which ones keep their original values (:gh:`3485` by `Rémi Gau`_).
