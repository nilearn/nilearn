.. currentmodule:: nilearn

.. include:: names.rst

0.10.1.dev
==========

NEW
---

Fixes
-----

- :func:`~nilearn.interfaces.fmriprep.load_confounds` can support searching preprocessed data in native space. (:gh:`3531` by `Hao-Ting Wang`_)

Enhancements
------------

- Surface plotting methods no longer automatically rescale background maps, which, among other things, allows to use curvature sign as a background map (:gh:`3173` by `Alexis Thual`_).

Changes
-------
- The behavior of :func:`~nilearn.datasets.fetch_atlas_craddock_2012`, :func:`~nilearn.datasets.fetch_atlas_smith_2009` and :func:`~nilearn.datasets.fetch_atlas_basc_multiscale_2015` is updated with their new parameters to return one map along with a deprecation cycle (:gh:`3353` by `Ahmad Chamma`_).


0.10.1rc1
=========

**Released February 2023**

This is a pre-release.


Fixes
-----

- :bdg-dark:`Code` Restore :func:`~image.resample_img` compatibility with all :class:`nibabel.spatialimages.SpatialImage` objects (:gh:`3462` by `Mathias Goncalves`_).

- :bdg-success:`API` :func:`~nilearn.glm.second_level.non_parametric_inference` now supports confounding variates when they are available in the input design matrix :func:`~nilearn.mass_univariate.permuted_ols` (:gh:`3465` by `Jelle Roelof Dalenberg`_).

- :bdg-success:`API` :func:`~nilearn.mass_univariate.permuted_ols` now checks if confounding variates contain a intercept and raises an warning when multiple intercepts are defined across target and confounding variates (:gh:`3465` by `Jelle Roelof Dalenberg`_).

- The label of the clusters in the label maps returned by :func:`~nilearn.reporting.get_clusters_table` now matches the Cluster IDs in the clusters table (:gh:`3563` by `Julio A Peraza`_).

Enhancements
------------

- :bdg-primary:`Doc` Addition to docs to note that :meth:`~maskers.BaseMasker.inverse_transform` only performs spatial unmasking (:gh:`3445` by `Robert Williamson`_).


- :bdg-success:`API` Give users control over Butterworth filter (:func:`~signal.butterworth`) parameters in :func:`~signal.clean` and Masker objects as kwargs (:gh:`3478` by `Taylor Salo`_).

- :bdg-success:`API` Allow users to output label maps from :func:`~reporting.get_clusters_table` (:gh:`3477` by `Steven Meisler`_).

Changes
-------

- :bdg-primary:`Doc` The documentation for :func:`~image.threshold_img` has been improved, with more information about which voxels are set to 0 and which ones keep their original values (:gh:`3485` by `Rémi Gau`_).

- :bdg-secondary:`Maint` Modules :mod:`~nilearn.decomposition` and :mod:`~nilearn.decoding` code and docstrings have been reformatted using black. Changes resulted in improved readability overall and increased consistency (:gh:`3491` and :gh:`3484` by `Rémi Gau`_).
