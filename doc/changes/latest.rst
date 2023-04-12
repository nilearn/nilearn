.. currentmodule:: nilearn

.. include:: names.rst

0.10.1.dev
==========

NEW
---

- New function :func:`~datasets.load_sample_motor_activation_image` to load example contrast map (:gh:`3498` by `Michelle Wang`_).

- ``fsaverage`` meshes accessed through :func:`~datasets.fetch_surf_fsaverage` now come with flat maps for all resolutions (:gh:`3444` by `Alexis Thual`_).

- surface plotting functions allow setting custom view angles and are no longer limited to predefined views (:gh:`3259` by `Sam Buck Johnson`_ and `Alexis Thual`_).

Fixes
-----

- Fixes several bugs in :func:`~glm.first_level.first_level_from_bids`. Refactors :func:`~glm.first_level.first_level_from_bids` and ``nilearn._utils.data_gen.create_fake_bids_dataset``. (:gh:`3525` by `Rémi Gau`_).

- Change calculation of TR in :func:`~.glm.first_level.compute_regressor` to be more precise (:gh:`3362` by `Anne-Sophie Kieslinger`_)

- :func:`~nilearn.interfaces.fmriprep.load_confounds` can support searching preprocessed data in native space. (:gh:`3531` by `Hao-Ting Wang`_)

- :func:`~glm.second_level.non_parametric_inference` can accept first level model as input without failing. (:gh:`3600` by `Rémi Gau`_)

- Add correct "zscore_sample" strategy to ``signal._standardize`` which will replace the default "zscore" strategy in release 0.13  (:gh:`3474` by `Yasmin Mzayek`_).


Enhancements
------------

- Updated example :ref:`sphx_glr_auto_examples_02_decoding_plot_haxby_frem.py` to include section on plotting a confusion matrix from a decoder family object (:gh:`3483` by `Michelle Wang`_).

- Surface plotting methods no longer automatically rescale background maps, which, among other things, allows to use curvature sign as a background map (:gh:`3173` by `Alexis Thual`_).

- :func:`~glm.first_level.first_level_from_bids` now takes an optional ``sub_labels`` argument and warns users of given subject labels that are not present in the dataset (:gh:`3351` by `Kevin Sitek`_).


Changes
-------
- The behavior of :func:`~nilearn.datasets.fetch_atlas_craddock_2012`, :func:`~nilearn.datasets.fetch_atlas_smith_2009` and :func:`~nilearn.datasets.fetch_atlas_basc_multiscale_2015` is updated with their new parameters to return one map along with a deprecation cycle (:gh:`3353` by `Ahmad Chamma`_).

- Modules :mod:`~nilearn.image` code and docstrings have been reformatted using black. Changes resulted in improved readability overall and increased consistency (:gh:`3548` by `Rémi Gau`_).

- Examples have been made PEP8 compliant and reformatted using black. (:gh:`3549`, :gh:`3550`, :gh:`3551`, :gh:`3552`, :gh:`3553`, :gh:`3554`, :gh:`3555`,  by `Rémi Gau`_).

- Extract helper-functions for input-image validation from :func:`~regions.img_to_signals_labels`, :func:`~regions.signals_to_img_labels`, :func:`~regions.img_to_signals_maps` :func:`~regions.signals_to_img_maps` (:gh:`3523` by `Rémi Gau`_ and `Christian Gerloff`_).

- Moved packaging from ``setup.py`` and setuptools build backend to ``pyproject.toml`` and hatchling backend. This change comes about as new standards are defined for Python packaging that are better met by the new configuration (:gh:`3635` by `Yasmin Mzayek`_).

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
