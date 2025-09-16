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

- :bdg-success:`API` Add a :class:`~exceptions.NotImplementedWarning` and make :class:`~exceptions.DimensionError`, :class:`~exceptions.AllVolumesRemovedError` and :class:`~exceptions.MeshDimensionError` part of the public API (:gh:`5508`, :gh:`5570` by `Rémi Gau`_).

Changes
-------

- :bdg-danger:`Deprecation` The attribute ``nifti_maps_masker_`` was removed from :class:`~decomposition.CanICA` and :class:`~decomposition.DictLearning`. Use ``maps_masker_`` instead (:gh:`5626` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` Extra key-words arguments (``kwargs``) have been removed from the constructor of all the Nifti maskers. Any extra-parameters to pass to the call to :func:`~image.clean_img` done by ``transform`` must be done via the parameter ``clean_args`` (:gh:`5628` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``darkness`` was removed from the function :func:`~plotting.view_img_on_surf`, :func:`~plotting.view_surf`, :func:`~plotting.plot_surf`, :func:`~plotting.plot_surf_stat_map` and :func:`~plotting.plot_surf_roi` (:gh:`5625` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``tr`` was replaced by ``t_r`` in :func:`~glm.first_level.glover_dispersion_derivative`, :func:`~glm.first_level.glover_hrf`, :func:`~glm.first_level.glover_time_derivative`, :func:`~glm.first_level.spm_dispersion_derivative`, :func:`~glm.first_level.spm_hrf` and :func:`~glm.first_level.spm_time_derivative`  (:gh:`5623` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``ax`` was replaced by ``axes`` in :func:`~plotting.plot_contrast_matrix` and :func:`~plotting.plot_design_matrix` (:gh:`5661` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` Systematically use ``imgs`` instead of ``X`` or ``img`` in the methods ``fit()``, ``transform()`` and ``fit_transform()`` for the Nilearn maskers and their derivative classes (:gh:`5624` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The attribute ``nifti_maps_masker_`` was removed from :class:`~decomposition.CanICA` and :class:`~decomposition.DictLearning`. Use ``maps_masker_`` instead. (:gh:`5626` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The default for ``keep_masked_maps`` and ``keep_masked_labels`` was changed from ``True`` to ``False`` in :class:`~maskers.NiftiMapsMasker`, :class:`~maskers.MultiNiftiMapsMasker`, :class:`~maskers.NiftiLabelsMasker`, :class:`~maskers.MultiNiftiLabelsMasker`, :class:`~regions.RegionExtractor` and :func:`~regions.img_to_signals_labels`. These parameters will be removed in Nilearn>=0.15.0 (:gh:`5632` by `Rémi Gau`_).

- :bdg-danger:`Deprecation`  In the ``scrubbing`` strategy used by :func:`~interfaces.fmriprep.load_confounds`, the default value of ``fd_threshold`` was changed from ``0.2`` to ``0.5`` and that of ``std_dvars_threshold`` was changed from ``3.0`` to ``1.5`` to better match the values used by fmriprep (:gh:`5633` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The default for the parameter ``homogeneity`` of :func:`~datasets.fetch_atlas_craddock_2012` was changed from ``None`` to ``'spatial'``. The only allowed values for this parameter must now be one of ``'spatial'``, ``'temporal'`` or ``'random'``. The fetcher now only returns a single map under the key ``maps`` for the requested ``homogeneity`` and ``grp_mean`` (:gh:`5640` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The default for the parameter ``dimension`` of :func:`~datasets.fetch_atlas_smith_2009` was changed from ``None`` to ``10``. The only allowed values for this parameter must now be one of ``10``, ``20`` or ``70``. The fetcher now only returns a single map under the key ``maps`` for the requested ``dimension`` and ``resting`` (:gh:`5640` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The default for the parameter ``extrapolate`` of :func:`~signal.clean` was changed from ``True`` to ``False`` (:gh:`5675` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The default for the parameters ``n_networks`` and ``thickness`` of :func:`~datasets.fetch_atlas_yeo_2011` were changed from ``None`` to ``7`` and ``"thick"``. The only allowed values for ``n_networks`` must now be one of ``7``, ``17``. The only allowed values for ``thickness`` must now be one of ``thin``, ``thick``. The fetcher now only returns a single map under the key ``maps`` for the requested ``n_networks`` and ``thickness`` (:gh:`5640` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``legacy_output`` of :func:`~datasets.fetch_language_localizer_demo_dataset` has been removed. The fetcher now always returns a Scikit-Learn Bunch (:gh:`5640` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The function ``nilearn.datasets.fetch_bids_langloc_dataset`` has been removed. Use :func:`~datasets.fetch_language_localizer_demo_dataset` instead. (:gh:`5640` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The default version returned by :func:`~datasets.fetch_atlas_aal` was changed from ``3v2`` (:gh:`5640` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The default to ``force_resample`` was set to True in :func:`~image.resample_img` (:gh:`5635` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``return_z_score`` of :func:`~glm.compute_fixed_effects` has been removed. :func:`~glm.compute_fixed_effects` will now always return 4 values instead of 3: the fourth one is ``fixed_fx_z_score_img`` (:gh:`5626` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``output_type`` of :func:`~mass_univariate.permuted_ols` was changed from ``legacy`` to ``dict``. The parameter ``output_type`` will be removed in Nilearn >= 0.15.0 (:gh:`5631` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``contrast_type`` of  :func:`~glm.compute_contrast` and :class:`~glm.Contrast` has been replaced by ``stat_type`` (:gh:`5630` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The value ``'nearest'`` for the parameter ``interpolation`` of :func:`~surface.vol_to_surf` is no longer allowed. Use ``'nearest_most_frequent'`` instead (:gh:`5662` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` :func:`~interfaces.bids.parse_bids_filename` will now always return a dictionary with the keys ``'file_path'``, ``'file_basename'``, ``'extension'``, ``'suffix'`` and ``'entities'`` (:gh:`5663` by `Rémi Gau`_).
