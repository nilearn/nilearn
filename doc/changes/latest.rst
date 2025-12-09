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
 | - scipy -- 1.9.0
 | - requests -- 2.30.0
 |
 | **A new dependency has been added:**
 | - jinja2 -- 3.1.2
 |
 | **A dependency has been removed:**
 | - lxml


NEW
---

Fixes
-----

- :bdg-primary:`Doc` Update allowed values for the parameter ``standardize`` to match those of :func:`~signal.clean` in :class:`~decoding.Decoder`, :class:`~decoding.DecoderRegressor`, :class:`~decoding.FREMClassifier`, :class:`~decoding.FREMRegressor`, :class:`~decoding.SpaceNetClassifier`, :class:`~decoding.SpaceNetRegressor`, :class:`~connectome.ConnectivityMeasure`, :class:`~decomposition.DictLearning`, :class:`~decomposition.CanICA` as well as for all maskers  (:gh:`5668` by `Rémi Gau`_).

- :bdg-info:`Plotting` Allow to pass files to :func:`~nilearn.plotting.plot_img_comparison` (:gh:`5825` by `Rémi Gau`_).

- :bdg-dark:`Code` Improve errors thrown when the confounds for a subject or group level analysis GLM contain NaN (:gh:`5739` by `Rémi Gau`_).

- :bdg-dark:`Code` Make sure names in atlas labels look up tables are not shifted when the background name is not properly indicated (:gh:`5826` by `Rémi Gau`_).

- :bdg-dark:`Code` Inform user correctly when reporting is enabled after the model is fit (:gh:`5836` by `Hande Gözükan`_).

- :bdg-dark:`Code` Raise warning when :func:`~nilearn.image.crop_img` is called with an empty image and return the original image (:gh:`5837` by `Hande Gözükan`_).

- :bdg-dark:`Code` Better handling of errors during plotting when cut coordinates are out of bounds (:gh:`5861` by `Sanjana Soni`_).

Enhancements
------------

- :bdg-success:`API` The functions :func:`~image.check_niimg`, :func:`~image.check_niimg_3d` and :func:`~image.check_niimg_4d` are now part of our public API. The content of ``nilearn._utils.niimg_conversions`` and ``nilearn.image.utils`` was moved to ``nilearn.image.image``. For a smoother transition, ``check_niimg``, ``check_niimg_3d`` and ``check_niimg_4d`` will still be importable from ``nilearn._utils.niimg_conversions`` till Nilearn 0.14.0 (:gh:`5788` by `Rémi Gau`_).

- :bdg-dark:`Code` Add tedana support for load_confounds and testing for tedana load_confounds support (:gh:`5410` by `Milton Camacho`_).

- :bdg-dark:`Code` Parameter ``head_tpl`` in  :class:`~reporting.HTMLReport` can now be a Jinja2 template (:gh:`5710` by `Rémi Gau`_).

- :bdg-dark:`Code` NaN values contained in the first row of the confounds loaded by :func:`~glm.first_level.first_level_from_bids` will be turned into 0 to avoid downstream errors when creating design matrices (:gh:`5739` by `Rémi Gau`_).

- :bdg-success:`API` Add :func:`~utils.all_estimators`, :func:`~utils.all_displays`, :func:`~utils.all_functions` to provide list all estimators and functions available in Nilearn (:gh:`5535` by `Rémi Gau`_).

- :bdg-success:`API` Add an ``exclude_subjects`` parameter to :func:`~glm.first_level.first_level_from_bids` to skip some subjects when creating GLM models from a BIDS dataset (:gh:`5741` by `Rémi Gau`_).

- :bdg-success:`API` Add ``view`` parameter to :func:`~plotting.view_img_on_surf` to select the default view that will be used when displaying the figure  (:gh:`5692` by `Rémi Gau`_).

- :bdg-dark:`Code` Add anterior and posterior views to :class:`~maskers.SurfaceLabelsMasker`, :class:`~maskers.SurfaceMapsMasker` and :class:`~maskers.SurfaceMasker` (:gh:`5473` by `Chloe Hampson`_).

- :bdg-success:`API` Add a :class:`~exceptions.NotImplementedWarning` and make :class:`~exceptions.MaskWarning`, :class:`~exceptions.DimensionError`, :class:`~exceptions.AllVolumesRemovedError` and :class:`~exceptions.MeshDimensionError` part of the public API (:gh:`5508`, :gh:`5570`, :gh:`5677` by `Rémi Gau`_).

- :bdg-success:`API` Add support for Scikit-Learn ``set_output()`` in several Nilearn feature extractors (nifti and surface non-multi maskers, and :class:`~regions.HierarchicalKMeans`) to allow ``transform()`` to output to either Pandas or Polars dataframe and not just numpy arrays (:gh:`5508` by `Rémi Gau`_).

- :bdg-success:`API` Add a :class:`~maskers.MultiSurfaceMasker`, a :class:`~maskers.MultiSurfaceLabelsMasker` (:gh:`5679`, :gh:`5726` by `Rémi Gau`_).

- :bdg-success:`API` Add a :class:`~maskers.MultiSurfaceMasker`, a :class:`~maskers.MultiSurfaceMapsMasker` (:gh:`5679`, :gh:`5727` by `Rémi Gau`_).

- :bdg-success:`API` Add support for ``cluster_threshold`` for :class:`~surface.SurfaceImage` in :func:`~image.threshold_img`, :func:`~glm.threshold_stats_img`, :func:`~reporting.make_glm_report`, :meth:`~glm.first_level.FirstLevelModel.generate_report` and :meth:`~glm.second_level.SecondLevelModel.generate_report` (:gh:`5715` by `Rémi Gau`_).

- :bdg-success:`API` Add support for :class:`~surface.SurfaceImage` in :func:`~glm.cluster_level_inference` (:gh:`5733` by `Rémi Gau`_).

- :bdg-success:`API` Add support for ``title`` parameter for Nilearn maskers ``generate_report`` method. (:gh:`5790` by `Hande Gözükan`_).

Changes
-------

- :bdg-danger:`Deprecation` The function :func:`~nilearn.glm.save_glm_to_bids` was moved to the :mod:`~nilearn.glm` module. It will be importable from its original :mod:`~nilearn.interfaces` till Nilearn version 0.15.0 (:gh:`5770` by `Rémi Gau`_).

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

- :bdg-danger:`Deprecation` The default for the parameter ``two_sided`` of :func:`~image.binarize_img` was changed from ``True`` to ``False`` (:gh:`5687` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``legacy_output`` of :func:`~datasets.fetch_language_localizer_demo_dataset` has been removed. The fetcher now always returns a Scikit-Learn Bunch (:gh:`5640` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The function ``nilearn.datasets.fetch_bids_langloc_dataset`` has been removed. Use :func:`~datasets.fetch_language_localizer_demo_dataset` instead. (:gh:`5640` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``subject_id`` of :func:`~datasets.fetch_spm_auditory` and   :func:`~datasets.fetch_spm_multimodal_fmri` has been removed (:gh:`5709` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``temp_file_lifetime`` of the :meth:`~reporting.HTMLReport.open_in_browser` has been removed (:gh:`5709` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The default version returned by :func:`~datasets.fetch_atlas_aal` was changed from ``SPM12`` to ``3v2`` (:gh:`5640` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The default to ``force_resample`` was set to ``True`` in :func:`~image.resample_img` (:gh:`5635` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``return_z_score`` of :func:`~glm.compute_fixed_effects` has been removed. :func:`~glm.compute_fixed_effects` will now always return 4 values instead of 3: the fourth one is ``fixed_fx_z_score_img`` (:gh:`5626` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``output_type`` of :func:`~mass_univariate.permuted_ols` was changed from ``legacy`` to ``dict``. The parameter ``output_type`` will be removed in Nilearn >= 0.15.0 (:gh:`5631` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``contrast_type`` of  :func:`~glm.compute_contrast` and :class:`~glm.Contrast` has been replaced by ``stat_type`` (:gh:`5630` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` ``roi_map`` for :func:`~plotting.plot_surf_roi` can now only be made of positive integer values (:gh:`5660` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The value ``'nearest'`` for the parameter ``interpolation`` of :func:`~surface.vol_to_surf` is no longer allowed. Use ``'nearest_most_frequent'`` instead (:gh:`5662` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` :func:`~interfaces.bids.parse_bids_filename` will now always return a dictionary with the keys ``'file_path'``, ``'file_basename'``, ``'extension'``, ``'suffix'`` and ``'entities'`` (:gh:`5663` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` Set ``copy_header`` default to True in :func:`~image` functions (:gh:`5656` by `Rémi Gau`_ and `Himanshu Aggarwal`_).

- :bdg-danger:`Deprecation` From Nilearn >= 0.15, the function :func:`~reporting.make_glm_report` will be removed. Use :meth:`~glm.first_level.FirstLevelModel.generate_report` or :meth:`~glm.second_level.SecondLevelModel.generate_report` instead. (:gh:`5876` by `Hande Gözükan`_).

Documentation
-------------

- :bdg-primary:`Doc` Clarified that ``nilearn.datasets.fetch_*`` functions do not re-download datasets already present locally. Added cross-reference to dataset storage documentation. (:gh:`5690` by `Victoria McCray`_)

- :bdg-primary:`Doc` Home-made sphinx directives are used instead of default sphinx directives relative to version changes (``versionadded``, ``versionchanged``, ``deprecated``...) to more easily distinguish between feature changes introduced in Nilearn versus those introduced in upstream dependencies (like in Scikit-Learn) (:gh:`5654` by `Rémi Gau`_).

- :bdg-info:`Plotting` Change cmap to ``'RdBu_r'`` for :func:`~plotting.plot_contrast_matrix` (:gh:`5780` by `Hande Gözükan`_).

- :bdg-info:`Plotting` Change background to ``'black'`` for functions :func:`~plotting.plot_img_comparison` and :func:`~plotting.plot_bland_altman` (:gh:`5785` by `Hande Gözükan`_).

- :bdg-dark:`Code` Reports (for maskers, GLMs...)  can now be generated even when no plotting engine is available (:gh:`5757` by `Rémi Gau`_).

- :bdg-dark:`Code` Dependency on ``lxml`` has been removed as it is only used during testing (:gh:`5862` by `Rémi Gau`_).
