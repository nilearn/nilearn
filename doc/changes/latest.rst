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

Changes
-------

- :bdg-danger:`Deprecation` Extra key-words arguments (``kwargs``) have been removed from the constructor of all the Nifti maskers. Any extra-parameters to pass to the call to :func:`~image.clean_img` done by ``transform`` must be done via the parameter ``clean_args`` (:gh:`5628` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``darkness`` was removed from the function :func:`~plotting.view_img_on_surf`, :func:`~plotting.view_surf`, :func:`~plotting.plot_surf`, :func:`~plotting.plot_surf_stat_map` and :func:`~plotting.plot_surf_roi` (:gh:`5625` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``tr`` was replaced by ``t_r`` in :func:`~glm.first_level.glover_dispersion_derivative`, :func:`~glm.first_level.glover_hrf`, :func:`~glm.first_level.glover_time_derivative`, :func:`~glm.first_level.spm_dispersion_derivative`, :func:`~glm.first_level.spm_hrf` and :func:`~glm.first_level.spm_time_derivative`  (:gh:`5623` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` Systematically use ``imgs`` instead of ``X`` or ``img`` in the methods ``fit()``, ``transform()`` and ``fit_transform()`` for the Nilearn maskers and their derivative classes (:gh:`5624` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The attribute ``nifti_maps_masker_`` was removed from :class:`~decomposition.CanICA` and :class:`~decomposition.DictLearning`. Use ``maps_masker_`` instead. (:gh:`5626` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The default to ``force_resample`` was set to True in :func:`~image.resample_img` (:gh:`5635` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``return_z_score`` of :func:`~glm.compute_fixed_effects` has been removed. :func:`~glm.compute_fixed_effects` will now always return 4 values instead of 3: the fourth one is ``fixed_fx_z_score_img`` (:gh:`5626` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``output_type`` of :func:`~mass_univariate.permuted_ols` was changed from ``legacy`` to ``dict``. The parameter ``output_type`` will be removed in Nilearn >= 0.15.0 (:gh:`5631` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``contrast_type`` of  :func:`~glm.compute_contrast` and :class:`~glm.Contrast` has been replaced by ``stat_type`` (:gh:`5630` by `Rémi Gau`_).
