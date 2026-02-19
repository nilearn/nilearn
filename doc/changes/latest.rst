.. currentmodule:: nilearn

.. include:: names.rst

0.14.0.dev
==========

HIGHLIGHTS
----------

- Nilearn can leverage scikit-learn's Array API-supported estimators to speed up neuroimaging ML analyses using GPU acceleration. See :ref:`user guide page <gpu_usage>`.

.. warning::

 | **Minimum supported versions of the following packages have been bumped up:**
 | - numpy -- 1.26.0
 | - pandas -- 2.3.0
 | - scikit-learn -- 1.5.0
 | - scipy -- 1.11.0

NEW
---

Fixes
-----

- :bdg-secondary:`Maint` Allow local installation with ```uv sync`` (:gh:`6024` by `Mathieu Dugré`_)

Enhancements
------------

- :bdg-dark:`Code` Add surface support to :func:`~nilearn.image.smooth_img` (:gh:`3267` by `Jason D. Yeatman`_ and `Noah C. Benson`_ ).

- :bdg-success:`API` Add a public method to access the fitted mask of GLM instances (:gh:`5981` by `Rémi Gau`_).

- :bdg-dark:`Code` Added ``screening_n_features`` parameter to :class:`~nilearn.decoding.Decoder`,  :class:`~nilearn.decoding.DecoderRegressor`, :class:`~nilearn.decoding.FREMClassifier`,  and :class:`~nilearn.decoding.FREMRegressor`.

- :bdg-success:`API` Support pathlike objects for ``cmap`` argument in :func:`~plotting.plot_surf_roi` (:gh:`5981` by `Joseph Paillard`_).

- :bdg-primary:`Doc` Added a :ref:`user guide page <gpu_usage>` to demonstrate speedup using GPU (:gh:`5958` by `Himanshu Aggarwal`_ and `Elizabeth DuPre`_).

Changes
-------

- :bdg-danger:`Deprecation` The function ``nilearn.datasets.utils.load_sample_motor_activation_image`` and ``nilearn.datasets.fetch_neurovault_motor_task`` were removed. Use :func:`~datasets.load_sample_motor_activation_image` instead (:gh:`5995` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The private functions ``nilearn._utils.niimg_conversions.check_niimg*`` have been removed, please use their public equivalent :func:`~image.check_niimg`, :func:`~image.check_niimg_3d` and :func:`~image.check_niimg_4d` (:gh:`5995` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` Accessing the maskers from ``nilearn.input_data`` is no longer possible, they now must be accessed via :mod:`nilearn.maskers` (:gh:`5995` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The ``version`` parameters of in :func:`~datasets.fetch_atlas_pauli_2017` has now permanently been replaced by ``atlas_type`` (:gh:`5995` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` ``plot_img_comparison`` is no longer accessible from ``nilearn.plotting.image.img_plotting``, access it from ``nilearn.plotting`` or from ``nilearn.plotting.img_comparison`` (:gh:`5995` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The ``"z_score"`` value for the ``standardize`` parameter is no longer supported. Use ``standardize="z_score_sample"`` instead (:gh:`5995` by `Rémi Gau`_).

- :bdg-dark:`Code` Remove aggressive garbage collection in safe_get_data for performance, mainly in CI. (:gh:`6039` by `Basile Pinsard`_).
