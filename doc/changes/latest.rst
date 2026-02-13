.. currentmodule:: nilearn

.. include:: names.rst

0.14.0.dev
==========

NEW
---

Fixes
-----

Enhancements
------------


- :bdg-dark:`Code` Added ``screening_n_features`` parameter to :class:`~nilearn.decoding.Decoder`,  :class:`~nilearn.decoding.DecoderRegressor`, :class:`~nilearn.decoding.FREMClassifier`,  and :class:`~nilearn.decoding.FREMRegressor`.

- :bdg-success:`API` Support pathlike objects for ``cmap`` argument in :func:`~plotting.plot_surf_roi` (:gh:`5981` by `Joseph Paillard`_).


Changes
-------

- :bdg-danger:`Deprecation` The function ``nilearn.datasets.utils.load_sample_motor_activation_image`` and ``nilearn.datasets.fetch_neurovault_motor_task`` were removed. Use :func:`~datasets.load_sample_motor_activation_image` instead (:gh:`5995` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The private functions ``nilearn._utils.niimg_conversions.check_niimg*`` have been removed, please use their public equivalent :func:`~image.check_niimg`, :func:`~image.check_niimg_3d` and :func:`~image.check_niimg_4d` (:gh:`5995` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` Accessing the maskers from ``nilearn.input_data`` is no longer possible, they now must be accessed via :mod:`nilearn.maskers` (:gh:`5995` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The ``version`` parameters of in :func:`~datasets.fetch_atlas_pauli_2017` has now permanently been replaced by ``atlas_type`` (:gh:`5995` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` ``plot_img_comparison`` is no longer accessible from ``nilearn.plotting.image.img_plotting``, access it from ``nilearn.plotting`` or from ``nilearn.plotting.img_comparison`` (:gh:`5995` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The ``"z_score"`` value for the ``standardize`` parameter is no longer supported. Use ``standardize="z_score_sample"`` instead (:gh:`5995` by `Rémi Gau`_).
