.. currentmodule:: nilearn

.. include:: names.rst

0.12.1.dev
==========

NEW
---

Fixes
-----

- :bdg-success:`API` Add a dummy ``y`` parameter to :meth:`~nilearn.decomposition.CanICA.score` and :meth:`~nilearn.decomposition.DictLearning.score` for compatibility with scikit-learn API (:gh:`5565` by `Rémi Gau`_).

- :bdg-dark:`Code` Fix several issues in :class:`~nilearn.maskers.NiftiLabelsMasker` and :class:`~nilearn.maskers.SurfaceLabelsMasker` that lead to invalid ``region_names_``, ``region_ids_`` or look-up-table content (:gh:`5492` by `Rémi Gau`_).

- :bdg-dark:`Code` Align ``symmetric_cmap`` behavior for ``plotly`` backend in :func:`~nilearn.plotting.plot_surf` function with ``matplotlib`` backend (:gh:`5492` by `Hande Gözükan`_).

- :bdg-dark:`Code` Fix type of ``t_r`` to support numpy dtypes for python < 3.10 (:gh:`5550` by `Rémi Gau`_).

- :bdg-dark:`Code` Enforce consistent ``dtype`` for all parts of :class:`~nilearn.surface.SurfaceImage` and :class:`~nilearn.surface.PolyData` (:gh:`5530` by `Rémi Gau`_).

- :bdg-success:`API` The ``is_classif`` public attribute has been removed for :class:`~decoding.SpaceNetClassifier` and :class:`~decoding.DecoderRegressor` as it is a characteristic of the estimator that must not be changed. Accessing an equivalent characteristic can be done via the estimator's tags (``__sklearn_tags__()``) (:gh:`5594` by `Rémi Gau`_).

- :bdg-success:`API` The ``is_classification`` public attribute has been removed for :class:`~decoding.Decoder`, :class:`~decoding.DecoderRegressor`, :class:`~decoding.FREMClassifier` and  :class:`~decoding.FREMRegressor` as it is a characteristic of the estimator that must not be changed. Accessing an equivalent characteristic can be done via the estimator's tags (``__sklearn_tags__()``) (:gh:`5557` by `Rémi Gau`_).

- :bdg-success:`API` The ``loss`` public attribute has been removed for :class:`~decoding.SpaceNetRegressor` it can only be ``'mse'`` and should not be changed (:gh:`5594` by `Rémi Gau`_).

- :bdg-success:`API` The ``clustering_percentile`` public attribute has been removed for :class:`~decoding.Decoder` and :class:`~decoding.DecoderRegressor` as it is only relevant for :class:`~decoding.FREMClassifier` and  :class:`~decoding.FREMRegressor` (:gh:`5557` by `Rémi Gau`_).


Enhancements
------------

- :bdg-dark:`Code` Enforce consistent ``dtype`` for all parts of :class:`~nilearn.surface.SurfaceImage` and :class:`~nilearn.surface.PolyData` (:gh:`5530` by `Rémi Gau`_).

- :bdg-success:`API` The fitted attribute ``n_elements_`` was added to following estimators: :class:`~nilearn.glm.first_level.FirstLevelModel`, :class:`~nilearn.glm.second_level.SecondLevelModel`, :class:`~nilearn.decoding.Decoder`, :class:`~nilearn.decoding.DecoderRegressor`, :class:`~nilearn.decoding.FREMClassifier`, :class:`~nilearn.decoding.FREMRegressor`, :class:`~nilearn.decoding.SearchLight`. This attribute is equivalent to the `n_features_in_ <https://scikit-learn.org/stable/developers/develop.html#universal-attribute>`_ of scikit-learn estimators.


Changes
-------

- :bdg-dark:`Code` Resampling of maps by :class:`~nilearn.maskers.NiftiMapsMasker` is now done with a linear instead of a continuous interpolation  (:gh:`5519` by `Rémi Gau`_).

- :bdg-dark:`Code` Move ``nilearn.plotting.img_plotting`` under ``nilearn.plotting.image`` (:gh:`5481` by `Hande Gözükan`_).

- :bdg-danger:`Deprecation` From Nilearn >= 0.15, the default value of ``threshold`` will be changed to ``scipy.stats.norm.isf(0.001)`` (``3.09023...``) in :func:`~glm.threshold_stats_img`, :func:`~glm.cluster_level_inference`, :func:`~reporting.make_glm_report`, :meth:`~glm.first_level.FirstLevelModel.generate_report`, :meth:`~glm.second_level.SecondLevelModel.generate_report` (:gh:`5601` by `Rémi Gau`_).

- :bdg-dark:`Code` Decoding estimators do not inherit from sklearn ``ClassifierMixin`` and ``RegressorMixing`` anymore. It is recommended to rely on estimator tags (accessible via the ``'__sklearn_tags__()'`` special method) to know more about the characteristics of an instance  (:gh:`5595` by `Rémi Gau`_).