.. currentmodule:: nilearn

.. include:: names.rst

0.12.1.dev
==========

NEW
---

Fixes
-----

- :bdg-dark:`Code` Fix several issues in :class:`~nilearn.maskers.NiftiLabelsMasker` and :class:`~nilearn.maskers.SurfaceLabelsMasker` that lead to invalid ``region_names_``, ``region_ids_`` or look-up-table content (:gh:`5492` by `Rémi Gau`_).

- :bdg-dark:`Code` Align ``symmetric_cmap`` behavior for ``plotly`` backend in :func:`~nilearn.plotting.plot_surf` function with ``matplotlib`` backend (:gh:`5492` by `Hande Gözükan`_).

- :bdg-dark:`Code` Fix type of ``t_r`` to support numpy dtypes for python < 3.10 (:gh:`5550` by `Rémi Gau`_).

- :bdg-dark:`Code` Ensure that estimators that accept images can work will several datatypes as input and that their methods can output arrays or images of the requested datatype (:gh:`5511` by `Rémi Gau`_).

- :bdg-dark:`Code` Enforce consistent ``dtype`` for all parts of :class:`~nilearn.surface.SurfaceImage` and :class:`~nilearn.surface.PolyData` (:gh:`5530` by `Rémi Gau`_).


Enhancements
------------

- :bdg-dark:`Code` Enforce consistent ``dtype`` for all parts of :class:`~nilearn.surface.SurfaceImage` and :class:`~nilearn.surface.PolyData` (:gh:`5530` by `Rémi Gau`_).

- :bdg-success:`API` The fitted attribute ``n_elements_`` was added to following estimators: :class:`~nilearn.glm.first_level.FirstLevelModel`, :class:`~nilearn.glm.second_level.SecondLevelModel`, :class:`~nilearn.decoding.Decoder`, :class:`~nilearn.decoding.DecoderRegressor`, :class:`~nilearn.decoding.FREMClassifier`, :class:`~nilearn.decoding.FREMRegressor`, :class:`~nilearn.decoding.SearchLight`. This attribute is equivalent to the `n_features_in_ <https://scikit-learn.org/stable/developers/develop.html#universal-attribute>`_ of scikit-learn estimators.


Changes
-------

- :bdg-dark:`Code` Resampling of maps by :class:`~nilearn.maskers.NiftiMapsMasker` is now done with a linear instead of a continuous interpolation  (:gh:`5519` by `Rémi Gau`_).

- :bdg-dark:`Code` Move ``nilearn.plotting.img_plotting`` under ``nilearn.plotting.image`` (:gh:`5481` by `Hande Gözükan`_).
