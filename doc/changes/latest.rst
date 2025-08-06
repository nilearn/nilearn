.. currentmodule:: nilearn

.. include:: names.rst

0.12.1.dev
==========

NEW
---

Fixes
-----

- :bdg-dark:`Code` Fix several issues in :class:`~nilearn.maskers.NiftiLabelsMasker` that lead to invalid ``region_names_``, ``region_ids_`` or look-up-table content (:gh:`5492` by `Rémi Gau`_).

- :bdg-dark:`Code` Align ``symmetric_cmap`` behavior for ``plotly`` backend in :func:`~nilearn.plotting.plot_surf` function with ``matplotlib`` backend (:gh:`5492` by `Hande Gözükan`_).

Enhancements
------------

Changes
-------

- :bdg-dark:`Code` Resampling of maps by :class:`~nilearn.maskers.NiftiMapsMasker` is now done with a linear insteadt of a continuous interpolation  (:gh:`5519` by `Rémi Gau`_).
