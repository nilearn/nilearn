.. currentmodule:: nilearn

.. include:: names.rst

0.12.1.dev
==========

NEW
---

Fixes
-----

- :bdg-dark:`Code` Align ``symmetric_cmap`` behavior for ``plotly`` backend in :func:`~nilearn.plotting.plot_surf` function with ``matplotlib`` backend (:gh:`5492` by `Hande Gözükan`_).
- :bdg-dark:`Code` Fix type of ``t_r`` to support numpy dtypes for python < 3.10 (:gh:`5550` by `Rémi Gau`_).

- :bdg-dark:`Code` Enforce consistent ``dtype`` for all parts of :class:`~nilearn.surface.SurfaceImage` and :class:`~nilearn.surface.PolyData` (:gh:`5530` by `Rémi Gau`_).


Enhancements
------------

Changes
-------

- :bdg-dark:`Code` Resampling of maps by :class:`~nilearn.maskers.NiftiMapsMasker` is now done with a linear instead of a continuous interpolation  (:gh:`5519` by `Rémi Gau`_).

- :bdg-dark:`Code` Move ``nilearn.plotting.img_plotting`` under ``nilearn.plotting.image`` (:gh:`5481` by `Hande Gözükan`_).
