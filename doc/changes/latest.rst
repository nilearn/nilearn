.. currentmodule:: nilearn

.. include:: names.rst

0.12.1.dev
==========

NEW
---

Fixes
-----

- :bdg-dark:`Code` Align ``symmetric_cmap`` behavior for ``plotly`` backend in :func:`~nilearn.plotting.plot_surf` function with ``matplotlib`` backend (:gh:`5492` by `Hande Gözükan`_).

- :bdg-dark:`Code` Enforce consistent ``dtype`` for all parts of :class:`~nilearn.surface.surface.SurfaceImage` and :class:`~nilearn.surface.surface.Polydata` (:gh:`5530` by `Rémi Gau`_).


Enhancements
------------

Changes
-------

- :bdg-dark:`Code` Resampling of maps by :class:`~nilearn.maskers.NiftiMapsMasker` is now done with a linear insteadt of a continuous interpolation  (:gh:`5519` by `Rémi Gau`_).
