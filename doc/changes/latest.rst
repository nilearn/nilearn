.. currentmodule:: nilearn

.. include:: names.rst

0.13.1.dev
==========

NEW
---

Fixes
-----

- :bdg-info:`Plotting` drop background color when using look up table as colormap (:gh:`5936` by `Rémi Gau`_).

- :bdg-dark:`Code` Reallow use non-multi maskers for :class:`~nilearn.regions.Parcellations` (:gh:`5930` by `Rémi Gau`_).


Enhancements
------------

- :bdg-success:`API` The parameter ``estimator_args`` was added to all decoding estimators to allow to pass parameters directly to the underlying Scikit-Learn estimators (:gh:`5641` by `Rémi Gau`_).

- :bdg-success:`API` Add support for dictionary as ``cut_coords`` for :class:`~plotting.displays.MosaicSlicer` and image plotting functions. (:gh:`5920` by `Hande Gözükan`_).

Changes
-------
