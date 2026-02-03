.. currentmodule:: nilearn

.. include:: names.rst

0.13.1.dev
==========

NEW
---

Fixes
-----

- :bdg-dark:`Code` Use sklearn HTML representation of estimators and their parameters in notebooks and reports (:gh:`5925` by `Rémi Gau`_).

- :bdg-dark:`Code` Only throw warning about non-interactive plotting backend when using :func:`nilearn.plotting.show` (:gh:`5929` by `Rémi Gau`_).

- :bdg-info:`Plotting` drop background color when using look up table as colormap (:gh:`5936` by `Rémi Gau`_).

- :bdg-dark:`Code` Reallow use non-multi maskers for :class:`~nilearn.regions.Parcellations` (:gh:`5930` by `Rémi Gau`_).

- :bdg-info:`Plotting` Add support for dictionary as ``cut_coords`` for :class:`~plotting.displays.MosaicSlicer` and image plotting functions. (:gh:`5920` by `Hande Gözükan`_).

- :bdg-dark:`Code` Change default slice order of slicers and projectors to be x, y, z. (:gh:`5944` by `Hande Gözükan`_).

- :bdg-primary:`Doc` Fix broken scikit-learn contributing anchor in the docs to avoid linkcheck failures (:gh:`5983` by `Mohammad Sadeghi Hardengi`_).


Enhancements
------------


Changes
-------

- :bdg-danger:`Deprecation` The default for the parameter ``return_masked_atlas`` of :func:`~regions.img_to_signals_labels` to True. This deprecation was planned for 0.13.0 but missed. The parameter will be removed in version >= 0.15 (:gh:`5942` by `Rémi Gau`_).
