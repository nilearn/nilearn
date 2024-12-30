.. currentmodule:: nilearn

.. include:: names.rst

0.11.2.dev
==========

NEW
---

Fixes
-----

Enhancements
------------

- :bdg-dark:`Code` Allow plotting both hemispheres together (:gh:`4991` by `Himanshu Aggarwal`_).

Changes
-------

- :bdg-danger:`Deprecation` Remove the ``legacy_format`` parameter from several dataset fetcher functions as it was due for deprecation in version 0.11.0  (:gh:`5004` by `Rémi Gau`_).

- :bdg-info:`Plotting` Change the default map to be ``"RdBu_r"`` or ``"gray"`` for most plotting functions. In several examples, use the "inferno" colormap when a sequential colormap is preferable (:gh:`4807` by `Rémi Gau`_).

