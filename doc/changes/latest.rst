.. currentmodule:: nilearn

.. include:: names.rst

0.11.2.dev
==========

NEW
---

Fixes
-----

- :bdg-dark:`Code` Fix ``two_sided`` image thresholding. (:gh:`4951` by `Hande Gözükan`_).

- :bdg-dark:`Code` Ensure that only valid surface meshes can be instantiated. (:gh:`5036` by `Rémi Gau`_).

Enhancements
------------

- :bdg-dark:`Code` Add reports for the surface based GLMs (:gh:`4442` by `Rémi Gau`_).

- :bdg-dark:`Code` Allow plotting both hemispheres together (:gh:`4991` by `Himanshu Aggarwal`_).

- :bdg-dark:`Code` Add a look up table to each of the deterministic atlas (:gh:`4820` by `Rémi Gau`_).
-
- :bdg-dark:`Code` Add a ``"template"`` to each atlas to describe the space they are provided in (:gh:`5041` by `Rémi Gau`_).

- :bdg-dark:`Code` Add an 'atlas_type' metadata to each atlas (:gh:`4820` by `Rémi Gau`_).

Changes
-------

- :bdg-dark:`Code` Fix labels of all deterministic atlases to be list of strings that contain a ``"Background"`` label (:gh:`4820`, :gh:`5006`, :gh:`5013`, :gh:`5041` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` Remove the ``legacy_format`` parameter from several dataset fetcher functions as it was due for deprecation in version 0.11.0  (:gh:`5004` by `Rémi Gau`_).

- :bdg-info:`Plotting` Change the default map to be ``"RdBu_r"`` or ``"gray"`` for most plotting functions. In several examples, use the "inferno" colormap when a sequential colormap is preferable (:gh:`4807` by `Rémi Gau`_).
