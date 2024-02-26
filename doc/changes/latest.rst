.. currentmodule:: nilearn

.. include:: names.rst

0.10.4.dev
==========

Fixes
-----

- :bdg-dark:`Code` Fix plotting of carpet plot due to a change in the coming version of matplolib (3.9.0) (:gh:`4279` by `Rémi Gau`_).
- :bdg-dark:`Code` Fix errant warning when using ``stat_type`` in :func:`nilearn.glm.compute_contrast` (:gh:`4257` by `Eric Larson`_).
- :bdg-dark:`Code` FIX when thresholding is applied to images by GLM reports (:gh:`4258` by `Rémi Gau`_).

Enhancements
------------

Changes
-------

- :bdg-primary:`Doc` Render the description of the templates, atlases and datasets of the :mod:`nilearn.datasets` as part of the documentation (:gh:`4232` by `Rémi Gau`_).
