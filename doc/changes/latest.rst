.. currentmodule:: nilearn

.. include:: names.rst

0.11.0.dev
==========

Fixes
-----

- :bdg-dark:`Code` Fix errant warning when using ``stat_type`` in :func:`nilearn.glm.compute_contrast` (:gh:`4257` by `Eric Larson`_).
- :bdg-dark:`Code` Fix when thresholding is applied to images by GLM reports (:gh:`4258` by `Rémi Gau`_).
- :bdg-dark:`Code` Make sure that :class:`nilearn.maskers.NiftiSpheresMasker` reports displays properly when it contains only 1 sphere (:gh:`4269` by `Rémi Gau`_).

Enhancements
------------

Changes
-------

- :bdg-primary:`Doc` Render the description of the templates, atlases and datasets of the :mod:`nilearn.datasets` as part of the documentation (:gh:`4232` by `Rémi Gau`_).
- :bdg-dark:`Code` Change the colormap to ``gray`` for the background image in the :class:`nilearn.maskers.NiftiSpheresMasker` (:gh:`4269` by `Rémi Gau`_).
