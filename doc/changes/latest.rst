.. currentmodule:: nilearn

.. include:: names.rst

0.10.4.dev
==========

Fixes
-----

- :bdg-dark:`Code` Fix plotting of carpet plot due to a change in the coming version of matplolib (3.9.0) (:gh:`4279` by `Rémi Gau`_).
- :bdg-dark:`Code` Fix errant warning when using ``stat_type`` in :func:`nilearn.glm.compute_contrast` (:gh:`4257` by `Eric Larson`_).
- :bdg-dark:`Code` Fix when thresholding is applied to images by GLM reports (:gh:`4258` by `Rémi Gau`_).
- :bdg-dark:`Code` Make masker report tables horizaontally scrollable (:gh:`4308` by `Rémi Gau`_).
- :bdg-dark:`Code` Masker reports tables do not overlap with report image (:gh:`4308` by `Rémi Gau`_).
- :bdg-dark:`Code` Fix color bar handling with color map with only 1 level (:gh:`4255` by `Rémi Gau`_).
- :bdg-dark:`Code` Fix positions of the markers on the images on the sphere masker reports (:gh:`4285` by `Rémi Gau`_).
- :bdg-dark:`Code` Make sure that :class:`nilearn.maskers.NiftiSpheresMasker` reports displays properly when it contains only 1 sphere (:gh:`4269` by `Rémi Gau`_).

Enhancements
------------

Changes
-------

- :bdg-primary:`Doc` Render several examples of GLM reports as part of the documentation (:gh:`4267` by `Rémi Gau`_).
- :bdg-dark:`Code` Improve colorbar size and labels in mosaic display (:gh:`4284` by `Rémi Gau`_).
- :bdg-primary:`Doc` Render the description of the templates, atlases and datasets of the :mod:`nilearn.datasets` as part of the documentation (:gh:`4232` by `Rémi Gau`_).
- :bdg-dark:`Code` Change the colormap to ``gray`` for the background image in the :class:`nilearn.maskers.NiftiSpheresMasker` (:gh:`4269` by `Rémi Gau`_).
- :bdg-dark:`Code` Remove unused ``**kwargs`` from :func:`nilearn.plotting.view_img` and :func:`nilearn.plotting.plot_surf` (:gh:`4270` by `Rémi Gau`_).
