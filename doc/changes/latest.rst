.. currentmodule:: nilearn

.. include:: names.rst

0.11.2.dev
==========

HIGHLIGHTS
----------

- Fixes the behavior of :class:`nilearn.decoding.Decoder` where it used to set the score to zero if all coefficients are zero, irrespective of the scoring metric used. This change ensures that the average cross-validated scores are comparable to a pure scikit-learn implementation. (:gh:`5097`).

Fixes
-----

- :bdg-dark:`Code` Fixes datasets that returned numpy arrays instead of pandas dataframes (:gh:`5109` by `Rémi Gau`_).

- :bdg-dark:`Code` Ensure that image plotting threshold is non-negative as one-sided behavior in image thresholding can be obtained with vmin and vmax values in plotting threshold. (:gh:`5088` by `Hande Gözükan`_).

- :bdg-dark:`Code` Allow one-sided image thresholding by introducing ``two_sided`` parameter to image thresholding and update behavior of negative threshold. (:gh:`4951` by `Hande Gözükan`_).

- :bdg-dark:`Code` Ensure that only valid surface meshes can be instantiated. (:gh:`5036` by `Rémi Gau`_).

- :bdg-dark:`Code` Do not set score to zero if all coefficients are zero in :class:`nilearn.decoding.Decoder` and ensure average scores are comparable to a pure scikit-learn implementation (:gh:`5097` by `Himanshu Aggarwal`_).

Enhancements
------------

- :bdg-dark:`Code` Add different reduction strategies to :class:`nilearn.maskers.SurfaceLabelsMasker` (:gh:`4809` by `Rémi Gau`_).

- :bdg-info:`Plotting` Add a :func:`~nilearn.plotting.img_comparison.plot_bland_altman` to create Bland-Altman plots to compare images (:gh:`5112` by `Rémi Gau`_).

- :bdg-dark:`Code` Add reports for the surface based GLMs (:gh:`4442` by `Rémi Gau`_).

- :bdg-dark:`Code` Enhance :func:`~nilearn.glm.second_level.non_parametric_inference` to support surface data. Please, note that cluster analysis, TFCE and smoothing are not yet implemented. (:gh:`5078` by `Rémi Gau`_).

- :bdg-dark:`Code` Allow plotting both hemispheres together (:gh:`4991` by `Himanshu Aggarwal`_).

- :bdg-dark:`Plotting` Colormaps can be passed as BIDS compliant look-up table via a :class:`pandas.DataFrame` to :func:`~nilearn.plotting.plot_roi` and :func:`~nilearn.plotting.plot_surf_roi` (:gh:`5160` by `Rémi Gau`_).
-
- :bdg-dark:`Code` Add a BIDS compliant look-up table to each of the deterministic atlas (:gh:`4820` by `Rémi Gau`_).
-
- :bdg-dark:`Code` Add a ``"template"`` to each atlas to describe the space they are provided in (:gh:`5041` by `Rémi Gau`_).

- :bdg-dark:`Code` Add an ``"atlas_type"`` metadata to each atlas (:gh:`4820` by `Rémi Gau`_).

- :bdg-dark:`Code` Add ``n_networks`` and ``thickness`` parameters to :func:`nilearn.datasets.fetch_atlas_yeo_2011` to specify which parcellation should be returned :gh:`5085` by `Rémi Gau`_).

- :bdg-dark:`Code` Add reports for SurfaceMapsMasker (:gh:`4968` by `Himanshu Aggarwal`_).

Changes
-------

- :bdg-info:`Deprecation` Add a ``nilearn.plotting.img_plotting.plot_img_comparison`` was moved to ``nilearn.plotting.img_comparison.plot_img_comparison`` (:gh:`5120` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` From version 0.13.2, :func:`nilearn.datasets.fetch_atlas_yeo_2011` will return a single parcellation (:gh:`5085` by `Rémi Gau`_).

- :bdg-dark:`Code` Fix labels of all deterministic atlases to be list of strings that contain a ``"Background"`` label (:gh:`4820`, :gh:`5006`, :gh:`5013`, :gh:`5041` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` Remove the ``legacy_format`` parameter from several dataset fetcher functions as it was due for deprecation in version 0.11.0  (:gh:`5004` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` Deprecate passing cleaning arguments to maskers for NiftiImages via ``kwargs``. Introduce a ``clean_args`` parameter to match API of Surface maskers.  (:gh:`5082` by `Rémi Gau`_).

- :bdg-info:`Plotting` Change the default map to be ``"RdBu_r"`` or ``"gray"`` for most plotting functions. In several examples, use the "inferno" colormap when a sequential colormap is preferable (:gh:`4807` by `Rémi Gau`_).

- :bdg-info:`Plotting` Improve sulci and subcortical schema for glass brain sagittal plots (:gh:`4807` by `John T. Johnson`_).
