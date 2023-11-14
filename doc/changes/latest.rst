.. currentmodule:: nilearn

.. include:: names.rst

0.11.0.dev
==========

HIGHLIGHTS
----------

NEW
---

Fixes
-----

- :bdg-success:`API` :class:`~maskers.MultiNiftiMasker` can now call :meth:`~maskers.NiftiMasker.generate_report` which will generate a report for the first subject in the list of subjects (:gh:`4001` by `Yasmin Mzayek`_).
- :bdg-dark:`Code` :func:`~image.clean_img` can now use kwargs ``clean__sample_mask`` argument to correctly reshape the nifti image to the dimensions of the mask in the output (:gh:`4051` by 'Mia Zwally`_).

Enhancements
------------

- :bdg-primary:`Doc` Add backslash to homogenize :class:`~nilearn.regions.Parcellations` documentation (:gh:`4042` by `Nikhil Krish`_).
- :bdg-success:`API` Allow passing Pandas Series of image filenames to :class:`~nilearn.glm.second_level.SecondLevelModel` (:gh:`4070` by `Rémi Gau`_).
- :bdg-info:`Plotting` Allow setting ``vmin`` in :func:`~nilearn.plotting.plot_glass_brain` and :func:`~nilearn.plotting.plot_stat_map` (:gh:`3993` by `Michelle Wang`_).
- :bdg-success:`API` Support passing t and F contrasts to :func:`~nilearn.glm.compute_contrast` that that have fewer columns than the number of estimated parameters. Remaining columns are padded with zero (:gh:`4067` by `Rémi Gau`_).

Changes
-------

- :bdg-danger:`Deprecation` :func:`~regions.img_to_signals_labels` will also return ``masked_atlas`` in release 0.15. Meanwhile, use ``return_masked_atlas`` parameter to enable/disable this behavior. (:gh:`3761` by `Mohammad Torabi`_).
- :bdg-success:`API` Expose scipy CubicSpline ``extrapolate`` parameter in :func:`~signal.clean` to control the interpolation of censored volumes in both ends of the BOLD signal data (:gh:`4028` by `Jordi Huguet`_).
- :bdg-dark:`Code` Private utility context manager ``write_tmp_imgs`` is refactored into function ``write_imgs_to_path`` (:gh:`4094` by `Yasmin Mzayek`_).
