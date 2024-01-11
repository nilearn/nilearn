.. currentmodule:: nilearn

.. include:: names.rst

0.11.0.dev
==========

HIGHLIGHTS
----------

NEW
---
- :bdg-info:`Plotting` The ``temp_file_lifetime`` parameter of interactive plots' ``open_in_browser`` method is deprecated and has no effect (:gh:`4180` by `Jerome Dockes`_)..

Fixes
-----

- :bdg-dark:`Code` Update the ``CompCor`` strategy in :func:`~interfaces.fmriprep.load_confounds` and :func:`~interfaces.fmriprep.load_confounds_strategy` to support ``fmriprep`` 21.x series and above. (:gh:`3285` by `Hao-Ting Wang`_).
- :bdg-success:`API` :class:`~maskers.MultiNiftiMasker` can now call :meth:`~maskers.NiftiMasker.generate_report` which will generate a report for the first subject in the list of subjects (:gh:`4001` by `Yasmin Mzayek`_).
- :bdg-dark:`Code` Fix :class:`~nilearn.glm.regression.SimpleRegressionResults` to accommodate for the lack of a ``model`` attribute (:gh:`4071` `Rémi Gau`_)
- :bdg-dark:`Code` :func:`~image.clean_img` can now use kwargs ``clean__sample_mask`` argument to correctly reshape the nifti image to the dimensions of the mask in the output (:gh:`4051` by `Mia Zwally`_).
- :bdg-dark:`Code` Fix plotting of an image with color bar when maximum value is exactly zero (:gh:`4204` by `Rémi Gau`_).

Enhancements
------------

- :bdg-primary:`Doc` Add backslash to homogenize :class:`~nilearn.regions.Parcellations` documentation (:gh:`4042` by `Nikhil Krish`_).
- :bdg-success:`API` Allow passing Pandas Series of image filenames to :class:`~nilearn.glm.second_level.SecondLevelModel` (:gh:`4070` by `Rémi Gau`_).
- :bdg-success:`API` Allow passing arguments to :func:`~nilearn.glm.first_level.first_level_from_bids` to build first level models that include specific set of confounds by relying on the strategies from :func:`~nilearn.interfaces.fmriprep.load_confounds` (:gh:`4103` by `Rémi Gau`_).
- :bdg-info:`Plotting` Allow setting ``vmin`` in :func:`~nilearn.plotting.plot_glass_brain` and :func:`~nilearn.plotting.plot_stat_map` (:gh:`3993` by `Michelle Wang`_).
- :bdg-success:`API` Support passing t and F contrasts to :func:`~nilearn.glm.compute_contrast` that that have fewer columns than the number of estimated parameters. Remaining columns are padded with zero (:gh:`4067` by `Rémi Gau`_).
- :bdg-dark:`Code` Multi-subject maskers' ``generate_report`` method no longer fails with 5D data but instead shows report of first subject. User can index input list to show report for different subjects (:gh:`3935` by `Yasmin Mzayek`_).
- :bdg-primary:`Code` Add ``two_sided`` option for :class:`~nilearn.image.binarize_img` (:gh:`4121` by `Steven Meisler`_).
- :bdg-dark:`Code` :meth:`~maskers.NiftiLabelsMasker.generate_report` now uses appropriate cut coordinates when functional image is provided (:gh:`4099` by `Yasmin Mzayek`_ and `Nicolas Gensollen`_).
- :bdg-info:`Plotting` When plotting thresholded statistical maps with a colorbar, the threshold value(s) will now be displayed as tick labels on the colorbar (:gh:`#2833` by `Nicolas Gensollen`_).
- :bdg-success:`API` :class:`~maskers.NiftiSpheresMasker` now has ``generate_report`` method (:gh:`3102` by `Yasmin Mzayek`_ and `Nicolas Gensollen`_).
- :bdg-primary:`Doc`  Mention the classification type (all-vs-one) in  :ref:`sphx_glr_auto_examples_02_decoding_plot_haxby_glm_decoding.py` (:gh:`4122` by `Tamer Gezici`_).
- :bdg-primary:`Doc`  Specify full form of LSS in  :ref:`sphx_glr_auto_examples_07_advanced_plot_beta_series.py` (:gh:`4141` by `Tamer Gezici`_).
- :bdg-primary:`Doc` Don't fetch tmaps in examples if tmaps aren't used in the example. (:gh:`4136` by `Christina Roßmanith`_).
- :bdg-primary:`Doc` Describe the return value in :func:`~nilearn.datasets.fetch_abide_pcp` documentation (:gh:`4159` by `Suramya Pokharel`_).

Changes
-------

- :bdg-dark:`Code` Make ``nilearn.reporting._get_clusters_table`` module public and move ``copy_img out`` of ``nilearn._utils.niimg`` (:gh:`4166` by `Rémi Gau`_).
- :bdg-danger:`Deprecation` :func:`~regions.img_to_signals_labels` will also return ``masked_atlas`` in release 0.15. Meanwhile, use ``return_masked_atlas`` parameter to enable/disable this behavior. (:gh:`3761` by `Mohammad Torabi`_).
- :bdg-success:`API` Expose scipy CubicSpline ``extrapolate`` parameter in :func:`~signal.clean` to control the interpolation of censored volumes in both ends of the BOLD signal data (:gh:`4028` by `Jordi Huguet`_).
- :bdg-secondary:`Maint` Switch to using tox to manage environments during development and testing. All plotting python dependencies (matplotlib AND plotly) are now installed when running ``pip install nilearn[plotting]`` (:gh:`4029` by `Rémi Gau`_).
- :bdg-dark:`Code` Private utility context manager ``write_tmp_imgs`` is refactored into function ``write_imgs_to_path`` (:gh:`4094` by `Yasmin Mzayek`_).
- :bdg-dark:`Code` Move user facing function ``concat_niimgs`` out of private module ``nilearn._utils.niimg_conversions`` (:gh:`4167` by `Rémi Gau`_).
- :bdg-danger:`Deprecation` Rename the parameter ``contrast_type`` in :func:`~glm.compute_contrast` and attribute ``contrast_type`` in  :class:`~glm.Contrast` to ``stat_type`` (:gh:`4191` by `Rémi Gau`_).
- :bdg-danger:`Deprecation` :func:`~plotting.plot_surf_roi` will raise a warning if ``roi_map`` contains negative or non-integer values; in version 0.13 this will be a ``ValueError`` (:gh:`4131` by `Michelle Wang`_).
- :bdg-dark:`Code` Remove leading underscore from non private functions to align with PEP8 (:gh:`4086` by `Rémi Gau`_).
- :bdg-dark:`Code` Make ``decoding/proximal_operator`` explicitly private to align with PEP8 (:gh:`4153` by `Rémi Gau`_).
- :bdg-dark:`Code` Make private functions public when used outside of their module ``nilearn.interface`` to align with PEP8 (:gh:`4168` by `Rémi Gau`_).
