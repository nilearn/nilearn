
.. currentmodule:: nilearn

.. include:: names.rst

0.8.2.dev
=========

.. warning::

 | **Python 3.6 is deprecated and will be removed in release 0.10.**
 | **We recommend upgrading to Python 3.9.**
 |
 | **Nibabel 2.x is no longer supported. Please consider upgrading to Nibabel >= 3.0.**
 |
 | **Minimum supported versions of packages have been bumped up:**
 | - Numpy -- v1.18
 | - SciPy -- v1.5
 | - Scikit-learn -- v0.22
 | - Pandas -- v1.0
 | - Joblib -- v0.15


NEW
---

- **Support for Python 3.6 is deprecated and will be removed in release 0.10.**
  Users with a Python 3.6 environment will be warned at their first Nilearn
  import and encouraged to update to more recent versions of Python.
- Masker objects like :class:`~nilearn.maskers.NiftiMasker` now belong to the
  new module :mod:`nilearn.maskers`. The old import style, through the module
  ``input_data``, still works but has been deprecated.
  (See PR `#3065 <https://github.com/nilearn/nilearn/pull/3065>`_).
- New module :mod:`nilearn.interfaces` to implement loading and saving utilities
  with various interfaces (fmriprep, bids...).
  (See PR `#3061 <https://github.com/nilearn/nilearn/pull/3061>`_).
- New submodule :mod:`nilearn.interfaces.fmriprep` to implement loading utilities
  for :term:`fMRIPrep`.
  (See PR `#3061 <https://github.com/nilearn/nilearn/pull/3061>`_).
- New function :func:`nilearn.interfaces.fmriprep.load_confounds` to load confound
  variables easily from :term:`fMRIPrep` outputs.
  (See PR `#2946 <https://github.com/nilearn/nilearn/pull/2946>`_).
- New function :func:`nilearn.interfaces.fmriprep.load_confounds_strategy` to load
  confound variables from :term:`fMRIPrep` outputs using four preset strategies:
  ``simple``, ``scrubbing``, ``compcor``, and ``ica_aroma``.
  (See PR `#3016 <https://github.com/nilearn/nilearn/pull/3016>`_).
- New submodule :mod:`nilearn.interfaces.bids` to implement loading utilities
  for :term:`BIDS` datasets.
  (See PR `#3126 <https://github.com/nilearn/nilearn/pull/3126>`_).
- New function :func:`nilearn.interfaces.bids.get_bids_files` to select files
  easily from :term:`BIDS` datasets.
  (See PR `#3126 <https://github.com/nilearn/nilearn/pull/3126>`_).
- New function :func:`nilearn.interfaces.bids.parse_bids_filename` to identify
  subparts of :term:`BIDS` filenames.
  (See PR `#3126 <https://github.com/nilearn/nilearn/pull/3126>`_).
- New submodule :mod:`nilearn.interfaces.fsl` to implement loading utilities
  for FSL outputs.
  (See PR `#3126 <https://github.com/nilearn/nilearn/pull/3126>`_).
- New function :func:`nilearn.interfaces.fsl.get_design_from_fslmat` to load
  design matrices from FSL files.
  (See PR `#3126 <https://github.com/nilearn/nilearn/pull/3126>`_).
- Surface plotting functions like :func:`nilearn.plotting.plot_surf_stat_map`
  now have an `engine` parameter, defaulting to "matplotlib", but which can be
  set to "plotly". If plotly and kaleido are installed, this will generate an
  interactive plot of the surface map using plotly instead of matplotlib.
  Note that this functionality is still experimental, and that some capabilities
  supported by our matplotlib engine are not yet supported by the plotly engine.
  (See PR `#2902 <https://github.com/nilearn/nilearn/pull/2902>`_).
- When using the `plotly` engine, surface plotting functions derived from
  :func:`~nilearn.plotting.plot_surf` return a new display object, a
  :class:`~nilearn.plotting.displays.PlotlySurfaceFigure`, which provides a
  similar interface to the :class:`~matplotlib.figure.Figure` returned with the
  `matplotlib` engine.
  (See PR `#3036 <https://github.com/nilearn/nilearn/pull/3036>`_).
- :class:`~nilearn.maskers.NiftiMapsMasker` can now generate HTML reports in the same
  way as :class:`~nilearn.maskers.NiftiMasker` and
  :class:`~nilearn.maskers.NiftiLabelsMasker`. The report enables the users to browse
  through the spatial maps with a previous and next button. The users can filter the maps
  they wish to display by passing an integer, or a list of integers to
  :meth:`~nilearn.maskers.NiftiMapsMasker.generate_report`.
- New class :class:`nilearn.regions.HierarchicalKMeans` which yields more
  balanced clusters than `KMeans`. It is also callable through
  :class:`nilearn.regions.Parcellations` using `method`=`hierarchical_kmeans`


Fixes
-----

- When a label image with non integer values was provided to the
  :class:`nilearn.maskers.NiftiLabelsMasker`, its `generate_report`
  method was raising an ``IndexError``.
  (See issue `#3007 <https://github.com/nilearn/nilearn/issues/3007>`_ and
  fix `#3009 <https://github.com/nilearn/nilearn/pull/3009>`_).
- :func:`nilearn.plotting.plot_markers` did not work when the `display_mode`
  parameter included `l` and `r` and the parameter `node_size` was provided
  as an array.
  (See issue `#3012 <https://github.com/nilearn/nilearn/issues/3012>`_) and fix
  `#3013 <https://github.com/nilearn/nilearn/pull/3013>`_).
- :meth:`nilearn.glm.first_level.FirstLevelModel.generate_report` threw a `TypeError`
  when `FirstLevelModel` was instantiated with `mask_img`
  being a :class:`~nilearn.maskers.NiftiMasker`.
  :func:`nilearn.reporting.make_glm_report` was fixed accordingly.
  (See issue `#3034 <https://github.com/nilearn/nilearn/issues/3034>`_) and fix
  `#3035 <https://github.com/nilearn/nilearn/pull/3035>`_).
- Function :func:`~nilearn.plotting.find_parcellation_cut_coords` now returns
  coordinates and labels having the same order as the one of the input labels
  index (See PR `#3078 <https://github.com/nilearn/nilearn/issues/3078>`_).
- Convert reference in `nilearn/regions/region_extractor.py` to use footcite / footbibliography.
  (See issue `#2787 <https://github.com/nilearn/nilearn/issues/2787>`_ and PR `#3111 <https://github.com/nilearn/nilearn/pull/3111>`_).
- Fixed Hommel value computation in `nilearn/glm/thresholding.py` used in the
  `cluster_level_inference` function. See PR `#3109 <https://github.com/nilearn/nilearn/pull/3109>`_
- Computation of Benjamini-Hocheberg threshold fixed in `nilearn/glm/thresholding.py` function (see issue `#2879 <https://github.com/nilearn/nilearn/issues/2879>`_ and PR `#3137 <https://github.com/nilearn/nilearn/pull/3137>`_)
- Attribute `scaling_axis` of :class:`~nilearn.glm.first_level.FirstLevelModel` has
  been deprecated and will be removed in 0.11.0. When scaling is performed, the
  attribute `signal_scaling` is used to define the axis instead.
  (See issue `#3134 <https://github.com/nilearn/nilearn/issues/3134>`_ and PR
  `#3135 <https://github.com/nilearn/nilearn/pull/3135>`_).

Enhancements
------------

- :func:`nilearn.image.threshold_img` accepts new parameters `cluster_threshold`
  and `two_sided`.
  `cluster_threshold` applies a cluster-size threshold (in voxels).
  `two_sided`, which is `True` by default, separately thresholds both positive
  and negative values in the map, as was done previously.
  When `two_sided` is `False`, only values greater than or equal to the threshold
  are retained.
- :func:`nilearn.signal.clean` raises a warning when the user sets
  parameters `detrend` and `standardize_confound` to False.
  The user is suggested to set one of
  those options to `True`, or standardize/demean the confounds before using the
  function.
- The :doc:`contributing documentation</development>` and
  :doc:`maintenance</maintenance>` pages were improved, especially towards ways
  of contributing to the project which do not require to write code.
  The roles of the :ref:`triage` were defined more clearly with sections on issue
  :ref:`issue_labels` and issue :ref:`closing_policy`.
  (See PR `#3010 <https://github.com/nilearn/nilearn/pull/3010>`_).
- It is now possible to provide custom :term:`HRF` models to
  :class:`nilearn.glm.first_level.FirstLevelModel`. The custom model should be
  defined as a function, or a list of functions, implementing the same API as
  Nilearn's usual models (see :func:`nilearn.glm.first_level.spm_hrf` for
  example). The example
  :ref:`sphx_glr_auto_examples_04_glm_first_level_plot_hrf.py` was
  also modified to demo how to define custom :term:`HRF` models.
  (See issue `#2940 <https://github.com/nilearn/nilearn/issues/2940>`_).
- :class:`nilearn.maskers.NiftiLabelsMasker` now gives a warning when some
  labels are removed from the label image at transform time due to resampling
  of the label image to the data image.
- Function :func:`~nilearn.glm.second_level.non_parametric_inference` now accepts
  :class:`~pandas.DataFrame` as possible values for its ``second_level_input``
  parameter. Note that a new parameter ``first_level_contrast`` has been added
  to this function to enable this feature.
  (See PR `#3042 <https://github.com/nilearn/nilearn/pull/3042>`_).
- Tests from `nilearn/plotting/tests/test_img_plotting.py` have been refactored
  and reorganized in separate files in new folder
  `nilearn/plotting/tests/test_img_plotting/`.
  (See PR `#3015 <https://github.com/nilearn/nilearn/pull/3015/files>`_)
- Once a :class:`~nilearn.glm.second_level.SecondLevelModel` has been fitted and
  contrasts have been computed, it is now possible to access the ``residuals``,
  ``predicted``, and ``r_square`` model attributes like it was already possible
  for :class:`~nilearn.glm.first_level.FirstLevelModel`.
  (See FR `#3027 <https://github.com/nilearn/nilearn/issues/3027>`_
  and PR `#3033 <https://github.com/nilearn/nilearn/pull/3033>`_)
- Importing :mod:`nilearn.plotting` will now raise a warning if the matplotlib
  backend has been changed from its original value, instead of silently modifying
  it.
- Function :func:`~nilearn.plotting.plot_img` and deriving functions like
  :func:`~nilearn.plotting.plot_anat`, :func:`~nilearn.plotting.plot_stat_map`, or
  :func:`~nilearn.plotting.plot_epi` now accept an optional argument
  ``cbar_tick_format`` to specify how numbers should be displayed on the colorbar.
  This is consistent with the API of surface plotting functions (see release 0.7.1).
  The default format is scientific notation.

Changes
-------

- Nibabel 2.x is no longer supported. Please consider upgrading to Nibabel >= 3.0.
  (See PR `#3106 <https://github.com/nilearn/nilearn/pull/3106>`_).
- Deprecated function ``nilearn.datasets.fetch_cobre`` has been removed.
  (See PR `#3081 <https://github.com/nilearn/nilearn/pull/3081>`_).
- Deprecated function ``nilearn.plotting.plot_connectome_strength`` has been removed.
  (See PR `#3082 <https://github.com/nilearn/nilearn/pull/3082>`_).
- Deprecated function ``nilearn.masking.compute_gray_matter_mask`` has been removed.
  (See PR `#3090 <https://github.com/nilearn/nilearn/pull/3090>`_).
- Deprecated parameter ``sessions`` of function :func:`~nilearn.signal.clean`
  has been removed. Use ``runs`` instead.
  (See PR `#3093 <https://github.com/nilearn/nilearn/pull/3093>`_).
- Deprecated parameters ``sessions`` and ``sample_mask`` of
  :class:`~nilearn.maskers.NiftiMasker` have been removed. Please use ``runs`` instead of
  ``sessions``, and provide a ``sample_mask`` through
  :meth:`~nilearn.maskers.NiftiMasker.transform`.
  (See PR `#3133 <https://github.com/nilearn/nilearn/pull/3133>`_).
- :func:`nilearn.glm.first_level.compute_regressor` will now raise an exception if
  parameter `cond_id` is not a string which could be used to name a python variable.
  For instance, number strings (ex: "1") will no longer be accepted as valid condition names.
  In particular, this will also impact
  :func:`nilearn.glm.first_level.make_first_level_design_matrix` and
  :class:`nilearn.glm.first_level.FirstLevelModel`, for which proper condition names
  will also be needed (see PR `#3025 <https://github.com/nilearn/nilearn/pull/3025>`_).
- Replace parameter `sessions` with `runs` in :func:`nilearn.image.clean_img` as this
  replacement was already made for :func:`nilearn.signal.clean` in
  `#2821 <https://github.com/nilearn/nilearn/pull/2821>`_ in order to match BIDS
  semantics. The use of `sessions` in :func:`nilearn.image.clean_img` is deprecated and
  will be removed in 0.10.0.
- Display objects have been reorganized. For example, Slicers (like the
  :class:`~nilearn.plotting.displays.OrthoSlicer`) are all in file
  `nilearn/plotting/displays/_slicers.py`, and Projectors (like the
  :class:`~nilearn.plotting.displays.OrthoProjector`) are all in file
  `nilearn/plotting/displays/_projectors.py`. All display objects have been added to
  the public API, and examples have been improved to show how to use these objects
  to customize figures obtained with plotting functions.
  (See PR `#3073 <https://github.com/nilearn/nilearn/pull/3073>`_).
- Descriptions of datasets retrieved with fetchers from :mod:`nilearn.datasets` are
  now python strings rather than `bytes`. Therefore, decoding the descriptions is no
  longer necessary.
