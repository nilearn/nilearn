0.8.2.dev
=========

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
- Computation of Benjamini-Hocheberg threshold fixed in `nilearn/glm/thresholding.py` function (see issue `#2879 <https://github.com/nilearn/nilearn/issues/2879>`_ and PR `#3137 <https://github.com/nilearn/nilearn/pull/3137>`_)


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

.. _v0.8.1:

0.8.1
=====
**Released September 2021**

HIGHLIGHTS
----------

- New atlas fetcher
  :func:`nilearn.datasets.fetch_atlas_juelich` to download Juelich atlas from FSL.
- New grey and white-matter template and mask loading functions:
  :func:`nilearn.datasets.load_mni152_gm_template`,
  :func:`nilearn.datasets.load_mni152_wm_template`,
  :func:`nilearn.datasets.load_mni152_gm_mask`, and
  :func:`nilearn.datasets.load_mni152_wm_mask`
- :ref:`development_process` has been reworked. It now provides insights on
  nilearn organization as a project as well as more explicit
  :ref:`contribution_guidelines`.
- :func:`nilearn.image.binarize_img` binarizes images into 0 and 1.

NEW
---
- New atlas fetcher
  :func:`nilearn.datasets.fetch_atlas_juelich` to download Juelich atlas from FSL.
- :ref:`development_process` has been reworked. It now provides insights on
  nilearn organization as a project as well as more explicit
  :ref:`contribution_guidelines`.
- :func:`nilearn.datasets.load_mni152_gm_template` takes the skullstripped
  1mm-resolution version of the grey-matter MNI152 template and re-samples it
  using a different resolution, if specified.
- :func:`nilearn.datasets.load_mni152_wm_template` takes the skullstripped
  1mm-resolution version of the white-matter MNI152 template and re-samples it
  using a different resolution, if specified.
- :func:`nilearn.datasets.load_mni152_gm_mask` loads mask from the grey-matter
  MNI152 template.
- :func:`nilearn.datasets.load_mni152_wm_mask` loads mask from the white-matter
  MNI152 template.
- :func:`nilearn.image.binarize_img` binarizes images into 0 and 1.
- :class:`nilearn.maskers.NiftiMasker`,
  :class:`nilearn.maskers.MultiNiftiMasker`, and objects relying on such maskers
  (:class:`nilearn.decoding.Decoder` or :class:`nilearn.decomposition.CanICA`
  for example) can now use new options for the argument `mask_strategy`:
  `whole-brain-template` for whole-brain template (same as previous option
  `template`), `gm-template` for grey-matter template, and `wm-template`
  for white-matter template.

Fixes
-----

- :func:`nilearn.masking.compute_multi_brain_mask` has replaced
  nilearn.masking.compute_multi_grey_matter_mask. A mask parameter has been added;
  it accepts three types of masks---i.e. whole-brain, grey-matter and
  white-matter---following the enhancements made in the function
  :func:`nilearn.masking.compute_brain_mask` in this release.
- Fix colorbar of :func:`nilearn.plotting.view_img` which was not visible for some
  combinations of `black_bg` and `bg_img` parameters.
  (See issue `#2874 <https://github.com/nilearn/nilearn/issues/2874>`_).
- Fix missing title with :func:`nilearn.plotting.plot_surf` and
  deriving functions.
  (See issue `#2941 <https://github.com/nilearn/nilearn/issues/2941>`_).

Enhancements
------------

- :func:`nilearn.datasets.load_mni152_template` resamples now the template to
  a preset resolution different from the resolution of the original template,
  i.e. 1mm. The default resolution is 2mm, which means that the new template is
  resampled to the resolution of the old template. Nevertheless, the shape of
  the template changed from (91, 109, 91) to (99, 117, 95); the affine also
  changed from array([[-2., 0., 0., 90.], [0., 2., 0., -126.],
  [0., 0., 2., -72.], [0., 0., 0., 1.]]) to array([[1., 0., 0., -98.],
  [0., 1., 0., -134.], [0., 0., 1., -72.], [0., 0., 0., 1.]]). Additionally,
  the new template has also been rescaled; whereas the old one varied between
  0 and 8339, the new one varies between 0 and 255.
- :func:`nilearn.datasets.load_mni152_brain_mask` accepts now the parameter
  resolution, which will set the resolution of the template used for the
  masking.
- :func:`nilearn.masking.compute_brain_mask` accepts now as input the
  whole-brain, 1mm-resolution, MNI152 T1 template instead of the averaged,
  whole-brain, 2mm-resolution MNI152 T1 template; it also accepts as input the
  grey-matter and white-matter ICBM152 1mm-resolution templates dated from 2009.
- Common parts of docstrings across Nilearn can now be filled automatically using
  the decorator `nilearn._utils.fill_doc`. This can be applied to common function
  parameters or common lists of options for example. The standard parts are defined
  in a single location (`nilearn._utils.docs.py`) which makes them easier to
  maintain and update. (See `#2875 <https://github.com/nilearn/nilearn/pull/2875>`_)
- The `data_dir` argument can now be either a `pathlib.Path` or a string. This
  extension affects datasets and atlas fetchers.

Changes
-------

- The version of the script `jquery.min.js` was bumped from 3.3.1 to 3.6.0 due
  to potential vulnerability issues with versions < 3.5.0.

.. _v0.8.0:

0.8.0
=====

**Released June 2021**

HIGHLIGHTS
----------

.. warning::

 | **Python 3.5 is no longer supported. We recommend upgrading to Python 3.8.**
 |
 | **Support for Nibabel 2.x is deprecated and will be removed in the 0.9 release.**
 | Users with a version of Nibabel < 3.0 will be warned at their first Nilearn import.
 |
 | **Minimum supported versions of packages have been bumped up:**
 | - Numpy -- v1.16
 | - SciPy -- v1.2
 | - Scikit-learn -- v0.21
 | - Nibabel -- v2.5
 | - Pandas -- v0.24

- :class:`nilearn.maskers.NiftiLabelsMasker` can now generate HTML reports in the same
  way as :class:`nilearn.maskers.NiftiMasker`.
- :func:`nilearn.signal.clean` accepts new parameter `sample_mask`.
  shape: (number of scans - number of volumes removed, )
- All inherent classes of `nilearn.maskers.BaseMasker` can use parameter `sample_mask`
  for sub-sample masking.
- Fetcher :func:`nilearn.datasets.fetch_surf_fsaverage` now accepts `fsaverage3`,
  `fsaverage4` and `fsaverage6` as values for parameter `mesh`, so that
  all resolutions of fsaverage from 3 to 7 are now available.
- Fetcher :func:`nilearn.datasets.fetch_surf_fsaverage` now provides attributes
  `{area, curv, sphere, thick}_{left, right}` for all fsaverage resolutions.
- :class:`nilearn.glm.first_level.run_glm` now allows auto regressive noise
  models of order greater than one.

NEW
---

- :func:`nilearn.signal.clean` accepts new parameter `sample_mask`.
  shape: (number of scans - number of volumes removed, )
  Masks the niimgs along time/fourth dimension to perform scrubbing (remove
  volumes with high motion) and/or non-steady-state volumes. Masking is applied
  before signal cleaning.
- All inherent classes of `nilearn.maskers.BaseMasker` can use
  parameter `sample_mask` for sub-sample masking.
- :class:`nilearn.maskers.NiftiLabelsMasker` can now generate HTML reports in the same
  way as :class:`nilearn.maskers.NiftiMasker`. The report shows the regions defined by
  the provided label image and provide summary statistics on each region (name, volume...).
  If a functional image was provided to fit, the middle image is plotted with the regions
  overlaid as contours. Finally, if a mask is provided, its contours are shown in green.

Fixes
-----

- Convert references in signal.py, atlas.py, func.py, neurovault.py, and struct.py
  to use footcite / footbibliography.
- Fix detrending and temporal filtering order for confounders
  in :func:`nilearn.signal.clean`, so that these operations are applied
  in the same order as for the signals, i.e., first detrending and
  then temporal filtering (https://github.com/nilearn/nilearn/issues/2730).
- Fix number of attributes returned by the
  `nilearn.glm.first_level.FirstLevelModel._get_voxelwise_model_attribute` method
  in the first level model. It used to return only the first attribute, and now returns
  as many attributes as design matrices.
- Plotting functions that show a stack of slices from a 3D image (e.g.
  :func:`nilearn.plotting.plot_stat_map`) will now plot the slices in the user
  specified order, rather than automatically sorting into ascending order
  (https://github.com/nilearn/nilearn/issues/1155).
- Fix the axes zoom on plot_img_on_surf function so brain would not be cutoff, and
  edited function so less white space surrounds brain views & smaller colorbar using
  gridspec (https://github.com/nilearn/nilearn/pull/2798).
- Fix inconsistency in prediction values of Dummy Classifier for Decoder
  object (https://github.com/nilearn/nilearn/issues/2767).

Enhancements
------------

- :func:`nilearn.plotting.view_markers` now accepts an optional argument `marker_labels`
  to provide labels to each marker.
- :func:`nilearn.plotting.plot_surf` now accepts new values for `avg_method` argument,
  such as `min`, `max`, or even a custom python function to compute the value displayed
  for each face of the plotted mesh.
- :func:`nilearn.plotting.view_img_on_surf` can now optionally pass through
  parameters to :func:`nilearn.surface.vol_to_surf` using the
  `vol_to_surf_kwargs` argument. One application is better HTML visualization of atlases.
  (https://nilearn.github.io/auto_examples/01_plotting/plot_3d_map_to_surface_projection.html)
- :func:`nilearn.plotting.view_connectome` now accepts an optional argument `node_color`
  to provide a single color for all nodes, or one color per node.
  It defaults to `auto` which colors markers according to the viridis colormap.
- Refactor :func:`nilearn.signal.clean` to clarify the data flow.
  Replace `sessions` with `runs` to matching BIDS semantics and deprecate `sessions` in 0.9.0.
  Add argument `filter` and allow a selection of signal filtering strategies:
  * "butterwoth" (butterworth filter)
  * "cosine" (discrete cosine transformation)
  * `False` (no filtering)
- Change the default strategy for Dummy Classifier from 'prior' to
  'stratified' (https://github.com/nilearn/nilearn/pull/2826/).
- :class:`nilearn.glm.first_level.run_glm` now allows auto regressive noise
  models of order greater than one.
- Moves parameter `sample_mask` from :class:`nilearn.maskers.NiftiMasker`
  to method `transform` in base class `nilearn.maskers.BaseMasker`.
- Fetcher :func:`nilearn.datasets.fetch_surf_fsaverage` now accepts
  `fsaverage3`, `fsaverage4` and `fsaverage6` as values for parameter `mesh`, so that
  all resolutions of fsaverage from 3 to 7 are now available.
- Fetcher :func:`nilearn.datasets.fetch_surf_fsaverage` now provides
  attributes `{area, curv, sphere, thick}_{left, right}` for all fsaverage
  resolutions.

Changes
-------

- Python 3.5 is no longer supported. We recommend upgrading to Python 3.7.
- Support for Nibabel 2.x is now deprecated and will be removed
  in the 0.9 release. Users with a version of Nibabel < 3.0 will
  be warned at their first Nilearn import.
- Minimum supported versions of packages have been bumped up:

    * Numpy -- v1.16
    * SciPy -- v1.2
    * Scikit-learn -- v0.21
    * Nibabel -- v2.5
    * Pandas -- v0.24

- Function sym_to_vec from :mod:`nilearn.connectome` was deprecated since release 0.4 and
  has been removed.
- Fetcher `nilearn.datasets.fetch_nyu_rest` is deprecated since release 0.6.2 and
  has been removed.
- :class:`nilearn.maskers.NiftiMasker` replaces `sessions` with `runs` and
  deprecates attribute `sessions` in 0.9.0. Match the relevant change in
  :func:`nilearn.signal.clean`.

.. _v0.7.1:

0.7.1
=====

**Released March 2021**

HIGHLIGHTS
----------

- New atlas fetcher
  :func:`nilearn.datasets.fetch_atlas_difumo` to download *Dictionaries of Functional Modes*,
  or “DiFuMo”, that can serve as atlases to extract functional signals with different
  dimensionalities (64, 128, 256, 512, and 1024). These modes are optimized to represent well
  raw BOLD timeseries, over a with range of experimental conditions.

- :class:`nilearn.decoding.Decoder` and :class:`nilearn.decoding.DecoderRegressor`
  is now implemented with random predictions to estimate a chance level.

- The functions :func:`nilearn.plotting.plot_epi`, :func:`nilearn.plotting.plot_roi`,
  :func:`nilearn.plotting.plot_stat_map`, :func:`nilearn.plotting.plot_prob_atlas`
  is now implemented with new display mode Mosaic. That implies plotting 3D maps
  in multiple columns and rows in a single axes.

- :func:`nilearn.plotting.plot_carpet` now supports discrete atlases.
  When an atlas is used, a colorbar is added to the figure,
  optionally with labels corresponding to the different values in the atlas.

NEW
---

- New atlas fetcher
  :func:`nilearn.datasets.fetch_atlas_difumo` to download *Dictionaries of Functional Modes*,
  or “DiFuMo”, that can serve as atlases to extract functional signals with different
  dimensionalities (64, 128, 256, 512, and 1024). These modes are optimized to represent well
  raw BOLD timeseries, over a with range of experimental conditions.

- :func:`nilearn.glm.Contrast.one_minus_pvalue` was added to ensure numerical
  stability of p-value estimation. It computes 1 - p-value using the Cumulative
  Distribution Function in the same way as `nilearn.glm.Contrast.p_value`
  computes the p-value using the Survival Function.

Fixes
-----

- Fix altered, non-zero baseline in design matrices where multiple events in the same condition
  end at the same time (https://github.com/nilearn/nilearn/issues/2674).

- Fix testing issues on ARM machine.

Enhancements
------------

- :class:`nilearn.decoding.Decoder` and :class:`nilearn.decoding.DecoderRegressor`
  is now implemented with random predictions to estimate a chance level.

- :class:`nilearn.decoding.Decoder`, :class:`nilearn.decoding.DecoderRegressor`,
  :class:`nilearn.decoding.FREMRegressor`, and :class:`nilearn.decoding.FREMClassifier`
  now override the `score` method to use whatever scoring strategy was defined through
  the `scoring` attribute instead of the sklearn default.
  If the `scoring` attribute of the decoder is set to None, the scoring strategy
  will default to accuracy for classifiers and to r2 score for regressors.

- :func:`nilearn.plotting.plot_surf` and deriving functions like :func:`nilearn.plotting.plot_surf_roi`
  now accept an optional argument `cbar_tick_format` to specify how numbers should be displayed on the
  colorbar of surface plots. The default format is scientific notation except for :func:`nilearn.plotting.plot_surf_roi`
  for which it is set as integers.

- :func:`nilearn.plotting.plot_carpet` now supports discrete atlases.
  When an atlas is used, a colorbar is added to the figure,
  optionally with labels corresponding to the different values in the atlas.

- :class:`nilearn.maskers.NiftiMasker`, :class:`nilearn.maskers.NiftiLabelsMasker`,
  :class:`nilearn.maskers.MultiNiftiMasker`, :class:`nilearn.maskers.NiftiMapsMasker`,
  and :class:`nilearn.maskers.NiftiSpheresMasker` can now compute high variance confounds
  on the images provided to `transform` and regress them out automatically. This behaviour is
  controlled through the `high_variance_confounds` boolean parameter of these maskers which
  default to False.

- :class:`nilearn.maskers.NiftiLabelsMasker` now automatically replaces NaNs in input data
  with zeros, to match the behavior of other maskers.

- :func:`nilearn.datasets.fetch_neurovault` now implements a `resample` boolean argument to either
  perform a fixed resampling during download or keep original images. This can be handy to reduce disk usage.
  By default, the downloaded images are not resampled.

- The functions :func:`nilearn.plotting.plot_epi`, :func:`nilearn.plotting.plot_roi`,
  :func:`nilearn.plotting.plot_stat_map`, :func:`nilearn.plotting.plot_prob_atlas`
  is now implemented with new display mode Mosaic. That implies plotting 3D maps
  in multiple columns and rows in a single axes.

- `psc` standardization option of :func:`nilearn.signal.clean` now allows time series with negative mean values.

- :func:`nilearn.reporting.make_glm_report` and
  :func:`nilearn.reporting.get_clusters_table` have a new argument,
  "two_sided", which allows for two-sided thresholding, which is disabled by default.

.. _v0.7.0:

0.7.0
=====

**Released November 2020**

HIGHLIGHTS
----------

- Nilearn now includes the functionality of `Nistats <https://nistats.github.io>`_ as :mod:`nilearn.glm`. This module is experimental, hence subject to change in any future release.
  :ref:`Here's a guide to replacing Nistats imports to work in Nilearn. <nistats_migration>`
- New decoder object
  :class:`nilearn.decoding.Decoder` (for classification) and
  :class:`nilearn.decoding.DecoderRegressor` (for regression) implement a model
  selection scheme that averages the best models within a cross validation loop.
- New FREM object
  :class:`nilearn.decoding.FREMClassifier` (for classification) and
  :class:`nilearn.decoding.FREMRegressor` (for regression) extend the decoder
  object with one fast clustering step at the beginning and  aggregates a high number of estimators trained on various splits of the training set.

- New plotting functions:

  * :func:`nilearn.plotting.plot_event` to visualize events file.
  * :func:`nilearn.plotting.plot_roi` can now plot ROIs in contours with `view_type` argument.
  * :func:`nilearn.plotting.plot_carpet` generates a "carpet plot" (also known
    as a "Power plot" or a "grayplot")
  * :func:`nilearn.plotting.plot_img_on_surf` generates multiple views of
    :func:`nilearn.plotting.plot_surf_stat_map` in a single figure.
  * :func:`nilearn.plotting.plot_markers` shows network nodes (markers) on a glass
    brain template
  * :func:`nilearn.plotting.plot_surf_contours` plots the contours of regions of
    interest on the surface

.. warning::

  Minimum required version of Joblib is now 0.12.


NEW
---
- Nilearn now includes the functionality of `Nistats <https://nistats.github.io>`_.
  :ref:`Here's a guide to replacing Nistats imports to work in Nilearn. <nistats_migration>`
- New decoder object
  :class:`nilearn.decoding.Decoder` (for classification) and
  :class:`nilearn.decoding.DecoderRegressor` (for regression) implement a model
  selection scheme that averages the best models within a cross validation loop.
  The resulting average model is the one used as a classifier or a regressor.
  These two objects also leverage the `NiftiMaskers` to provide a direct
  interface with the Nifti files on disk.
- New FREM object
  :class:`nilearn.decoding.FREMClassifier` (for classification) and
  :class:`nilearn.decoding.FREMRegressor` (for regression) extend the decoder
  object pipeline with one fast clustering step at the beginning (yielding an
  implicit spatial regularization) and  aggregates a high number of estimators
  trained on various splits of the training set. This returns a state-of-the-art
  decoding pipeline at a low computational cost.
  These two objects also leverage the `NiftiMaskers` to provide a direct
  interface with the Nifti files on disk.
- Plot events file
  Use :func:`nilearn.plotting.plot_event` to visualize events file.
  The function accepts the :term:`BIDS` events file read using `pandas`
  utilities.
- Plotting function :func:`nilearn.plotting.plot_roi` can now plot ROIs
  in contours with `view_type` argument.
- New plotting function
  :func:`nilearn.plotting.plot_carpet` generates a "carpet plot" (also known
  as a "Power plot" or a "grayplot"), for visualizing global patterns in
  4D functional data over time.
- New plotting function
  :func:`nilearn.plotting.plot_img_on_surf` generates multiple views of
  :func:`nilearn.plotting.plot_surf_stat_map` in a single figure.
- :func:`nilearn.plotting.plot_markers` shows network nodes (markers) on a glass
  brain template and color code them according to provided nodal measure (i.e.
  connection strength). This function will replace function
  ``nilearn.plotting.plot_connectome_strength``.
- New plotting function
  :func:`nilearn.plotting.plot_surf_contours` plots the contours of regions of
  interest on the surface, optionally overlaid on top of a statistical map.
- The position annotation on the plot methods now implements the `decimals` option
  to enable annotation of a slice coordinate position with the float.
- New example in
  :ref:`sphx_glr_auto_examples_02_decoding_plot_haxby_searchlight_surface.py`
  to demo how to do cortical surface-based searchlight decoding with Nilearn.
- confounds or additional regressors for design matrix can be specified as
  numpy arrays or pandas DataFrames interchangeably
- The decomposition estimators will now accept argument `per_component`
  with `score` method to explain the variance for each component.


Fixes
-----

- :class:`nilearn.maskers.NiftiLabelsMasker` no longer ignores its `mask_img`
- :func:`nilearn.masking.compute_brain_mask` has replaced
  nilearn.masking.compute_gray_matter_mask. Features remained the same but
  some corrections regarding its description were made in the docstring.
- the default background (MNI template) in plotting functions now has the
  correct orientation; before left and right were inverted.
- first level modelling can deal with regressors
  having multiple events which share onsets or offsets.
  Previously, such cases could lead to an erroneous baseline shift.
- :func:`nilearn.mass_univariate.permuted_ols` no longer returns transposed
  t-statistic arrays when no permutations are performed.
- Fix decomposition estimators returning explained variance score as 0.
  based on all components i.e., when per_component=False.
- Fix readme file of the Destrieux 2009 atlas.


Changes
-------

- Function ``nilearn.datasets.fetch_cobre`` has been deprecated and will be
  removed in release 0.9 .
- Function ``nilearn.plotting.plot_connectome_strength`` has been deprecated and will
  be removed in release 0.9 .

- :class:`nilearn.connectome.ConnectivityMeasure` can now remove
  confounds in its transform step.
- :func:`nilearn.surface.vol_to_surf` can now sample between two nested surfaces
  (eg white matter and pial surfaces) at specific cortical depths
- :func:`nilearn.datasets.fetch_surf_fsaverage` now also downloads white matter
  surfaces.


.. _v0.6.2:

0.6.2
======

ENHANCEMENTS
------------

- Generated documentation now includes Binder links to launch examples interactively
  in the browser
- :class:`nilearn.maskers.NiftiSpheresMasker` now has an inverse transform,
  projecting spheres to the corresponding mask_img.

Fixes
-----

- More robust matplotlib backend selection
- Typo in example fixed

Changes
-------

- Atlas `nilearn.datasets.fetch_nyu_rest` has been deprecated and will be removed in Nilearn 0.8.0 .

Contributors
------------

The following people contributed to this release::

     Elizabeth DuPre
     Franz Liem
     Gael Varoquaux
     Jon Haitz Legarreta Gorroño
     Joshua Teves
     Kshitij Chawla (kchawla-pi)
     Zvi Baratz
     Simon R. Steinkamp


.. _v0.6.1:

0.6.1
=====

ENHANCEMENTS
------------

- html pages use the user-provided plot title, if any, as their title

Fixes
-----

- Fetchers for developmental_fmri and localizer datasets resolve URLs correctly.

Contributors
------------

The following people contributed to this release::

     Elizabeth DuPre
     Jerome Dockes
     Kshitij Chawla (kchawla-pi)

0.6.0
=====

**Released December 2019**

HIGHLIGHTS
----------

.. warning::

 | **Python2 and 3.4 are no longer supported. We recommend upgrading to Python 3.6 minimum.**
 |
 | **Support for Python3.5 will be removed in the 0.7.x release.**
 | Users with a Python3.5 environment will be warned at their first Nilearn import.
 |
 | **joblib is now a dependency**
 |
 | **Minimum supported versions of packages have been bumped up.**
 | - Matplotlib -- v2.0
 | - Scikit-learn -- v0.19
 | - Scipy -- v0.19

NEW
---

- A new method for :class:`nilearn.maskers.NiftiMasker` instances
  for generating reports viewable in a web browser, Jupyter Notebook, or VSCode.

- A new function :func:`nilearn.image.get_data` to replace the deprecated
  nibabel method `Nifti1Image.get_data`. Now use `nilearn.image.get_data(img)`
  rather than `img.get_data()`. This is because Nibabel is removing the
  `get_data` method. You may also consider using the Nibabel
  `Nifti1Image.get_fdata`, which returns the data cast to floating-point.
  See https://github.com/nipy/nibabel/wiki/BIAP8 .
  As a benefit, the `get_data` function works on niimg-like objects such as
  filenames (see http://nilearn.github.io/manipulating_images/input_output.html ).

- Parcellation method ReNA: Fast agglomerative clustering based on recursive
  nearest neighbor grouping.
  Yields very fast & accurate models, without creation of giant
  clusters.
  :class:`nilearn.regions.ReNA`
- Plot connectome strength
  Use function ``nilearn.plotting.plot_connectome_strength`` to plot the strength of a
  connectome on a glass brain.  Strength is absolute sum of the edges at a node.
- Optimization to image resampling
- New brain development fMRI dataset fetcher
  :func:`nilearn.datasets.fetch_development_fmri` can be used to download
  movie-watching data in children and adults. A light-weight dataset
  implemented for teaching and usage in the examples. All the connectivity examples
  are changed from ADHD to brain development fmri dataset.

ENHANCEMENTS
------------

- :func:`nilearn.plotting.view_img_on_surf`, :func:`nilearn.plotting.view_surf`
  and :func:`nilearn.plotting.view_connectome` can display a title, and allow
  disabling the colorbar, and setting its height and the fontsize of its ticklabels.

- Rework of the standardize-options of :func:`nilearn.signal.clean` and the various Maskers
  in `nilearn.maskers`. You can now set `standardize` to `zscore` or `psc`. `psc` stands
  for `Percent Signal Change`, which can be a meaningful metric for BOLD.

- Class :class:`nilearn.maskers.NiftiLabelsMasker` now accepts an optional
  `strategy` parameter which allows it to change the function used to reduce
  values within each labelled ROI. Available functions include mean, median,
  minimum, maximum, standard_deviation and variance.
  This change is also introduced in :func:`nilearn.regions.img_to_signals_labels`.

- :func:`nilearn.plotting.view_surf` now accepts surface data provided as a file
  path.

CHANGES
-------

- :func:`nilearn.plotting.plot_img` now has explicit keyword arguments `bg_img`,
  `vmin` and `vmax` to control the background image and the bounds of the
  colormap. These arguments were already accepted in `kwargs` but not documented
  before.

FIXES
-----

- :class:`nilearn.maskers.NiftiLabelsMasker` no longer truncates region means to their integral part
  when input images are of integer type.
- The arg `version='det'` in :func:`nilearn.datasets.fetch_atlas_pauli_2017` now  works as expected.
- `pip install nilearn` now installs the necessary dependencies.

**Lots of other fixes in documentation and examples.** More detailed change list follows:

0.6.0rc
NEW
---
.. warning::

  - :func:`nilearn.plotting.view_connectome` no longer accepts old parameter names.
    Instead of `coords`, `threshold`, `cmap`, and `marker_size`,
    use `node_coords`, `edge_threshold`, `edge_cmap`, `node_size` respectively.

  - :func:`nilearn.plotting.view_markers` no longer accepts old parameter names.
    Instead of `coord` and `color`, use `marker_coords` and `marker_color` respectively.


- **Support for Python3.5 will be removed in the 0.7.x release.**
  Users with a Python3.5 environment will be warned
  at their first Nilearn import.

Changes
-------

- Add a warning to :class:`nilearn.regions.Parcellations`
  if the generated number of parcels does not match the requested number
  of parcels.
- Class :class:`nilearn.maskers.NiftiLabelsMasker` now accepts an optional
  `strategy` parameter which allows it to change the function used to reduce
  values within each labelled ROI. Available functions include mean, median,
  minimum, maximum, standard_deviation and variance.
  This change is also introduced in :func:`nilearn.regions.img_to_signals_labels`.

Fixes
-----

- :class:`nilearn.maskers.NiftiLabelsMasker` no longer truncates region means to their integral part
  when input images are of integer type.
- :func: `nilearn.image.smooth_image` no longer fails if `fwhm` is a `numpy.ndarray`.
- `pip install nilearn` now installs the necessary dependencies.
- :func:`nilearn.image.new_img_like` no longer attempts to copy non-iterable headers. (PR #2212)
- Nilearn no longer raises ImportError for nose when Matplotlib is not installed.
- The arg `version='det'` in :func:`nilearn.datasets.fetch_atlas_pauli_2017` now  works as expected.
- :func:`nilearn.maskers.NiftiLabelsMasker.inverse_transform` now works without the need to call
  transform first.

Contributors
------------

The following people contributed to this release (in alphabetical order)::

    Chris Markiewicz
    Dan Gale
    Daniel Gomez
    Derek Pisner
    Elizabeth DuPre
    Eric Larson
    Gael Varoquaux
    Jerome Dockes
    JohannesWiesner
    Kshitij Chawla (kchawla-pi)
    Paula Sanz-Leon
    ltetrel
    ryanhammonds


0.6.0b0
=======

**Released November 2019**


.. warning::

 | **Python2 and 3.4 are no longer supported. Pip will raise an error in these environments.**
 | **Minimum supported version of Python is now 3.5 .**
 | **We recommend upgrading to Python 3.6 .**


NEW
---

- A new function :func:`nilearn.image.get_data` to replace the deprecated
  nibabel method `Nifti1Image.get_data`. Now use `nilearn.image.get_data(img)`
  rather than `img.get_data()`. This is because Nibabel is removing the
  `get_data` method. You may also consider using the Nibabel
  `Nifti1Image.get_fdata`, which returns the data cast to floating-point.
  See https://github.com/nipy/nibabel/wiki/BIAP8 .
  As a benefit, the `get_data` function works on niimg-like objects such as
  filenames (see http://nilearn.github.io/manipulating_images/input_output.html ).

Changes
-------

- All functions and examples now use `nilearn.image.get_data` rather than the
  deprecated method `nibabel.Nifti1Image.get_data`.

- :func:`nilearn.datasets.fetch_neurovault` now does not filter out images that
  have their metadata field `is_valid` cleared by default.

- Users can now specify fetching data for adults, children, or both from
  :func:`nilearn.datasets.fetch_development_fmri` .


Fixes
-----

- :func:`nilearn.plotting.plot_connectome` now correctly displays marker size on 'l'
  and 'r' orientations, if an array or a list is passed to the function.

Contributors
------------

The following people contributed to this release (in alphabetical order)::

    Jake Vogel
    Jerome Dockes
    Kshitij Chawla (kchawla-pi)
    Roberto Guidotti

0.6.0a0
=======

**Released October 2019**

NEW
---

.. warning::

 | **Python2 and 3.4 are no longer supported. We recommend upgrading to Python 3.6 minimum.**
 |
 | **Minimum supported versions of packages have been bumped up.**
 | - Matplotlib -- v2.0
 | - Scikit-learn -- v0.19
 | - Scipy -- v0.19

- A new method for :class:`nilearn.maskers.NiftiMasker` instances
  for generating reports viewable in a web browser, Jupyter Notebook, or VSCode.

- joblib is now a dependency

- Parcellation method ReNA: Fast agglomerative clustering based on recursive
  nearest neighbor grouping.
  Yields very fast & accurate models, without creation of giant
  clusters.
  :class:`nilearn.regions.ReNA`
- Plot connectome strength
  Use function ``nilearn.plotting.plot_connectome_strength`` to plot the strength of a
  connectome on a glass brain.  Strength is absolute sum of the edges at a node.
- Optimization to image resampling
  :func:`nilearn.image.resample_img` has been optimized to pad rather than
  resample images in the special case when there is only a translation
  between two spaces. This is a common case in :class:`nilearn.maskers.NiftiMasker`
  when using the `mask_strategy="template"` option for brains in MNI space.
- New brain development fMRI dataset fetcher
  :func:`nilearn.datasets.fetch_development_fmri` can be used to download
  movie-watching data in children and adults; a light-weight dataset
  implemented for teaching and usage in the examples.
- New example in `examples/05_advanced/plot_age_group_prediction_cross_val.py`
  to compare methods for classifying subjects into age groups based on
  functional connectivity. Similar example in
  `examples/03_connectivity/plot_group_level_connectivity.py` simplified.

- Merged `examples/03_connectivity/plot_adhd_spheres.py` and
  `examples/03_connectivity/plot_sphere_based_connectome.py` to remove
  duplication across examples. The improved
  `examples/03_connectivity/plot_sphere_based_connectome.py` contains
  concepts previously reviewed in both examples.
- Merged `examples/03_connectivity/plot_compare_decomposition.py`
  and `examples/03_connectivity/plot_canica_analysis.py` into an improved
  `examples/03_connectivity/plot_compare_decomposition.py`.

- The Localizer dataset now follows the :term:`BIDS` organization.

Changes
-------

- All the connectivity examples are changed from ADHD to brain development
  fmri dataset.
- Examples plot_decoding_tutorial, plot_haxby_decoder,
  plot_haxby_different_estimators, plot_haxby_full_analysis, plot_oasis_vbm now
  use :class:`nilearn.decoding.Decoder` and :class:`nilearn.decoding.DecoderRegressor`
  instead of sklearn SVC and SVR.

- :func:`nilearn.plotting.view_img_on_surf`, :func:`nilearn.plotting.view_surf`
  and :func:`nilearn.plotting.view_connectome` now allow disabling the colorbar,
  and setting its height and the fontsize of its ticklabels.

- :func:`nilearn.plotting.view_img_on_surf`, :func:`nilearn.plotting.view_surf`
  and :func:`nilearn.plotting.view_connectome` can now display a title.

- Rework of the standardize-options of :func:`nilearn.signal.clean` and the various Maskers
  in `nilearn.maskers`. You can now set `standardize` to `zscore` or `psc`. `psc` stands
  for `Percent Signal Change`, which can be a meaningful metric for BOLD.

- :func:`nilearn.plotting.plot_img` now has explicit keyword arguments `bg_img`,
  `vmin` and `vmax` to control the background image and the bounds of the
  colormap. These arguments were already accepted in `kwargs` but not documented
  before.

- :func:`nilearn.plotting.view_connectome` now converts NaNs in the adjacency
  matrix to 0.

- Removed the plotting connectomes example which used the Seitzman atlas
  from `examples/03_connectivity/plot_sphere_based_connectome.py`.
  The atlas data is unsuitable for the method & the example is redundant.

Fixes
-----

- :func:`nilearn.plotting.plot_glass_brain` with colorbar=True does not crash when
  images have NaNs.
- add_contours now accepts `threshold` argument for filled=False. Now
  `threshold` is equally applied when asked for fillings in the contours.
- :func:`nilearn.plotting.plot_surf` and
  :func:`nilearn.plotting.plot_surf_stat_map` no longer threshold zero values
  when no threshold is given.
- When :func:`nilearn.plotting.plot_surf_stat_map` is used with a thresholded map
  but without a background map, the surface mesh is displayed in
  half-transparent grey to maintain a 3D perception.
- :func:`nilearn.plotting.view_surf` now accepts surface data provided as a file
  path.
- :func:`nilearn.plotting.plot_glass_brain` now correctly displays the left 'l' orientation even when
  the given images are completely masked (empty images).
- :func:`nilearn.plotting.plot_matrix` providing labels=None, False, or an empty list now correctly disables labels.
- :func:`nilearn.plotting.plot_surf_roi` now takes vmin, vmax parameters
- :func:`nilearn.datasets.fetch_surf_nki_enhanced` is now downloading the correct
  left and right functional surface data for each subject
- :func:`nilearn.datasets.fetch_atlas_schaefer_2018` now downloads from release
  version 0.14.3 (instead of 0.8.1) by default, which includes corrected region label
  names along with 700 and 900 region parcelations.
- Colormap creation functions have been updated to avoid matplotlib deprecation warnings
  about colormap reversal.
- Neurovault fetcher no longer fails if unable to update dataset metadata file due to faulty permissions.

Contributors
------------

The following people contributed to this release (in alphabetical order)::

	Alexandre Abraham
	Alexandre Gramfort
	Ana Luisa
	Ana Luisa Pinho
	Andrés Hoyos Idrobo
	Antoine Grigis
	BAZEILLE Thomas
	Bertrand Thirion
	Colin Reininger
	Céline Delettre
	Dan Gale
	Daniel Gomez
	Elizabeth DuPre
	Eric Larson
	Franz Liem
	Gael Varoquaux
	Gilles de Hollander
	Greg Kiar
	Guillaume Lemaitre
	Ian Abenes
	Jake Vogel
	Jerome Dockes
	Jerome-Alexis Chevalier
	Julia Huntenburg
	Kamalakar Daddy
	Kshitij Chawla (kchawla-pi)
	Mehdi Rahim
	Moritz Boos
	Sylvain Takerkart

0.5.2
=====

**Released April 2019**

NEW
---

.. warning::

 | This is the **last** release supporting Python2 and 3.4 .
 | The lowest Python version supported is now Python3.5.
 | We recommend switching to Python3.6 .

Fixes
-----

- Plotting ``.mgz`` files in MNE broke in ``0.5.1`` and has been fixed.

Contributors
------------

The following people contributed to this release::

    11  Kshitij Chawla (kchawla-pi)
     3  Gael Varoquaux
     2  Alexandre Gramfort

0.5.1
=====

**Released April 2019**

NEW
---
- **Support for Python2 & Python3.4 will be removed in the next release.**
  We recommend Python 3.6 and up.
  Users with a Python2 or Python3.4 environment will be warned
  at their first Nilearn import.

- Calculate image data dtype from header information
- New display mode 'tiled' which allows 2x2 plot arrangement when plotting three cuts
  (see :ref:`plotting`).
- NiftiLabelsMasker now consumes less memory when extracting the signal from a 3D/4D
  image. This is especially noteworthy when extracting signals from large 4D images.
- New function :func:`nilearn.datasets.fetch_atlas_schaefer_2018`
- New function :func:`nilearn.datasets.fetch_coords_seitzman_2018`

Changes
-------

- Lighting used for interactive surface plots changed; plots may look a bit
  different.
- :func:`nilearn.plotting.view_connectome` default colormap is `bwr`, consistent with plot_connectome.
- :func:`nilearn.plotting.view_connectome` parameter names are consistent with plot_connectome:

  - coords is now node_coord
  - marker_size is now node_size
  - cmap is now edge_cmap
  - threshold is now edge_threshold

- :func:`nilearn.plotting.view_markers` and :func:`nilearn.plotting.view_connectome` can accept different marker
  sizes for each node / marker.

- :func:`nilearn.plotting.view_markers()` default marker color is now 'red', consistent with add_markers().
- :func:`nilearn.plotting.view_markers` parameter names are consistent with add_markers():

  - coords is now marker_coords
  - colors is now marker_color

- :func:`nilearn.plotting.view_img_on_surf` now accepts a `symmetric_cmap`
  argument to control whether the colormap is centered around 0 and a `vmin`
  argument.

- Users can now control the size and fontsize of colorbars in interactive
  surface and connectome plots, or disable the colorbar.

Fixes
-----

- Example plot_seed_to_voxel_correlation now really saves z-transformed maps.
- region_extractor.connected_regions and regions.RegionExtractor now correctly
  use the provided mask_img.
- load_niimg no longer drops header if dtype is changed.
- NiftiSpheresMasker no longer silently ignores voxels if no `mask_img` is specified.
- Interactive brainsprites generated from `view_img` are correctly rendered in Jupyter Book.

Known Issues
-------------------

- On Python2, :func:`nilearn.plotting.view_connectome()` &
  :func:`nilearn.plotting.view_markers()`
  do not show parameters names in function signature
  when using help() and similar features.
  Please refer to their docstrings for this information.
- Plotting ``.mgz`` files in MNE is broken.

Contributors
------------

The following people contributed to this release::

   2  Bertrand Thirion
   90  Kshitij Chawla (kchawla-pi)
   22  fliem
   16  Jerome Dockes
   11  Gael Varoquaux
   8  Salma Bougacha
   7  himanshupathak21061998
   2  Elizabeth DuPre
   1  Eric Larson
   1  Pierre Bellec

0.5.0
=====

**Released November 2018**

NEW
---

  :ref:`interactive plotting functions <interactive-plotting>`,
  eg for use in a notebook.

- New functions :func:`nilearn.plotting.view_surf` and
  :func:`nilearn.plotting.view_img_on_surf` for interactive visualization of
  maps on the cortical surface in a web browser.

- New functions :func:`nilearn.plotting.view_connectome` and
  :func:`nilearn.plotting.view_markers` for interactive visualization of
  connectomes and seed locations in 3D

- New function :func:`nilearn.plotting.view_img` for interactive
  visualization of volumes with 3 orthogonal cuts.

:Note: :func:`nilearn.plotting.view_img` was `nilearn.plotting.view_stat_map` in alpha and beta releases.

- :func:`nilearn.plotting.find_parcellation_cut_coords` for
  extraction of coordinates on brain parcellations denoted as labels.

- Added :func:`nilearn.plotting.find_probabilistic_atlas_cut_coords` for
  extraction of coordinates on brain probabilistic maps.


**Minimum supported versions of packages have been bumped up.**
  - scikit-learn -- v0.18
  - scipy -- v0.17
  - pandas -- v0.18
  - numpy -- v1.11
  - matplotlib -- v1.5.1

**Nilearn Python2 support is being removed in the near future.**
  Users with a Python2 environment will be warned
  at their first Nilearn import.

**Additional dataset downloaders for examples and tutorials.**

- :func:`nilearn.datasets.fetch_surf_fsaverage`
- :func:`nilearn.datasets.fetch_atlas_pauli_2017`
- :func:`nilearn.datasets.fetch_neurovault_auditory_computation_task`
- :func:`nilearn.datasets.fetch_neurovault_motor_task`


ENHANCEMENTS
------------

 :func:`nilearn.image.clean_img` now accepts a mask to restrict
 the cleaning of the image, reducing memory load and computation time.

 NiftiMaskers now have a `dtype` parameter, by default keeping the same data type as the input data.

 Displays by plotting functions can now add a scale bar (see :ref:`plotting`)


IMPROVEMENTS
------------

 - Lots of other fixes in documentation and examples.
 - A cleaner layout and improved navigation for the website, with a better introduction.
 - Dataset fetchers are now  more reliable, less verbose.
 - Searchlight().fit() now accepts 4D niimgs.
 - Anaconda link in the installation documentation updated.
 - Scipy is listed as a dependency for Nilearn installation.

Notable Changes
---------------

 Default value of `t_r` in :func:`nilearn.signal.clean` and
 :func:`nilearn.image.clean_img` is None
 and cannot be None if `low_pass` or `high_pass` is specified.

Lots of changes and improvements. Detailed change list for each release follows.

0.5.0 rc
========

Highlights
----------

:func:`nilearn.plotting.view_img` (formerly `nilearn.plotting.view_stat_map` in
Nilearn 0.5.0 pre-release versions) generates significantly smaller notebooks
and HTML pages while getting a more consistent look and feel with Nilearn's
plotting functions. Huge shout out to Pierre Bellec (pbellec) for
making a great feature awesome and for sportingly accommodating all our feedback.

:func:`nilearn.image.clean_img` now accepts a mask to restrict the cleaning of
  the image. This approach can help to reduce the memory load and computation time.
  Big thanks to Michael Notter (miykael).

Enhancements
------------

- :func:`nilearn.plotting.view_img` is now using the brainsprite.js library,
  which results in much smaller notebooks or html pages. The interactive viewer
  also looks more similar to the plots generated by
  :func:`nilearn.plotting.plot_stat_map`, and most parameters found in
  `plot_stat_map` are now supported in `view_img`.
- :func:`nilearn.image.clean_img` now accepts a mask to restrict the cleaning of
  the image. This approach can help to reduce the memory load and computation time.
- :func:`nilearn.decoding.SpaceNetRegressor.fit` raises a meaningful error in regression tasks
  if the target Y contains all 1s.

Changes
-------

- Default value of `t_r` in :func:`nilearn.signal.clean` and
  :func:`nilearn.image.clean_img` is changed from 2.5 to None. If `low_pass` or
  `high_pass` is specified, then `t_r` needs to be specified as well otherwise
  it will raise an error.
- Order of filters in :func:`nilearn.signal.clean` and :func:`nilearn.image.clean_img`
  has changed to detrend, low- and high-pass filter, remove confounds and
  standardize. To ensure orthogonality between temporal filter and confound
  removal, an additional temporal filter will be applied on the confounds before
  removing them. This is according to Lindquist et al. (2018).
- :func:`nilearn.image.clean_img` now accepts a mask to restrict the cleaning of
  the image. This approach can help to reduce the memory load and computation time.
- :func:`nilearn.plotting.view_img` is now using the brainsprite.js library,
  which results in much smaller notebooks or html pages. The interactive viewer
  also looks more similar to the plots generated by
  :func:`nilearn.plotting.plot_stat_map`, and most parameters found in
  `plot_stat_map` are now supported in `view_img`.


Contributors
-------------

The following people contributed to this release::

  15 Gael Varoquaux
  114 Pierre Bellec
  30 Michael Notter
  28 Kshitij Chawla (kchawla-pi)
  4 Kamalakar Daddy
  4 himanshupathak21061998
  1 Horea Christian
  7 Jerome Dockes

0.5.0 beta
==========

Highlights
----------

**Nilearn Python2 support is being removed in the near future.
Users with a Python2 environment will be warned at their first Nilearn import.**

Enhancements
------------

Displays created by plotting functions can now add a scale bar
 to indicate the size in mm or cm (see :ref:`plotting`),
 contributed by Oscar Esteban

Colorbars in plotting functions now have a middle gray background
 suitable for use with custom colormaps with a non-unity alpha channel.
 Contributed by Eric Larson (larsoner)

Loads of fixes and quality of life improvements

- A cleaner layout and improved navigation for the website, with a better introduction.
- Less warnings and verbosity while using certain functions and during dataset downloads.
- Improved backend for the dataset fetchers means more reliable dataset downloads.
- Some datasets, such as the ICBM, are now compressed to take up less disk space.


Fixes
-----

- Searchlight().fit() now accepts 4D niimgs. Contributed by Dan Gale (danjgale).
- plotting.view_markers.open_in_browser() in js_plotting_utils fixed
- Brainomics dataset has been replaced in several examples.
- Lots of other fixes in documentation and examples.


Changes
-------

- In nilearn.regions.img_to_signals_labels, the See Also section in documentation now also points to NiftiLabelsMasker and NiftiMapsMasker
- Scipy is listed as a dependency for Nilearn installation.
- Anaconda link in the installation documentation updated.

Contributors
-------------

The following people contributed to this release::

  58  Gael Varoquaux
  115  Kshitij Chawla (kchawla-pi)
  15  Jerome Dockes
  14  oesteban
  10  Eric Larson
  6  Kamalakar Daddy
  3  Bertrand Thirion
  5  Alexandre Abadie
  4  Sourav Singh
  3  Alex Rothberg
  3  AnaLu
  3  Demian Wassermann
  3  Horea Christian
  3  Jason Gors
  3  Jean Remi King
  3  MADHYASTHA Meghana
  3  SRSteinkamp
  3  Simon Steinkamp
  3  jerome-alexis_chevalier
  3  salma
  3  sfvnMAC
  2  Akshay
  2  Daniel Gomez
  2  Guillaume Lemaitre
  2  Pierre Bellec
  2  arokem
  2  erramuzpe
  2  foucault
  2  jehane
  1  Sylvain LANNUZEL
  1  Aki Nikolaidis
  1  Christophe Bedetti
  1  Dan Gale
  1  Dillon Plunkett
  1  Dimitri Papadopoulos Orfanos
  1  Greg Operto
  1  Ivan Gonzalez
  1  Yaroslav Halchenko
  1  dtyulman

0.5.0 alpha
===========

This is an alpha release: to download it, you need to explicitly ask for
the version number::

   pip install nilearn==0.5.0a0

Highlights
----------

    - **Minimum supported versions of packages have been bumped up.**
        - scikit-learn -- v0.18
        - scipy -- v0.17
        - pandas -- v0.18
        - numpy -- v1.11
        - matplotlib -- v1.5.1

    - New :ref:`interactive plotting functions <interactive-plotting>`,
      eg for use in a notebook.

Enhancements
------------

    - All NiftiMaskers now have a `dtype` argument. For now the default behaviour
      is to keep the same data type as the input data.

    - Displays created by plotting functions can now add a scale bar to
      indicate the size in mm or cm (see :ref:`plotting`), contributed by
      Oscar Esteban

    - New functions :func:`nilearn.plotting.view_surf` and
      :func:`nilearn.plotting.view_surf` and
      :func:`nilearn.plotting.view_img_on_surf` for interactive visualization of
      maps on the cortical surface in a web browser.

    - New functions :func:`nilearn.plotting.view_connectome` and
      :func:`nilearn.plotting.view_markers` to visualize connectomes and
      seed locations in 3D

    - New function `nilearn.plotting.view_stat_map` (renamed to
      :func:`nilearn.plotting.view_img` in stable release) for interactive
      visualization of volumes with 3 orthogonal cuts.

    - Add :func:`nilearn.datasets.fetch_surf_fsaverage` to download either
      fsaverage or fsaverage 5 (Freesurfer cortical meshes).

    - Added :func:`nilearn.datasets.fetch_atlas_pauli_2017` to download a
      recent subcortical neuroimaging atlas.

    - Added :func:`nilearn.plotting.find_parcellation_cut_coords` for
      extraction of coordinates on brain parcellations denoted as labels.

    - Added :func:`nilearn.plotting.find_probabilistic_atlas_cut_coords` for
      extraction of coordinates on brain probabilistic maps.

    - Added :func:`nilearn.datasets.fetch_neurovault_auditory_computation_task`
      and :func:`nilearn.datasets.fetch_neurovault_motor_task` for simple example data.

Changes
-------

    - `nilearn.datasets.fetch_surf_fsaverage5` is deprecated and will be
      removed in a future release. Use :func:`nilearn.datasets.fetch_surf_fsaverage`,
      with the parameter mesh="fsaverage5" (the default) instead.

    - fsaverage5 surface data files are now shipped directly with Nilearn.
      Look to issue #1705 for discussion.

    - `sklearn.cross_validation` and `sklearn.grid_search` have been
      replaced by `sklearn.model_selection` in all the examples.

    - Colorbars in plotting functions now have a middle gray background
      suitable for use with custom colormaps with a non-unity alpha channel.


Contributors
------------

The following people contributed to this release::

    49  Gael Varoquaux
    180  Jerome Dockes
    57  Kshitij Chawla (kchawla-pi)
    38  SylvainLan
    36  Kamalakar Daddy
    10  Gilles de Hollander
    4  Bertrand Thirion
    4  MENUET Romuald
    3  Moritz Boos
    1  Peer Herholz
    1  Pierre Bellec

0.4.2
=====
Few important bugs fix release for OHBM conference.

Changes
-------
    - Default colormaps for surface plotting functions have changed to be more
      consistent with slice plotting.
      :func:`nilearn.plotting.plot_surf_stat_map` now uses "cold_hot", as
      :func:`nilearn.plotting.plot_stat_map` does, and
      :func:`nilearn.plotting.plot_surf_roi` now uses "gist_ncar", as
      :func:`nilearn.plotting.plot_roi` does.

    - Improve 3D surface plotting: lock the aspect ratio of the plots and
      reduce the whitespace around the plots.

Bug fixes
---------

    - Fix bug with input repetition time (TR) which had no effect in signal
      cleaning. Fixed by Pradeep Raamana.

    - Fix issues with signal extraction on list of 3D images in
      :class:`nilearn.regions.Parcellations`.

    - Fix issues with raising AttributeError rather than HTTPError in datasets
      fetching utilities. By Jerome Dockes.

    - Fix issues in datasets testing function uncompression of files. By Pierre Glaser.

0.4.1
=====

This bug fix release is focused on few bug fixes and minor developments.

Enhancements
------------

    - :class:`nilearn.decomposition.CanICA` and
      :class:`nilearn.decomposition.DictLearning` has new attribute
      `components_img_` providing directly the components learned as
      a Nifti image. This avoids the step of unmasking the attribute
      `components_` which is true for older versions.

    - New object :class:`nilearn.regions.Parcellations` for learning brain
      parcellations on fmri data.

    - Add optional reordering of the matrix using a argument `reorder`
      with :func:`nilearn.plotting.plot_matrix`.

      .. note::
        This feature is usable only if SciPy version is >= 1.0.0

Changes
-------

    - Using output attribute `components_` which is an extracted components
      in :class:`nilearn.decomposition.CanICA` and
      :class:`nilearn.decomposition.DictLearning` is deprecated and will
      be removed in next two releases. Use `components_img_` instead.

Bug fixes
---------

    - Fix issues using :func:`nilearn.plotting.plot_connectome` when string is
      passed in `node_color` with display modes left and right hemispheric cuts
      in the glass brain.

    - Fix bug while plotting only coordinates using add_markers on glass brain.
      See issue #1595

    - Fix issues with estimators in decomposition module when input images are
      given in glob patterns.

    - Fix bug loading Nifti2Images.

    - Fix bug while adjusting contrast of the background template while using
      :func:`nilearn.plotting.plot_prob_atlas`

    - Fix colormap bug with recent matplotlib 2.2.0

0.4.0
=====

**Highlights**:

    - :func:`nilearn.surface.vol_to_surf` to project volume data to the
      surface.

    - :func:`nilearn.plotting.plot_matrix` to display matrices, eg connectomes

Enhancements
-------------

    - New function :func:`nilearn.surface.vol_to_surf` to project a 3d or
      4d brain volume on the cortical surface.

    - New matrix plotting function, eg to display connectome matrices:
      :func:`nilearn.plotting.plot_matrix`

    - Expose :func:`nilearn.image.coord_transform` for end users. Useful
      to transform coordinates (x, y, z) from one image space to
      another space.

    - :func:`nilearn.image.resample_img` now takes a linear resampling
      option (implemented by Joe Necus)

    - :func:`nilearn.datasets.fetch_atlas_talairach` to fetch the Talairach
      atlas (http://talairach.org)

    - Enhancing new surface plotting functions, added new parameters
      "axes" and "figure" to accept user-specified instances in
      :func:`nilearn.plotting.plot_surf` and
      :func:`nilearn.plotting.plot_surf_stat_map` and
      :func:`nilearn.plotting.plot_surf_roi`

    - :class:`nilearn.decoding.SearchLight` has new parameter "groups" to
      do LeaveOneGroupOut type cv with new scikit-learn module model selection.

    - Enhancing the glass brain plotting in back view 'y' direction.

    - New parameter "resampling_interpolation" is added in most used
      plotting functions to have user control for faster visualizations.

    - Upgraded to Sphinx-Gallery 0.1.11

Bug fixes
----------

    - Dimming factor applied to background image in plotting
      functions with "dim" parameter will no longer accepts as
      string ('-1'). An error will be raised.

    - Fixed issues with matplotlib 2.1.0.

    - Fixed issues with SciPy 1.0.0.

Changes
---------

    - **Backward incompatible change**: :func:`nilearn.plotting.find_xyz_cut_coords`
      now takes a `mask_img` argument which is a niimg, rather than a `mask`
      argument, which used to be a numpy array.

    - The minimum required version for scipy is now 0.14

    - Dropped support for Nibabel older than 2.0.2.

    - :func:`nilearn.image.smooth_img` no longer accepts smoothing
      parameter fwhm as 0. Behavior is changed in according to the
      issues with recent SciPy version 1.0.0.

    - "dim" factor range is slightly increased to -2 to 2 from -1 to 1.
      Range exceeding -1 meaning more increase in contrast should be
      cautiously set.

    - New 'anterior' and 'posterior' view added to the plot_surf family views

    - Using argument `anat_img` for placing background image in
      :func:`nilearn.plotting.plot_prob_atlas` is deprecated. Use argument
      `bg_img` instead.

    - The examples now use pandas for the behavioral information.

Contributors
-------------

The following people contributed to this release::

   127  Jerome Dockes
    62  Gael Varoquaux
    36  Kamalakar Daddy
    11  Jeff Chiang
     9  Elizabeth DuPre
     9  Jona Sassenhagen
     7  Sylvain Lan
     6  J Necus
     5  Pierre-Olivier Quirion
     3  AnaLu
     3  Jean Remi King
     3  MADHYASTHA Meghana
     3  Salma Bougacha
     3  sfvnMAC
     2  Eric Larson
     2  Horea Christian
     2  Moritz Boos
     1  Alex Rothberg
     1  Bertrand Thirion
     1  Christophe Bedetti
     1  John Griffiths
     1  Mehdi Rahim
     1  Sylvain LANNUZEL
     1  Yaroslav Halchenko
     1  clfs


0.3.1
=====

This is a minor release for BrainHack.

Highlights
----------

* **Dropped support for scikit-learn older than 0.14.1** Minimum supported version
  is now 0.15.

Changelog
---------

    - The function sym_to_vec is deprecated and will be removed in
      release 0.4. Use :func:`nilearn.connectome.sym_matrix_to_vec` instead.

    - Added argument `smoothing_fwhm` to
      :class:`nilearn.regions.RegionExtractor` to control smoothing according
      to the resolution of atlas images.

Bug fix
-------

    - The helper function `largest_connected_component` should now work with
      inputs of non-native data dtypes.

    - Fix plotting issues when non-finite values are present in background
      anatomical image.

    - A workaround to handle non-native endianness in the Nifti images passed
      to resampling the image.

Enhancements
-------------
    - New data fetcher functions :func:`nilearn.datasets.fetch_neurovault` and
      :func:`nilearn.datasets.fetch_neurovault_ids` help you download
      statistical maps from the Neurovault (http://neurovault.org) platform.

    - New function :func:`nilearn.connectome.vec_to_sym_matrix` reshapes
      vectors to symmetric matrices. It acts as the reverse of function
      :func:`nilearn.connectome.sym_matrix_to_vec`.

    - Add an option allowing to vectorize connectivity matrices returned by the
      "transform" method of :class:`nilearn.connectome.ConnectivityMeasure`.

    - :class:`nilearn.connectome.ConnectivityMeasure` now exposes an
      "inverse_transform" method, useful for going back from vectorized
      connectivity coefficients to connectivity matrices. Also, it allows to
      recover the covariance matrices for the "tangent" kind.

    - Reworking and renaming of connectivity measures example. Renamed from
      plot_connectivity_measures to plot_group_level_connectivity.

    - Tighter bounding boxes when using add_contours for plotting.

    - Function :func:`nilearn.image.largest_connected_component_img` to
      directly extract the largest connected component from Nifti images.

    - Improvements in plotting, decoding and functional connectivity examples.

0.3.0
======

In addition, more details of this release are listed below. Please checkout
in **0.3.0 beta** release section for minimum version support of dependencies,
latest updates, highlights, changelog and enhancements.

Changelog
---------

    - Function :func:`nilearn.plotting.find_cut_slices` now supports to accept
      Nifti1Image as an input for argument `img`.

    - Helper functions `_get_mask_volume` and `_adjust_screening_percentile`
      are now moved to param_validation file in utilities module to be used in
      common with Decoder object.

Bug fix
--------

    - Fix bug uncompressing tar files with datasets fetcher.

    - Fixed bunch of CircleCI documentation build failures.

    - Fixed deprecations `set_axis_bgcolor` related to matplotlib in
      plotting functions.

    - Fixed bug related to not accepting a list of arrays as an input to
      unmask, in masking module.

Enhancements
-------------

    - ANOVA SVM example on Haxby datasets `plot_haxby_anova_svm` in Decoding section
      now uses `SelectPercentile` to select voxels rather than `SelectKBest`.

    - New function `fast_svd` implementation in base decomposition module to
      Automatically switch between randomized and lapack SVD (heuristic
      of scikit-learn).

0.3.0 beta
===========

To install the beta version, use::

  pip install --upgrade --pre nilearn

Highlights
----------

* Simple surface plotting

* A function to break a parcellation into its connected components

* **Dropped support for scikit-learn older than 0.14.1** Minimum supported version
  is now 0.14.1.

* **Dropped support for Python 2.6**

* Minimum required version of NiBabel is now 1.2.0, to support loading annotated
  data with freesurfer.

Changelog
---------

    - A helper function _safe_get_data as a nilearn utility now safely
      removes NAN values in the images with argument ensure_finite=True.

    - Connectome functions :func:`nilearn.connectome.cov_to_corr` and
      :func:`nilearn.connectome.prec_to_partial` can now be used.

Bug fix
--------

    - Fix colormap issue with colorbar=True when using qualitative colormaps
      Fixed in according with changes of matplotlib 2.0 fixes.

    - Fix plotting functions to work with NAN values in the images.

    - Fix bug related get dtype of the images with nibabel get_data().

    - Fix bug in nilearn clean_img

Enhancements
............

    - A new function :func:`nilearn.regions.connected_label_regions` to
      extract the connected components represented as same label to regions
      apart with each region labelled as unique label.

    - New plotting modules for surface plotting visualization. Matplotlib with
      version higher 1.3.1 is required for plotting surface data using these
      functions.

    - Function :func:`nilearn.plotting.plot_surf` can be used for plotting
      surfaces mesh data with optional background.

    - A function :func:`nilearn.plotting.plot_surf_stat_map` can be used for
      plotting statistical maps on a brain surface with optional background.

    - A function :func:`nilearn.plotting.plot_surf_roi` can be used for
      plotting statistical maps rois onto brain surface.

    - A function `nilearn.datasets.fetch_surf_fsaverage5` can be used
      for surface data object to be as background map for the above plotting
      functions.

    - A new data fetcher function
      :func:`nilearn.datasets.fetch_atlas_surf_destrieux`
      can give you Destrieux et. al 2010 cortical atlas in fsaverage5
      surface space.

    - A new functional data fetcher function
      :func:`nilearn.datasets.fetch_surf_nki_enhanced` gives you resting state
      data preprocessed and projected to fsaverage5 surface space.

    - Two good examples in plotting gallery shows how to fetch atlas and NKI
      data and used for plotting on brain surface.

    - Helper function `load_surf_mesh` in surf_plotting module for loading
      surface mesh data into two arrays, containing (x, y, z) coordinates
      for mesh vertices and indices of mesh faces.

    - Helper function `load_surf_data` in surf_plotting module for loading
      data of numpy array to represented on a surface mesh.

    - Add fetcher for Allen et al. 2011 RSN atlas in
      :func:`nilearn.datasets.fetch_atlas_allen_2011`.

    - A function ``nilearn.datasets.fetch_cobre`` is now updated to new
      light release of COBRE data (schizophrenia)

    - A new example to show how to extract regions on labels image in example
      section manipulating images.

    - coveralls is replaces with codecov

    - Upgraded to Sphinx version 0.1.7

    - Extensive plotting example shows how to use contours and filled contours
      on glass brain.

0.2.6
=====

Changelog
---------

This release enhances usage of several functions by fine tuning their
parameters. It allows to select which Haxby subject to fetch. It also refactors
documentation to make it easier to understand.
Sphinx-gallery has been updated and nilearn is ready for new nibabel 2.1 version.
Several bugs related to masks in Searchlight and ABIDE fetching have been
resolved.

Bug fix
........

    - Change default dtype in :func:`nilearn.image.concat_imgs` to be the
      original type of the data (see #1238).

    - Fix SearchLight that did not run without process_mask or with one voxel
      mask.

    - Fix flipping of left hemisphere when plotting glass brain.

    - Fix bug when downloading ABIDE timeseries

Enhancements
............

   - Sphinx-gallery updated to version 0.1.3.

   - Refactoring of examples and documentation.

   - Better ordering of regions in
     :func:`nilearn.datasets.fetch_coords_dosenbach_2010`.

   - Remove outdated power atlas example.


API changes summary
...................

    - The parameter 'n_subjects' is deprecated and will be removed in future
      release. Use 'subjects' instead in :func:`nilearn.datasets.fetch_haxby`.

    - The function :func:`nilearn.datasets.fetch_haxby` will now fetch the
      data accepting input given in 'subjects' as a list than integer.

    - Replace `get_affine` by `affine` with recent versions of nibabel.

0.2.5.1
=======

Changelog
---------

This is a bugfix release.
The new minimum required version of scikit-learn is 0.14.1

API changes summary
...................

    - default option for `dim` argument in plotting functions which uses MNI
      template as a background image is now changed to 'auto' mode. Meaning
      that an automatic contrast setting on background image is applied by
      default.

    - Scikit-learn validation tools have been imported and are now used to check
      consistency of input data, in SpaceNet for example.

New features
............

    - Add an option to select only off-diagonal elements in sym_to_vec. Also,
      the scaling of matrices is modified: we divide the diagonal by sqrt(2)
      instead of multiplying the off-diagonal elements.

    - Connectivity examples rely on
      :class:`nilearn.connectome.ConnectivityMeasure`

Bug fix
........

    - Scipy 0.18 introduces a bug in a corner-case of resampling. Nilearn
      0.2.5 can give wrong results with scipy 0.18, but this is fixed in
      0.2.6.

    - Broken links and references fixed in docs

0.2.5
=====

Changelog
---------

The 0.2.5 release includes plotting for connectomes and glass brain with
hemisphere-specific projection, as well as more didactic examples and
improved documentation.

New features
............

    - New display_mode options in :func:`nilearn.plotting.plot_glass_brain`
      and :func:`nilearn.plotting.plot_connectome`. It
      is possible to plot right and left hemisphere projections separately.

    - A function to load canonical brain mask image in MNI template space,
      :func:`nilearn.datasets.load_mni152_brain_mask`

    - A function to load brain grey matter mask image,
      :func:`nilearn.datasets.fetch_icbm152_brain_gm_mask`

    - New function :func:`nilearn.image.load_img` loads data from a filename or a
      list of filenames.

    - New function :func:`nilearn.image.clean_img` applies the cleaning function
      :func:`nilearn.signal.clean` on all voxels.

    - New simple data downloader
      :func:`nilearn.datasets.fetch_localizer_button_task` to simplify
      some examples.

    - The dataset function
      :func:`nilearn.datasets.fetch_localizer_contrasts` can now download
      a specific list of subjects rather than a range of subjects.

    - New function :func:`nilearn.datasets.get_data_dirs` to check where
      nilearn downloads data.

Contributors
-------------

Contributors (from ``git shortlog -ns 0.2.4..0.2.5``)::

    55  Gael Varoquaux
    39  Alexandre Abraham
    26  Martin Perez-Guevara
    20  Kamalakar Daddy
     8  amadeuskanaan
     3  Alexandre Abadie
     3  Arthur Mensch
     3  Elvis Dohmatob
     3  Loïc Estève
     2  Jerome Dockes
     1  Alexandre M. S
     1  Bertrand Thirion
     1  Ivan Gonzalez
     1  robbisg

0.2.4
=====

Changelog
---------

The 0.2.4 is a small release focused on documentation for teaching.

New features
............
    - The path given to the "memory" argument of object now have their
      "~" expanded to the homedir

    - Display object created by plotting now uniformly expose an
      "add_markers" method.

    - plotting plot_connectome with colorbar is now implemented in function
      :func:`nilearn.plotting.plot_connectome`

    - New function :func:`nilearn.image.resample_to_img` to resample one
      img on another one (just resampling / interpolation, no
      coregistration)

API changes summary
...................
    - Atlas fetcher :func:`nilearn.datasets.fetch_atlas_msdl` now returns directly
      labels of the regions in output variable 'labels' and its coordinates
      in output variable 'region_coords' and its type of network in 'networks'.
    - The output variable name 'regions' is now changed to 'maps' in AAL atlas
      fetcher in :func:`nilearn.datasets.fetch_atlas_aal`.
    - AAL atlas now returns directly its labels in variable 'labels' and its
      index values in variable 'indices'.

0.2.3
=====

Changelog
---------

The 0.2.3 is a small feature release for BrainHack 2016.

New features
............
    - Mathematical formulas based on numpy functions can be applied on an
      image or a list of images using :func:`nilearn.image.math_img`.
    - Downloader for COBRE datasets of 146 rest fMRI subjects with
      function ``nilearn.datasets.fetch_cobre``.
    - Downloader for Dosenbach atlas
      :func:`nilearn.datasets.fetch_coords_dosenbach_2010`
    - Fetcher for multiscale functional brain parcellations (BASC)
      :func:`nilearn.datasets.fetch_atlas_basc_multiscale_2015`

Bug fixes
.........
    - Better dimming on white background for plotting

0.2.2
======

Changelog
---------

The 0.2.2 is a bugfix + dependency update release (for sphinx gallery). It
aims at preparing a renewal of the tutorials.

New features
............
   - Fetcher for Megatrawl Netmats dataset.

Enhancements
............
   - Flake8 is now run on pull requests.
   - Reworking of the documentation organization.
   - Sphinx-gallery updated to version 0.1.1
   - The default n_subjects=None in :func:`nilearn.datasets.fetch_adhd` is now
     changed to n_subjects=30.

Bug fixes
.........
   - Fix `symmetric_split` behavior in
     :func:`nilearn.datasets.fetch_atlas_harvard_oxford`
   - Fix casting errors when providing integer data to
     :func:`nilearn.image.high_variance_confounds`
   - Fix matplotlib 1.5.0 compatibility in
     :func:`nilearn.plotting.plot_prob_atlas`
   - Fix matplotlib backend choice on Mac OS X.
   - :func:`nilearn.plotting.find_xyz_cut_coords` raises a meaningful error
     when 4D data is provided instead of 3D.
   - :class:`nilearn.maskers.NiftiSpheresMasker` handles radius smaller than
     the size of a voxel
   - :class:`nilearn.regions.RegionExtractor` handles data containing Nans.
   - Confound regression does not force systematically the normalization of
     the confounds.
   - Force time series normalization in
     :class:`nilearn.connectome.ConnectivityMeasure`
     and check dimensionality of the input.
   - `nilearn._utils.numpy_conversions.csv_to_array` could consider
     valid CSV files as invalid.

API changes summary
...................
   - Deprecated dataset downloading function have been removed.
   - Download progression message refreshing rate has been lowered to sparsify
     CircleCI logs.

Contributors
.............

Contributors (from ``git shortlog -ns 0.2.1..0.2.2``)::

    39  Kamalakar Daddy
    22  Alexandre Abraham
    21  Loïc Estève
    19  Gael Varoquaux
    12  Alexandre Abadie
     7  Salma
     3  Danilo Bzdok
     1  Arthur Mensch
     1  Ben Cipollini
     1  Elvis Dohmatob
     1  Óscar Nájera

0.2.1
======

Changelog
---------

Small bugfix for more flexible input types (targetter in particular at
making code easier in nistats).

0.2
===

Changelog
---------

The new minimum required version of scikit-learn is 0.13

New features
............
   - The new module :mod:`nilearn.connectome` now has class
     :class:`nilearn.connectome.ConnectivityMeasure` can be useful for
     computing functional connectivity matrices.
   - The function nilearn.connectome.sym_to_vec in same module
     :mod:`nilearn.connectome` is also implemented as a helper function to
     :class:`nilearn.connectome.ConnectivityMeasure`.
   - The class :class:`nilearn.decomposition.DictLearning` in
     :mod:`nilearn.decomposition` is a decomposition method similar to ICA
     that imposes sparsity on components instead of independence between them.
   - Integrating back references template from sphinx-gallery of 0.0.11
     version release.
   - Globbing expressions can now be used in all nilearn functions expecting a
     list of files.
   - The new module :mod:`nilearn.regions` now has class
     :class:`nilearn.regions.RegionExtractor` which can be used for post
     processing brain regions of interest extraction.
   - The function :func:`nilearn.regions.connected_regions` in
     :mod:`nilearn.regions` is also implemented as a helper function to
     :class:`nilearn.regions.RegionExtractor`.
   - The function :func:`nilearn.image.threshold_img` in :mod:`nilearn.image`
     is implemented to use it for thresholding statistical maps.

Enhancements
............
   - Making website a bit elaborated & modernise by using sphinx-gallery.
   - Documentation enhancement by integrating sphinx-gallery notebook style
     examples.
   - Documentation about :class:`nilearn.maskers.NiftiSpheresMasker`.

Bug fixes
.........
   - Fixed bug to control the behaviour when cut_coords=0. in function
     :func:`nilearn.plotting.plot_stat_map` in :mod:`nilearn.plotting`.
     See issue # 784.
   - Fixed bug in :func:`nilearn.image.copy_img` occurred while caching
     the Nifti images. See issue # 793.
   - Fixed bug causing an IndexError in fast_abs_percentile. See issue # 875

API changes summary
...................
   - The utilities in function group_sparse_covariance has been moved
     into :mod:`nilearn.connectome`.
   - The default value for number of cuts (n_cuts) in function
     :func:`nilearn.plotting.find_cut_slices` in :mod:`nilearn.plotting` has
     been changed from 12 to 7 i.e. n_cuts=7.

Contributors
.............

Contributors (from ``git shortlog -ns 0.1.4..0.2.0``)::

   822  Elvis Dohmatob
   142  Gael Varoquaux
   119  Alexandre Abraham
    90  Loïc Estève
    85  Kamalakar Daddy
    65  Alexandre Abadie
    43  Chris Filo Gorgolewski
    39  Salma BOUGACHA
    29  Danilo Bzdok
    20  Martin Perez-Guevara
    19  Mehdi Rahim
    19  Óscar Nájera
    17  martin
     8  Arthur Mensch
     8  Ben Cipollini
     4  ainafp
     4  juhuntenburg
     2  Martin_Perez_Guevara
     2  Michael Hanke
     2  arokem
     1  Bertrand Thirion
     1  Dimitri Papadopoulos Orfanos


0.1.4
=====

Changelog
---------

Highlights:

- NiftiSpheresMasker: extract signals from balls specified by their
  coordinates
- Obey Debian packaging rules
- Add the Destrieux 2009 and Power 2011 atlas
- Better caching in maskers


Contributors (from ``git shortlog -ns 0.1.3..0.1.4``)::

   141  Alexandre Abraham
    15  Gael Varoquaux
    10  Loïc Estève
     2  Arthur Mensch
     2  Danilo Bzdok
     2  Michael Hanke
     1  Mehdi Rahim


0.1.3
=====

Changelog
---------

The 0.1.3 release is a bugfix release that fixes a lot of minor bugs. It
also includes a full rewamp of the documentation, and support for Python
3.

Minimum version of supported packages are now:

- numpy 1.6.1
- scipy 0.9.0
- scikit-learn 0.12.1
- Matplotlib 1.1.1 (optional)

A non exhaustive list of issues fixed:

- Dealing with NaNs in plot_connectome
- Fix extreme values in colorbar were sometimes brok
- Fix confounds removal with single confounds
- Fix frequency filtering
- Keep header information in images
- add_overlay finds vmin and vmax automatically
- vmin and vmax support in plot_connectome
- detrending 3D images no longer puts them to zero


Contributors (from ``git shortlog -ns 0.1.2..0.1.3``)::

   129  Alexandre Abraham
    67  Loïc Estève
    57  Gael Varoquaux
    44  Ben Cipollini
    37  Danilo Bzdok
    20  Elvis Dohmatob
    14  Óscar Nájera
     9  Salma BOUGACHA
     8  Alexandre Gramfort
     7  Kamalakar Daddy
     3  Demian Wassermann
     1  Bertrand Thirion

0.1.2
=====

Changelog
---------

The 0.1.2 release is a bugfix release, specifically to fix the
NiftiMapsMasker.

0.1.1
=====

Changelog
---------

The main change compared to 0.1 is the addition of connectome plotting
via the nilearn.plotting.plot_connectome function. See the
`plotting documentation <building_blocks/plotting.html>`_
for more details.

Contributors (from ``git shortlog -ns 0.1..0.1.1``)::

    81  Loïc Estève
    18  Alexandre Abraham
    18  Danilo Bzdok
    14  Ben Cipollini
     2  Gaël Varoquaux


0.1
===

Changelog
---------
First release of nilearn.

Contributors (from ``git shortlog -ns 0.1``)::

   600  Gaël Varoquaux
   483  Alexandre Abraham
   302  Loïc Estève
   254  Philippe Gervais
   122  Virgile Fritsch
    83  Michael Eickenberg
    59  Jean Kossaifi
    57  Jaques Grobler
    46  Danilo Bzdok
    35  Chris Filo Gorgolewski
    28  Ronald Phlypo
    25  Ben Cipollini
    15  Bertrand Thirion
    13  Alexandre Gramfort
    12  Fabian Pedregosa
    11  Yannick Schwartz
     9  Mehdi Rahim
     7  Óscar Nájera
     6  Elvis Dohmatob
     4  Konstantin Shmelkov
     3  Jason Gors
     3  Salma Bougacha
     1  Alexandre Savio
     1  Jan Margeta
     1  Matthias Ekman
     1  Michael Waskom
     1  Vincent Michel
