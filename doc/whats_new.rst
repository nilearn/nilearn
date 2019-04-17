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
- **Support for Python2 & Python3.4 wil be removed in the next release.**
  We recommend Python 3.6 and up.
  Users with a Python2 or Python3.4 environment will be warned
  at their first Nilearn import.

- Calculate image data dtype from header information
- New display mode 'tiled' which allows 2x2 plot arrangement when plotting three cuts
  (see :ref:`plotting`).
- NiftiLabelsMasker now consumes less memory when extracting the signal from a 3D/4D
  image. This is especially noteworthy when extracting signals from large 4D images.
- New function :func:`nilearn.datasets.fetch_atlas_schaefer_2018`

Changes
-------

- Lighting used for interactive surface plots changed; plots may look a bit
  different.
- :func:`nilearn.plotting.view_connectome` default colormap is `bwr`, consistent with plot_connectome.
- :func:`nilearn.plotting.view_connectome` parameter names are consistent with plot_connectome:

  - coords is now node_coord
  - marker_size is noe node_size
  - cmap is now edge_cmap
  - threshold is now edge_threshold

- :func:`nilearn.plotting.view_markers` and :func:`nilearn.plotting.view_connectome` can accept different marker
  sizes for each node / marker.

- :func:`nilearn.plotting.view_markers()` default marker color is now 'red', consistent with add_markers().
- :func:`nilearn.plotting.view_markers` parameter names are consistent with add_markers():

  - coords is now marker_coords
  - colors is now marker_color

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

This bug fix release is focussed on few bug fixes and minor developments.

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
      Range exceeding -1 meaning more increase in constrast should be
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

    - A workaround to handle non-native endianess in the Nifti images passed
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
      are now moved to param_validation file in utilties module to be used in
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

* Minimum required version of NiBabel is now 1.2.0, to support loading annoted
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

    - A function :func:`nilearn.datasets.fetch_cobre` is now updated to new
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

    - Display object created by plotting now uniformely expose an
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
      :func:`nilearn.datasets.fetch_cobre`
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
   - :class:`nilearn.input_data.NiftiSpheresMasker` handles radius smaller than
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
   - The function :func:`nilearn.connectome.sym_to_vec` in same module
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
   - Documentation about :class:`nilearn.input_data.NiftiSpheresMasker`.

Bug fixes
.........
   - Fixed bug to control the behaviour when cut_coords=0. in function
     :func:`nilearn.plotting.plot_stat_map` in :mod:`nilearn.plotting`.
     See issue # 784.
   - Fixed bug in :func:`nilearn.image.copy_img` occured while caching
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
