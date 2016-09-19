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

   - Sphinx-gallery updated to version 0.1.3

   - Refactoring of examples and documentation

   - Better ordering of regions in
     :func:`nilearn.datasets.fetch_coords_dosenbach_2010`

   - Remove outdated power atlas example

   
API changes summary
...................

    - The parameter 'n_subjects' is deprecated and will be removed in future
      release. Use 'subjects' instead in :func:`nilearn.datasets.fetch_haxby`

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
