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

   822  Elvis Dohmatob    (hasn't learn squashing yet)
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
