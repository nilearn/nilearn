.. currentmodule:: nilearn

Version 0.14.1dev
=================

..
    Each changelog entry should begin with one of the following badges:

    - :bdg-primary:`Doc`
    - :bdg-secondary:`Maint`
    - :bdg-success:`API`
    - :bdg-info:`Plotting`
    - :bdg-warning:`Test`
    - :bdg-danger:`Deprecation`
    - :bdg-dark:`Code`

NEW
---

Fixes
-----

- :bdg-primary:`Doc` Fix docstrings that name a parameter which is not in the signature (:func:`~plotting.plot_img_comparison`, :meth:`~plotting.displays.PlotlySurfaceFigure.add_contours` and seven private helpers), the documented default of ``two_sided_test`` in ``calculate_tfce``, and the sentence stating that a source image is resampled to match itself (:gh:`6469` by `Anton Karpov`_).

- :bdg-dark:`Code` Fix the masked atlas returned by :func:`~regions.img_to_signals_labels` and exposed as ``NiftiLabelsMasker.region_atlas_`` being cast to ``int8``, which wrapped every label above 127 around and aliased label 256 onto the background; atlases such as Schaefer-400 label well past that (:gh:`6447` by `Andrew Chen`_).

- :bdg-info:`Plotting` Fix Brainsprite figures becoming blank or incorrect when multiple masker reports are embedded in the same HTML document by giving each viewer unique DOM element IDs (:gh:`6419` by `Mohammad Sadeghi Hardengi`_).

- :bdg-dark:`Code` Fix :func:`~regions.connected_label_regions` attaching the names given in ``labels`` to the wrong regions, because the sorted labels from ``np.unique`` were put through a ``set`` before being zipped against the names; contiguous labels happened to survive that, but the sparse labels real atlases use did not (:gh:`6445` by `Andrew Chen`_).

- :bdg-dark:`Code` Fix :class:`~maskers.NiftiMapsMasker` silently extracting a zero signal for a map whose weights are negative, because ``_trim_maps`` decides which maps to keep sign-agnostically with ``abs()`` but then builds their support with ``> 0``, so a kept negative map covered no voxel; this matters for the signed ICA and statistical maps the masker targets (:gh:`6443` by `Andrew Chen`_).

- :bdg-dark:`Code` Fix :func:`~image.smooth_img` and ``smooth_array`` truncating the smoothed signal for unsigned integer input, because only signed integers were promoted to float before ``gaussian_filter1d`` wrote its float result back into the input buffer in place; unsigned is the common case since ``uint8`` is the standard on-disk dtype for masks and atlases (:gh:`6440` by `Andrew Chen`_).

- :bdg-dark:`Code` Fix ``examples/04_glm_first_level/plot_bids_features.py``, ``doc/get_data_examples.py``, and ``doc/visual_testing/reporter_visual_inspection_suite.py`` downloading the full FSL derivatives for the ``bart`` and ``taskswitch`` tasks of the ``ds000030`` dataset in addition to ``stopsignal``, because the ``*task-task*`` exclusion filter did not match the ``derivatives/task/sub-*/taskswitch.feat`` folder name; switch to ``inclusion_filters`` to only fetch the files needed for the ``stopsignal`` analysis (:gh:`6432` by `Rémi Gau`_).

- :bdg-dark:`Code` Fix :func:`~interfaces.bids.get_bids_files` missing subject-level anatomical derivatives in multi-session datasets (:gh:`6291` by `Mohammad Sadeghi Hardengi`_).

- :bdg-dark:`Code` Fix a bug in :func:`~plotting.view_img` where the background colormap was not being applied correctly. (:gh:`6466` by `Pierre-Louis Barbarant`_).

- :bdg-dark:`Code` Fix :func:`~image.smooth_img` and ``smooth_array`` truncating the smoothed signal for unsigned integer input, because only signed integers were promoted to float before ``gaussian_filter1d`` wrote its float result back into the input buffer in place; unsigned is the common case since ``uint8`` is the standard on-disk dtype for masks and atlases (:gh:`6440` by `Andrew Chen`_).

- :bdg-dark:`Code` Allow custom scikit-learn-compatible estimators in decoders to use an empty default parameter grid, and clarify how to use ``param_grid`` to tune them (:gh:`6227` by `Mohammad Sadeghi Hardengi`_).

- :bdg-dark:`Code` Fix :func:`~image.resample_img` raising an ``AttributeError`` instead of resampling correctly when ``target_affine`` is passed as a :obj:`list` or :obj:`tuple` together with ``target_shape`` (:gh:`6408` by `Rémi Gau`_).

- :bdg-dark:`Code` Allow :func:`~glm.first_level.first_level_from_bids` to work with BIDS dataset that have a single events file in the root of the dataset for all runs (:gh:`6278` by `Rémi Gau`_).

- :bdg-secondary:`Maint` Add return type annotations and :obj:`~typing.overload` signatures to :func:`~connectome.vec_to_sym_matrix`, :func:`~connectome.group_sparse_covariance`, and :func:`~reporting.get_clusters_table` (:gh:`6368` by `Rémi Gau`_).

- :bdg-primary:`Doc` Fix eleven docstrings naming a parameter the function does not take, including the typos ``lispchitz_constant`` and ``flag_tedata``, and a pair of neighboring helpers documenting each other's parameter names (:gh:`6474` by `Anton Karpov`_).


Enhancements
------------

- :bdg-dark:`Code` Improve type annotations (and :obj:`~typing.overload` signatures where the return type depends on the arguments given) in :mod:`nilearn.glm` (:gh:`6370`), :mod:`nilearn.regions` (:gh:`6369`), :mod:`nilearn.connectome` (:gh:`6368`), :mod:`nilearn.reporting` (:gh:`6368`), :mod:`nilearn.interfaces` (:gh:`6362`), :mod:`nilearn.image` (:gh:`6408`, :gh:`6438`), :mod:`nilearn.utils`, :mod:`nilearn.surface` (:gh:`6410`), :mod:`nilearn.datasets`  (:gh:`6438`),  :mod:`nilearn.plotting` (:gh:`6438` and :gh:`6439`),  :mod:`nilearn.glm` and :mod:`nilearn.mass_univariate` (:gh:`6439`) (by `Rémi Gau`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for one function in the public API: :func:`~nilearn.masking.compute_epi_mask` (:gh:`6306` by `Marco Flores`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section to :func:`~nilearn.utils.all_displays`, :func:`~nilearn.utils.all_estimators`, :func:`~nilearn.utils.all_functions` (:gh:`6322`, :gh:`6324`, :gh:`6325` by `Alice Schiavone`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for a utility function in the public API: :func:`~nilearn.image.iter_img` (:gh:`6304` by `Ruben Dörfel`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.plotting.plot_design_matrix` (:gh:`6380` by `Nirmitee Mulay`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.signal.butterworth` function (:gh:`6311` by `Hande Gözükan`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.regions.img_to_signals_labels` function (:gh:`6315` by `Hande Gözükan`_).

Changes
-------

- :bdg-dark:`Code` Add ``asv`` benchmark for TFCE computation (:gh:`6394` by `Fabricio Cravo`_).

- :bdg-dark:`Code` Update plotting functions to return figure or axes instead of None when an output file is specified to save the figure (:gh:`6272` by `Hande Gözükan`_).
