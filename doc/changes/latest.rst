.. currentmodule:: nilearn

.. include:: names.rst

0.14.1dev
=========

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

- :bdg-dark:`Code` Fix the deprecated ``nilearn.interfaces.bids.save_glm_to_bids`` redirect silently returning ``None`` instead of the fitted model, because it called the relocated :func:`~glm.save_glm_to_bids` without returning its result (:gh:`6439` by `Rémi Gau`_).

- :bdg-dark:`Code` Fix ``examples/04_glm_first_level/plot_bids_features.py``, ``doc/get_data_examples.py``, and ``doc/visual_testing/reporter_visual_inspection_suite.py`` downloading the full FSL derivatives for the ``bart`` and ``taskswitch`` tasks of the ``ds000030`` dataset in addition to ``stopsignal``, because the ``*task-task*`` exclusion filter did not match the ``derivatives/task/sub-*/taskswitch.feat`` folder name; switch to ``inclusion_filters`` to only fetch the files needed for the ``stopsignal`` analysis (:gh:`6432` by `Rémi Gau`_).

- :bdg-warning:`Test` Fix the ``test_html`` GitHub Actions workflow to set ``NILEARN_DATA`` to the same path used to restore its dataset cache, and add ``NILEARN_DATA`` to ``tox.ini``'s shared ``passenv`` list so ``tox`` actually forwards it to the ``generate_html``/``test_html`` environments, since without it nilearn's fetchers default to ``~/nilearn_data`` instead and every dataset was silently re-downloaded from the network on every run regardless of whether the cache was restored (:gh:`6427` by `Rémi Gau`_).

- :bdg-dark:`Code` Fix the ``build-docs`` GitHub Actions workflow so the monthly dataset cache is only saved once per month, by a full build, instead of on every run, which was causing it to be evicted from the repository's Actions cache quota (or frozen with only a partial-build subset of the data) and forcing datasets such as ``development_fmri`` and ``difumo_atlases`` to be re-downloaded live from OSF, a frequent source of flaky ``test_html`` and ``build-docs`` failures (:gh:`6425` by `Rémi Gau`_).

- :bdg-dark:`Code` Allow custom scikit-learn-compatible estimators in decoders to use an empty default parameter grid, and clarify how to use ``param_grid`` to tune them (:gh:`6227` by `Mohammad Sadeghi Hardengi`_).

- :bdg-dark:`Code` Fix :func:`~image.resample_img` raising an ``AttributeError`` instead of resampling correctly when ``target_affine`` is passed as a :obj:`list` or :obj:`tuple` together with ``target_shape`` (:gh:`6408` by `Rémi Gau`_).

- :bdg-primary:`Doc` Fix the "View this page" and "Edit this page" buttons leading to a 404 on autosummary-generated API reference pages (e.g. :func:`~image.concat_imgs`), by hiding them there since those ``.rst`` files are generated at build time and never committed to the repository (:gh:`6435` by `Rémi Gau`_).

- :bdg-secondary:`Maint` Add return type annotations to :func:`~interfaces.fsl.get_design_from_fslmat`, :func:`~interfaces.bids.parse_bids_filename`, :func:`~interfaces.fmriprep.load_confounds`, and :func:`~interfaces.fmriprep.load_confounds_strategy` (:gh:`6362` by `Rémi Gau`_).

- :bdg-dark:`Code` Allow :func:`~glm.first_level.first_level_from_bids` to work with BIDS dataset that have a single events file in the root of the dataset for all runs (:gh:`6278` by `Rémi Gau`_).

- :bdg-secondary:`Maint` Add return type annotations and :obj:`~typing.overload` signatures to :func:`~connectome.vec_to_sym_matrix`, :func:`~connectome.group_sparse_covariance`, and :func:`~reporting.get_clusters_table` (:gh:`6368` by `Rémi Gau`_).

Enhancements
------------

- :bdg-primary:`Doc` Clarify the "Performance monitoring" section of ``CONTRIBUTING.rst``, extend ``asv_benchmarks/hashestobenchmark.txt`` with all major releases since 0.8, and add benchmarks for :func:`~nilearn.plotting.plot_design_matrix_correlation` and :func:`~nilearn.utils.all_estimators` demonstrating the local-import pattern needed when benchmarking functions absent from older nilearn versions (:gh:`6422` by `Rémi Gau`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for one function in the public API: :func:`~nilearn.masking.compute_epi_mask` (:gh:`6306` by `Marco Flores`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section to :func:`~nilearn.utils.all_displays`, :func:`~nilearn.utils.all_estimators`, :func:`~nilearn.utils.all_functions` (:gh:`6322`, :gh:`6324`, :gh:`6325` by `Alice Schiavone`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for a utility function in the public API: :func:`~nilearn.image.iter_img` (:gh:`6304` by `Ruben Dörfel`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.plotting.plot_design_matrix` (:gh:`6380` by `Nirmitee Mulay`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.signal.butterworth` function (:gh:`6311` by `Hande Gözükan`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.regions.img_to_signals_labels` function (:gh:`6315` by `Hande Gözükan`_).

Changes
-------

- :bdg-dark:`Code` Add return type annotations to :func:`~datasets.fetch_icbm152_2009`, :func:`~datasets.fetch_icbm152_brain_gm_mask`, :func:`~datasets.fetch_oasis_vbm`, :func:`~image.crop_img`, :func:`~plotting.plot_carpet`, and to internal helpers ``nilearn.conftest.check_parameters_doctring``, ``nilearn.conftest.check_methods_docstring``, ``nilearn.image.image.smooth_array``, and ``nilearn.surface.surface.combine_hemispheres_meshes`` (:gh:`6438` by `Rémi Gau`_).

- :bdg-dark:`Code` Add return type annotations to :func:`~glm.first_level.check_design_matrix`, :func:`~glm.second_level.non_parametric_inference`, and :func:`~mass_univariate.permuted_ols`, with :obj:`~typing.overload` signatures for the latter two since their return type depends on their arguments (:gh:`6439` by `Rémi Gau`_).

- :bdg-dark:`Code` Add a return type annotation to :func:`~plotting.find_cut_slices` (:gh:`6439` by `Rémi Gau`_).

- :bdg-secondary:`Maint` Add a matrix job to the benchmark CI workflow that benchmarks each commit in ``asv_benchmarks/hashestobenchmark.txt`` in parallel, then combines the results into a single viewable artifact, instead of benchmarking them one after another in a single, slow job, and make the version-gated local imports in ``discovery.py`` and ``plotting.py`` raise ``NotImplementedError`` so asv reports them as skipped rather than failed on nilearn versions that lack the benchmarked function (:gh:`6430` by `Rémi Gau`_).

- :bdg-secondary:`Maint` Drop nilearn versions older than 0.11.0 from ``asv_benchmarks/hashestobenchmark.txt`` (they cannot currently be benchmarked, see ``CONTRIBUTING.rst``), make the benchmark CI workflow fail when a benchmark reports as failed instead of silently ignoring it, fix an always-failing ``IndexImgBenchmark`` slice bound that this newly surfaced, and reorganize ``asv_benchmarks/benchmarks/glm`` to mirror the structure of :mod:`nilearn.glm` (:gh:`6426` by `Rémi Gau`_).

- :bdg-dark:`Code` Add type annotations to the public functions in ``nilearn._utils.data_gen`` (:gh:`6420` by `Rémi Gau`_).

- :bdg-dark:`Code` Add return type annotations to the public methods of :class:`~surface.FileMesh`, :class:`~surface.InMemoryMesh`, :class:`~surface.PolyData`, :class:`~surface.PolyMesh`, :class:`~surface.SurfaceImage`, and :class:`~surface.SurfaceMesh` (:gh:`6410` by `Rémi Gau`_).

- :bdg-secondary:`Maint` Replace the ``nilearn/connectome`` ``D103`` glob ignore in ``pyproject.toml`` with per-file entries, add missing docstrings to test functions and fixtures in files that had 10 or fewer ``D103`` errors, and drop their now-unnecessary per-file ignores (:gh:`6406` by `Rémi Gau`_).

- :bdg-dark:`Code` Add return type annotations to :func:`~image.coord_transform`, :func:`~image.reorder_img`, :func:`~image.resample_img`, and :func:`~image.resample_to_img` (:gh:`6408` by `Rémi Gau`_).

- :bdg-dark:`Code` Add return type annotations to :func:`~utils.all_displays`, :func:`~utils.all_estimators`, and :func:`~utils.all_functions` (:gh:`6409` by `Rémi Gau`_).

- :bdg-dark:`Code` Add return type annotations to public functions in :mod:`nilearn.glm` (:gh:`6370` by `Rémi Gau`_).

- :bdg-dark:`Code` Add return type annotations to public functions in :mod:`nilearn.regions` (:gh:`6369` by `Rémi Gau`_).

- :bdg-dark:`Code` Update plotting functions to return figure or axes instead of None when an output file is specified to save the figure (:gh:`6272` by `Hande Gözükan`_).
