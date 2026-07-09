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

- :bdg-secondary:`Maint` Add return type annotations to :func:`~interfaces.fsl.get_design_from_fslmat`, :func:`~interfaces.bids.parse_bids_filename`, :func:`~interfaces.fmriprep.load_confounds`, and :func:`~interfaces.fmriprep.load_confounds_strategy` (:gh:`6362` by `Rémi Gau`_).

- :bdg-secondary:`Maint` Add return type annotations and :obj:`~typing.overload` signatures to :func:`~connectome.vec_to_sym_matrix`, :func:`~connectome.group_sparse_covariance`, and :func:`~reporting.get_clusters_table` (:gh:`6368` by `Rémi Gau`_).

Enhancements
------------

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for one function in the public API: :func:`~nilearn.masking.compute_epi_mask` (:gh:`6306` by `Marco Flores`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section to :func:`~nilearn.utils.all_displays`, :func:`~nilearn.utils.all_estimators`, :func:`~nilearn.utils.all_functions` (:gh:`6322`, :gh:`6324`, :gh:`6325` by `Alice Schiavone`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for a utility function in the public API: :func:`~nilearn.image.iter_img` (:gh:`6304` by `Ruben Dörfel`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.plotting.plot_design_matrix` (:gh:`6380` by `Nirmitee Mulay`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.signal.butterworth` function (:gh:`6311` by `Hande Gözükan`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.regions.img_to_signals_labels` function (:gh:`6315` by `Hande Gözükan`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.plotting.plot_design_matrix_correlation` function (:gh:`6415` by `Nirmitee Mulay`_).

Changes
-------

- :bdg-dark:`Code` Add return type annotations to public functions in :mod:`nilearn.glm` (:gh:`6370` by `Rémi Gau`_).

- :bdg-dark:`Code` Add return type annotations to public functions in :mod:`nilearn.regions` (:gh:`6369` by `Rémi Gau`_).

- :bdg-dark:`Code` Update plotting functions to return figure or axes instead of None when an output file is specified to save the figure (:gh:`6272` by `Hande Gözükan`_).
