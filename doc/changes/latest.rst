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

- :bdg-secondary:`Maint` Add return type annotations and :obj:`~typing.overload` signatures to :func:`~connectome.vec_to_sym_matrix`, :func:`~connectome.group_sparse_covariance`, and :func:`~reporting.get_clusters_table` (:gh:`6368` by `Rémi Gau`_).

Enhancements
------------

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for one function in the public API: :func:`~nilearn.masking.compute_epi_mask` (:gh:`6306` by `Marco Flores`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section to :func:`~nilearn.utils.all_displays`, :func:`~nilearn.utils.all_estimators`, :func:`~nilearn.utils.all_functions` (:gh:`6322`, :gh:`6324`, :gh:`6325` by `Alice Schiavone`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for a utility function in the public API: :func:`~nilearn.image.iter_img` (:gh:`6304` by `Ruben Dörfel`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.regions.img_to_signals_labels` function (:gh:`6315` by `Hande Gözükan`_).

Changes
-------

- :bdg-dark:`Code` Add ``asv`` benchmark for TFCE computation (:gh:`6394` by `Fabricio Cravo`_).

- :bdg-dark:`Code` Add return type annotations to public functions in :mod:`nilearn.regions` (:gh:`6369` by `Rémi Gau`_).
