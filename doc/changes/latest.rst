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

- :bdg-dark:`Code` Fix :func:`~image.smooth_img` and ``smooth_array`` truncating the smoothed signal for unsigned integer input, because only signed integers were promoted to float before ``gaussian_filter1d`` wrote its float result back into the input buffer in place; unsigned is the common case since ``uint8`` is the standard on-disk dtype for masks and atlases (:gh:`6440` by `Andrew Chen`_).

- :bdg-dark:`Code` Allow custom scikit-learn-compatible estimators in decoders to use an empty default parameter grid, and clarify how to use ``param_grid`` to tune them (:gh:`6227` by `Mohammad Sadeghi Hardengi`_).

- :bdg-dark:`Code` Fix :func:`~image.resample_img` raising an ``AttributeError`` instead of resampling correctly when ``target_affine`` is passed as a :obj:`list` or :obj:`tuple` together with ``target_shape`` (:gh:`6408` by `Rémi Gau`_).

- :bdg-dark:`Code` Allow :func:`~glm.first_level.first_level_from_bids` to work with BIDS dataset that have a single events file in the root of the dataset for all runs (:gh:`6278` by `Rémi Gau`_).

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
