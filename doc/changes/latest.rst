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

Enhancements
------------

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for a utility function in the public API: :func:`~nilearn.image.iter_img` (:gh:`6304` by `Ruben Dörfel`_).

Changes
-------
