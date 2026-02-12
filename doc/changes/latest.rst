.. currentmodule:: nilearn

.. include:: names.rst

0.14.0.dev
==========

NEW
---

Fixes
-----

- :bdg-dark:`Code` Ensure ``inverse_transform`` for :class:`~maskers.NiftiMapsMasker` and :class:`~maskers.NiftiLabelsMasker` expects the correct number of features when some maps / labels were dropped at fit or transform time (:gh:`5963` by `Rémi Gau`_).



Enhancements
------------

- :bdg-success:`API` Support pathlike objects for ``cmap`` argument in :func:`~plotting.plot_surf_roi` (:gh:`5981` by `Joseph Paillard`_).

Changes
-------
