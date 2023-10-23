.. currentmodule:: nilearn

.. include:: names.rst

0.11.0.dev
==========

HIGHLIGHTS
----------

NEW
---

Fixes
-----

- :bdg-success:`API` :class:`~maskers.MultiNiftiMasker` can now call :meth:`~maskers.NiftiMasker.generate_report` which will generate a report for the first subject in the list of subjects (:gh:`4001` by `Yasmin Mzayek`_).
- :bdg-dark:`Code` Fix :class:`~nilearn.glm.regression.SimpleRegressionResults` to accommodate for the lack of a ``model`` attribute (:gh:`4071` `Remi Gau`_)

Enhancements
------------

- :bdg-primary:`Doc` Add backslash to homogenize :class:`~nilearn.regions.Parcellations` documentation (:gh:`4042` by `Nikhil Krish`_).

- Allow setting ``vmin`` in :func:`~nilearn.plotting.plot_glass_brain` and :func:`~nilearn.plotting.plot_stat_map` (:gh:`3993` by `Michelle Wang`_).

Changes
-------
