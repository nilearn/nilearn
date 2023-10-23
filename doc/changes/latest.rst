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

Enhancements
------------

- Allow setting ``vmin`` in :func:`~nilearn.plotting.plot_glass_brain` and :func:`~nilearn.plotting.plot_stat_map` (:gh:`3993` by `Michelle Wang`_).
- Support passing t and F contrasts to :func:`~nilearn.glm.contrasts.compute_contrast` that that have fewer columns than the number of estimated parameters. Remaining columns are padded with zero (:gh:`4067` by `Remi Gau`_).

Changes
-------
