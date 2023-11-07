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

Changes
-------

- :bdg-danger:`Deprecation` :func:`~regions.img_to_signals_labels` will also return ``masked_atlas`` in release 0.15. Meanwhile, use ``return_masked_atlas`` parameter to enable/disable this behavior. (:gh:`3761` by `Mohammad Torabi`_).
