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
- :bdg-dark:`Code` :func:`~image.clean_img` can now accept a :var:`clean__sample_mask` argument that is passed into :func:`~signal.clean` to reshape the file to the dimensions of the mask (:gh:`4051` by 'Mia Zwally`_)
- :bdg-warning:`Test` added :func:`~test_clean_img_kwargs` to :file:`~image.test.py` to test functionality of calling :func:`~image.clean_img` with a :var:`clean__sample_mask` kwarg (:gh:`4051` by 'Mia Zwally`_)

Enhancements
------------

- Allow setting ``vmin`` in :func:`~nilearn.plotting.plot_glass_brain` and :func:`~nilearn.plotting.plot_stat_map` (:gh:`3993` by `Michelle Wang`_).

Changes
-------
