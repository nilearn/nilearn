
.. currentmodule:: nilearn

.. include:: names.rst

0.9.1.dev
=========

NEW
---


Fixes
-----

- Fix function :func:`~datasets.fetch_abide_pcp` which was returning empty phenotypes and ``func_preproc`` after release ``0.9.0`` due to supporting pandas dataframes in fetchers (:gh:`3174` by `Nicolas Gensollen`_).
- Fix function :func:`~datasets.fetch_atlas_harvard_oxford` and :func:`~datasets.fetch_atlas_juelich` which were returning the image in the `filename` attribute instead of the path to the image (:gh:`3179` by `Raphael Meudec`_).
- Fix colorbars in :func:`~plotting.plot_stat_map`, :func:`~plotting.plot_glass_brain` and :func:`~plotting.plot_surf_stat_map` which could extend beyond the figure for users with newest matplotlib version (>=3.5.1) (:gh:`3188` by `Raphael Meudec`_)

Enhancements
------------

- Function :func:`~plotting.plot_carpet` now accepts a ``t_r`` parameter, which allows users to provide the TR of the image when the image's header may not be accurate. (:gh:`3165` by `Taylor Salo`_).
- Terms :term:`Probabilistic atlas` and :term:`Deterministic atlas` were added to the glossary and references were added to atlas fetchers (:gh:`3152` by `Nicolas Gensollen`_).

Changes
-------

