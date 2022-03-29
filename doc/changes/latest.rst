
.. currentmodule:: nilearn

.. include:: names.rst

0.9.1.dev
=========

NEW
---

:func:`~mass_univariate.permuted_ols` and :func:`~glm.second_level.non_parametric_inference` now support :term:`TFCE` statistic (:gh:`3196` by `Taylor Salo`_).

Fixes
-----

- Fix function :func:`~mass_univariate.permuted_ols`, which was only returning the null distribution (``h0_fmax``) for the first regressor (:gh:`3184` by `Taylor Salo`_).
- Fix function :func:`~datasets.fetch_abide_pcp` which was returning empty phenotypes and ``func_preproc`` after release ``0.9.0`` due to supporting pandas dataframes in fetchers (:gh:`3174` by `Nicolas Gensollen`_).
- Fix function :func:`~datasets.fetch_atlas_harvard_oxford` and :func:`~datasets.fetch_atlas_juelich` which were returning the image in the `filename` attribute instead of the path to the image (:gh:`3179` by `Raphael Meudec`_).

Enhancements
------------

- New example in
  :ref:`sphx_glr_auto_examples_07_advanced_plot_beta_series.py`
  to demonstrate how to implement common beta series models with nilearn (:gh:`3127` by `Taylor Salo`_).
- Function :func:`~plotting.plot_carpet` now accepts a ``t_r`` parameter, which allows users to provide the TR of the image when the image's header may not be accurate. (:gh:`3165` by `Taylor Salo`_).
- Terms :term:`Probabilistic atlas` and :term:`Deterministic atlas` were added to the glossary and references were added to atlas fetchers (:gh:`3152` by `Nicolas Gensollen`_).

Changes
-------

- Requirements files have been consolidated into a ``setup.cfg`` file and installation instructions have been simplified (:gh:`2953` by `Taylor Salo`_).
