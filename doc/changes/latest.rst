
.. currentmodule:: nilearn

.. include:: names.rst

0.9.1.dev
=========

NEW
---

- :func:`~mass_univariate.permuted_ols` and :func:`~glm.second_level.non_parametric_inference` now support cluster-level FWE correction (:gh:`3181` by `Taylor Salo`_).

Fixes
-----

- Fix function :func:`~datasets.fetch_abide_pcp` which was returning empty phenotypes and ``func_preproc`` after release ``0.9.0`` due to supporting pandas dataframes in fetchers (:gh:`3174` by `Nicolas Gensollen`_).

Enhancements
------------

- Function :func:`~plotting.plot_carpet` now accepts a ``t_r`` parameter, which allows users to provide the TR of the image when the image's header may not be accurate. (:gh:`3165` by `Taylor Salo`_).
- Terms :term:`Probabilistic atlas` and :term:`Deterministic atlas` were added to the glossary and references were added to atlas fetchers (:gh:`3152` by `Nicolas Gensollen`_).

Changes
-------

