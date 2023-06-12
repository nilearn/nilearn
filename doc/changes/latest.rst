.. currentmodule:: nilearn

.. include:: names.rst

0.10.2.dev
==========

NEW
---

Fixes
-----

- Fix bug in `first_level_from_bids` that returned no confound files if the corresponding bold files contained derivaties BIDS entities (:gh:`3742` by `Rémi Gau`_).
- :bdg-dark:`Code` Fix bug where the `cv_params_` attribute of fitter Decoder objects sometimes had missing entries if `grid_param` is a sequence of dicts with different keys (:gh:`3733` by `Michelle Wang`_).

Enhancements
------------

- Update Decoder objects to use the more efficient ``LogisticRegressionCV`` (:gh:`3736` by `Michelle Wang`_).

Changes
-------

- Removed old files and test code from deprecated datasets COBRE and NYU resting state (:gh:`3743` by `Michelle Wang`_).
