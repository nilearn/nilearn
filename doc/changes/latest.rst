.. currentmodule:: nilearn

.. include:: names.rst

0.11.0.dev
==========

NEW
---

Fixes
-----

- :bdg-dark:`Code` Fix failing test in ``test_nilearn_standardize`` on MacOS 14 by adding trend in simulated data (:gh:`4411` by `Hao-Ting Wang`_).

Enhancements
------------

- :bdg-success:`API` Add option to resize output image width ``width_view`` in :func:`nilearn.plotting.view_img` (:gh:`4416` by `Alexandre Sayal`_).

Changes
-------

- :bdg-dark:`Code` Remove the unused argument ``url`` from  :func:`nilearn.datasets.fetch_localizer_contrasts`, :func:`nilearn.datasets.fetch_localizer_calculation_task` and :func:`nilearn.datasets.fetch_localizer_button_task` (:gh:`4273` by `Rémi Gau`_).

- :bdg-dark:`Code` Remove the unused argument ``rank`` from the constructor of :class:`nilearn.glm.LikelihoodModelResults` (:gh:`4273` by `Rémi Gau`_).

- :bdg-dark:`Code` Implement argument ``sample_mask`` for :meth:`nilearn.maskers.MultiNiftiMasker.transform_imgs` (:gh:`4273` by `Rémi Gau`_).

- :bdg-dark:`Code` Remove the unused arguments ``upper_cutoff`` and ``exclude_zeros`` for :func:`nilearn.masking.compute_multi_background_mask` (:gh:`4273` by `Rémi Gau`_).

- :bdg-dark:`Code` Throw error in :func:`nilearn.glm.first_level.first_level_from_bids` if unknown ``kwargs`` are passed (:gh:`4414` by `Michelle Wang`_).

- :bdg-primary:`Doc` Refactor design matrix and contrast formula for the two-sample T-test example in :ref:`sphx_glr_auto_examples_05_glm_second_level_plot_second_level_two_sample_test` (:gh:`4407` by `Yichun Huang`_).
