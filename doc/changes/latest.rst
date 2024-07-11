.. currentmodule:: nilearn

.. include:: names.rst

0.11.0.dev
==========

NEW
---

Fixes
-----

- :bdg-dark:`Code` Fix failing test in ``test_nilearn_standardize`` on MacOS 14 by adding trend in simulated data (:gh:`4411` by `Hao-Ting Wang`_).
- :bdg-dark:`Code` Add a new attribute ``_region_id_name`` to :class:`nilearn.maskers.NiftiLabelsMasker` which is used to fix the issue with creating ``region_names_`` attribute (:gh:`4360` by `Mohammad Torabi`_).

- :bdg-dark:`Code` Fix previous Glover HRF implementation to fit the original paper (Glover, 1999) (:gh:`4452` by `Kun CHEN`_).

Enhancements
------------

- :bdg-dark:`Code` Add a new function :func:`nilearn.plotting.plot_design_matrix_correlation` to plot the correlation between regressors of a GLM design matrix (:gh:`4467` by `Rémi Gau`_).

- :bdg-success:`API` Add option to resize output image width ``width_view`` in :func:`nilearn.plotting.view_img` (:gh:`4416` by `Alexandre Sayal`_).

- :bdg-primary:`Doc` Add example to demonstrate the use of the new ``copy_header_from`` parameter in :func:`nilearn.image.math_img` (:gh:`4392` by `Himanshu Aggarwal`_).

- :bdg-primary:`Doc` Adapt examples showing how to plot events and design matrices to show how to use parametric modulation. Also implement modulation of events in :func:`nilearn.plotting.plot_event` (:gh:`4436` by `Rémi Gau`_).

- :bdg-dark:`Code` Add footer to masker reports (:gh:`4307` by `Rémi Gau`_).


Changes
-------

- :bdg-dark:`Code` Remove the unused argument ``url`` from  :func:`nilearn.datasets.fetch_localizer_contrasts`, :func:`nilearn.datasets.fetch_localizer_calculation_task` and :func:`nilearn.datasets.fetch_localizer_button_task` (:gh:`4273` by `Rémi Gau`_).

- :bdg-dark:`Code` Remove the unused argument ``rank`` from the constructor of :class:`nilearn.glm.LikelihoodModelResults` (:gh:`4273` by `Rémi Gau`_).

- :bdg-dark:`Code` Implement argument ``sample_mask`` for :meth:`nilearn.maskers.MultiNiftiMasker.transform_imgs` (:gh:`4273` by `Rémi Gau`_).

- :bdg-dark:`Code` Remove the unused arguments ``upper_cutoff`` and ``exclude_zeros`` for :func:`nilearn.masking.compute_multi_background_mask` (:gh:`4273` by `Rémi Gau`_).

- :bdg-dark:`Code` Throw error in :func:`nilearn.glm.first_level.first_level_from_bids` if unknown ``kwargs`` are passed (:gh:`4414` by `Michelle Wang`_).

- :bdg-danger:`Deprecation` The parameter ``tr`` for :term:`Repetition time<TR>` will be replaced by ``t_r`` in the "HRF" functions in version 0.13.0. The affected functions are :func:`nilearn.glm.first_level.glover_dispersion_derivative`, :func:`nilearn.glm.first_level.glover_hrf`, :func:`nilearn.glm.first_level.glover_time_derivative`, :func:`nilearn.glm.first_level.spm_dispersion_derivative`, :func:`nilearn.glm.first_level.spm_hrf`, :func:`nilearn.glm.first_level.spm_time_derivative` (:gh:`4470` by `Rémi Gau`_).


- :bdg-primary:`Doc` Refactor design matrix and contrast formula for the two-sample T-test example in :ref:`sphx_glr_auto_examples_05_glm_second_level_plot_second_level_two_sample_test.py` (:gh:`4407` by `Yichun Huang`_).

- :bdg-success:`API` The default for ``force_resample`` in :func:`nilearn.image.resample_img` and :func:`nilearn.image.resample_to_img` will be set to ``True`` from Nilearn 0.13.0. (:gh:`4412` by `Rémi Gau`_ and `Anand Joshi`_)

- :bdg-success:`API` Allow users to control copying header to the output in :func:`nilearn.image` functions and add future warning to copy headers by default in release 0.13.0 onwards (:gh:`4397` by `Himanshu Aggarwal`_).
