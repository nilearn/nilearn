.. currentmodule:: nilearn

.. include:: names.rst

0.11.0.dev
==========

HIGHLIGHTS
----------

.. warning::

 | **Support for Python 3.8 has been dropped.**
 | **We recommend upgrading to Python 3.11 or above.**
 |
 | **Minimum supported versions of the following packages have been bumped up:**
 | - numpy -- 1.22.4
 | - nibabel -- 5.2.0
 | - scikit-learn -- 1.4.0
 | - joblib -- 1.2.0
 | - pandas -- 2.2.0

NEW
---

Fixes
-----

- :bdg-dark:`Code` Make sure that radiological view is applied when requested and not only when figures are annotated (:gh:`4556` by `Rémi Gau`_).

- :bdg-dark:`Code` Fix failing test in ``test_nilearn_standardize`` on MacOS 14 by adding trend in simulated data (:gh:`4411` by `Hao-Ting Wang`_).

- :bdg-dark:`Code` Add a new attribute ``_region_id_name`` to :class:`nilearn.maskers.NiftiLabelsMasker` which is used to fix the issue with creating ``region_names_`` attribute (:gh:`4360` by `Mohammad Torabi`_).

- :bdg-dark:`Code` Fix previous Glover HRF implementation to fit the original paper (Glover, 1999) (:gh:`4452` by `Kun CHEN`_).

- :bdg-dark:`Code` :func:`nilearn.image.binarize_img` explicitly cast images to ``int8`` to avoid warnings about ``int64`` when working with ``float64`` images (:gh:`4498` by `Patrick Sadil`_).

- :bdg-dark:`Code` Fix bug that would lead ``compute_contrast`` to return 4D images even for one dimensional contrasts (:gh:`4413` by `Bertrand Thirion`_ and `Rémi Gau`_).

- :bdg-dark:`Code` ``first_level_from_bids`` will now return subjects in order (:gh:`4582` by `Tharun K`_).

- :bdg-dark:`Code` Fix PTH errors: PTH110, PTH107, PTH102, PTH113, PTH111, PTH101, PTH201, PTH109, PTH115 (:gh:`4620` by `Prakhar Jain`_).

- :bdg-dark:`Code` Fix PTH errors: PTH106, PTH112, PTH114, PTH117, PTH122, PTH202, PTH203, PTH204 (:gh:`4607`, :gh:`4612`, :gh:`4590`, :gh:`4618` by `Hande Gözükan`_).


Enhancements
------------

- :bdg-primary:`Doc` Add example showing how to compute hemisphere-wise connectivity for Yeo 17 networks (:gh:`4585` by `Victoria Shevchenko`_).

- :bdg-primary:`Doc` Add an option in :func:`nilearn.datasets.fetch_atlas_aal` to fetch the latest AAL version, 3v2 (:gh:`4554` by `Jeremy Lefort-Besnard`_ and `Rémi Gau`_).

- :bdg-dark:`Code` Improve input/output for ``SurfaceImage`` by loading meshes from files on disk, loading data from files or Nifti object, and saving meshes to file (:gh:`4446`, :gh:`4593` by `Rémi Gau`_ and `Jerome Dockes`_).

- :bdg-dark:`Code` Add a new function :func:`nilearn.plotting.plot_design_matrix_correlation` to plot the correlation between regressors of a GLM design matrix (:gh:`4467` by `Rémi Gau`_).

- :bdg-dark:`Code` Allow :class:`nilearn.decoding.Decoder` and :class:`nilearn.decoding.DecoderRegressor` to work with surface objects (:gh:`4205` by `Yasmin Mzayek`_ and `Rémi Gau`_).

- :bdg-success:`API` Add option to resize output image width ``width_view`` in :func:`nilearn.plotting.view_img` (:gh:`4416` by `Alexandre Sayal`_).

- :bdg-primary:`Doc` Add example to demonstrate the use of the new ``copy_header_from`` parameter in :func:`nilearn.image.math_img` (:gh:`4392` by `Himanshu Aggarwal`_).

- :bdg-primary:`Doc` Adapt examples showing how to plot events and design matrices to show how to use parametric modulation. Also implement modulation of events in :func:`nilearn.plotting.plot_event` (:gh:`4436` by `Rémi Gau`_).

- :bdg-dark:`Code` Add footer to masker reports (:gh:`4307` by `Rémi Gau`_).

- :bdg-info:`Plotting` Improve plotting contours for :class:`nilearn.plotting.displays.PlotlySurfaceFigure` objects by adding :meth:`nilearn.plotting.displays.PlotlySurfaceFigure.add_contours` method that accepts arguments to adjust line aesthetics (:gh:`3949` by `Patrick Sadil`_).

- :bdg-primary:`Doc` Add example to provide a clear understanding of the :class:`nilearn.decoding.Decoder` object by demonstrating underlying steps via a Scikit-Learn pipeline. (:gh:`4437` by `Himanshu Aggarwal`_).

Changes
-------

- :bdg-secondary:`Maint` Replace ``pytest.warns(DeprecationWarning)`` with ``pytest.deprecated_call()`` in tests (:gh:`4637` by `Victoria Shevchenko`_).

- :bdg-dark:`Code` Warn the user when all volumes would be scrubbed when loading fmriprep confounds as this would lead to an empty ``sample_mask`` (:gh:`4558` by `Victoria Shevchenko`_).

- :bdg-dark:`Code` Throw error if ``sample_mask`` is empty when scrubbing an fMRI time series (:gh:`4558` by `Victoria Shevchenko`_).

- :bdg-dark:`Code` Remove the unused argument ``url`` from  :func:`nilearn.datasets.fetch_localizer_contrasts`, :func:`nilearn.datasets.fetch_localizer_calculation_task` and :func:`nilearn.datasets.fetch_localizer_button_task` (:gh:`4273` by `Rémi Gau`_).

- :bdg-dark:`Code` Remove the unused argument ``rank`` from the constructor of :class:`nilearn.glm.LikelihoodModelResults` (:gh:`4273` by `Rémi Gau`_).

- :bdg-dark:`Code` Use ruff as formatter and linter instead of black, isort, flake8... (:gh:`4574` by `Rémi Gau`_).

- :bdg-dark:`Code` Implement argument ``sample_mask`` for :meth:`nilearn.maskers.MultiNiftiMasker.transform_imgs` (:gh:`4273` by `Rémi Gau`_).

- :bdg-dark:`Code` Remove the unused arguments ``upper_cutoff`` and ``exclude_zeros`` for :func:`nilearn.masking.compute_multi_background_mask` (:gh:`4273` by `Rémi Gau`_).

- :bdg-dark:`Code` Throw error in :func:`nilearn.glm.first_level.first_level_from_bids` if unknown ``kwargs`` are passed (:gh:`4414` by `Michelle Wang`_).

- :bdg-dark:`Code` Improve logging by relying only on the Nilearn logger and adding optional support for rich printing if `rich <https://github.com/Textualize/rich>`_ is installed (:gh:`4469` and :gh:`4544` by `Rémi Gau`_).

- :bdg-dark:`Code` Parcellations returned by :class:`nilearn.regions.Parcellations` will now be of type ``np.int32`` to avoid unnecessary warnings (:gh:`4555` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``tr`` for :term:`Repetition time<TR>` will be replaced by ``t_r`` in the "HRF" functions in version 0.13.0. The affected functions are :func:`nilearn.glm.first_level.glover_dispersion_derivative`, :func:`nilearn.glm.first_level.glover_hrf`, :func:`nilearn.glm.first_level.glover_time_derivative`, :func:`nilearn.glm.first_level.spm_dispersion_derivative`, :func:`nilearn.glm.first_level.spm_hrf`, :func:`nilearn.glm.first_level.spm_time_derivative` (:gh:`4470` by `Rémi Gau`_).

- :bdg-primary:`Doc` Refactor design matrix and contrast formula for the two-sample T-test example in :ref:`sphx_glr_auto_examples_05_glm_second_level_plot_second_level_two_sample_test.py` (:gh:`4407` by `Yichun Huang`_).

- :bdg-success:`API` The default for ``force_resample`` in :func:`nilearn.image.resample_img` and :func:`nilearn.image.resample_to_img` will be set to ``True`` from Nilearn 0.13.0. (:gh:`4412` by `Rémi Gau`_ and `Anand Joshi`_)

- :bdg-success:`API` Allow users to control copying header to the output in :func:`nilearn.image` functions and add future warning to copy headers by default in release 0.13.0 onwards (:gh:`4397` by `Himanshu Aggarwal`_).

- :bdg-dark:`Code` Extend coverage for data generating utility functions (:gh:`4465` by `Sin Kim`_).

- :bdg-danger:`Deprecation` The parameter ``ax`` will be replaced by ``axes`` in :func:`nilearn.plotting.plot_contrast_matrix` and :func:`nilearn.plotting.plot_design_matrix` in release 0.13.0. (:gh:`4476` by `Mudassir Chapra`_)

- :bdg-dark:`Code` Reorder condition in internal call of :func:`nilearn.image.resample_img` to skip checking of array values if interpolation is ``nearest`` (:gh:`4571` by `Jason Kai`_).

- :bdg-dark:`Code` Remove redundant sorting of ``np.unique(data)`` in internal call of :func:`nilearn.image.resample_img` when checking array values (:gh:`4571` by `Jason Kai`_).

- :bdg-primary:`Doc` Add missing default values to the docstrings in 'nilearn/glm'(:gh:`4656` by `Anupriya Kumari`_).
