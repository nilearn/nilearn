================================================
Reference documentation: all nistats functions
================================================

This is the class and function reference of nistats. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and function raw specifications may not be enough to give full guidelines on their
uses.

.. contents:: **List of modules**
   :local:


.. _datasets_ref:

:mod:`nistats.datasets`: Datasets
====================================================

.. automodule:: nistats.datasets
   :no-members:
   :no-inherited-members:

**Functions**:

.. currentmodule:: nistats.datasets

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fetch_language_localizer_demo_dataset
   fetch_openneuro_dataset_index
   select_from_index
   fetch_openneuro_dataset
   fetch_localizer_first_level
   fetch_spm_auditory
   fetch_spm_multimodal_fmri
   fetch_fiac_first_level

.. _hemodynamic_models_ref:

:mod:`nistats.hemodynamic_models`: Hemodynamic Models
=======================================================

.. automodule:: nistats.hemodynamic_models
   :no-members:
   :no-inherited-members:

**Functions**:

.. currentmodule:: nistats.hemodynamic_models

.. autosummary::
   :toctree: generated/
   :template: function.rst

   spm_hrf
   glover_hrf
   spm_time_derivative
   glover_time_derivative
   spm_dispersion_derivative
   glover_dispersion_derivative
   compute_regressor

.. _design_matrix_ref:

:mod:`nistats.design_matrix`: Design Matrix Creation
=====================================================

.. automodule:: nistats.design_matrix
   :no-members:
   :no-inherited-members:

**Functions**:

.. currentmodule:: nistats.design_matrix

.. autosummary::
   :toctree: generated/
   :template: function.rst

   make_first_level_design_matrix
   check_design_matrix
   make_second_level_design_matrix

.. _experimental_paradigm_ref:

:mod:`nistats.experimental_paradigm`: Experimental Paradigm
============================================================

.. automodule:: nistats.experimental_paradigm
   :no-members:
   :no-inherited-members:

**Functions**:

.. currentmodule:: nistats.experimental_paradigm

.. autosummary::
   :toctree: generated/
   :template: function.rst

   check_events

.. _model_ref:

:mod:`nistats.model`: Statistical models
====================================================

.. automodule:: nistats.model
   :no-members:
   :no-inherited-members:

**Classes**:

.. currentmodule:: nistats.model

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LikelihoodModelResults
   TContrastResults
   FContrastResults

.. _regression_ref:

:mod:`nistats.regression`: Regression Models
====================================================

.. automodule:: nistats.regression
   :no-members:
   :no-inherited-members:

**Classes**:

.. currentmodule:: nistats.regression

.. autosummary::
   :toctree: generated/
   :template: class.rst

   OLSModel
   ARModel
   RegressionResults
   SimpleRegressionResults

.. _first_level_models_ref:

:mod:`nistats.first_level_model`: First Level Model
====================================================

.. automodule:: nistats.first_level_model
   :no-members:
   :no-inherited-members:

**Classes**:

.. currentmodule:: nistats.first_level_model

.. autosummary::
   :toctree: generated/
   :template: class.rst

   FirstLevelModel

**Functions**:

.. currentmodule:: nistats.first_level_model

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mean_scaling
   run_glm
   first_level_models_from_bids

.. _second_level_model_ref:

:mod:`nistats.second_level_model`: Second Level Model
======================================================

.. automodule:: nistats.second_level_model
   :no-members:
   :no-inherited-members:

**Classes**:

.. currentmodule:: nistats.second_level_model

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SecondLevelModel

.. _contrasts_ref:

:mod:`nistats.contrasts`: Contrasts
====================================================

.. automodule:: nistats.contrasts
   :no-members:
   :no-inherited-members:

**Classes**:

.. currentmodule:: nistats.contrasts

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Contrast

**Functions**:

.. currentmodule:: nistats.contrasts

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_contrast

.. _thresholding_ref:

:mod:`nistats.thresholding`: Thresholding Maps
====================================================

.. automodule:: nistats.thresholding
   :no-members:
   :no-inherited-members:

**Functions**:

.. currentmodule:: nistats.thresholding

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fdr_threshold
   map_threshold

.. _reporting_ref:

:mod:`nistats.reporting`: Report plotting functions
====================================================

.. automodule:: nistats.reporting
   :no-members:
   :no-inherited-members:

**Functions**:

.. currentmodule:: nistats.reporting

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compare_niimgs
   plot_design_matrix
   plot_contrast_matrix
   get_clusters_table
   make_glm_report

.. _utils_ref:

:mod:`nistats.utils`: Utility functions
====================================================

.. automodule:: nistats.utils
   :no-members:
   :no-inherited-members:

**Functions**:

.. currentmodule:: nistats.utils

.. autosummary::
   :toctree: generated/
   :template: function.rst

   z_score
   multiple_fast_inverse
   multiple_mahalanobis
   full_rank
   positive_reciprocal
   get_bids_files
   parse_bids_filename
