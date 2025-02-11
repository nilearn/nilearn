.. _stats_ref:

:mod:`nilearn.glm`: Generalized Linear Models
=============================================

.. automodule:: nilearn.glm
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.

Classes
-------

.. currentmodule:: nilearn.glm

.. autosummary::
   :toctree: generated/
   :template: class.rst

    Contrast
    FContrastResults
    TContrastResults
    ARModel
    OLSModel
    LikelihoodModelResults
    RegressionResults
    SimpleRegressionResults

.. autoclasstree:: nilearn.glm
   :full:

Functions
---------

.. currentmodule:: nilearn.glm

.. autosummary::
   :toctree: generated/
   :template: function.rst

    compute_contrast
    compute_fixed_effects
    expression_to_contrast_vector
    fdr_threshold
    cluster_level_inference
    threshold_stats_img


:mod:`nilearn.glm.first_level`
------------------------------

.. automodule:: nilearn.glm.first_level
   :no-members:
   :no-inherited-members:

Classes
^^^^^^^

.. currentmodule:: nilearn.glm.first_level

.. autosummary::
   :toctree: generated/
   :template: class.rst

   FirstLevelModel

Functions
^^^^^^^^^

.. currentmodule:: nilearn.glm.first_level

.. autosummary::
   :toctree: generated/
   :template: function.rst

    check_design_matrix
    compute_regressor
    first_level_from_bids
    glover_dispersion_derivative
    glover_hrf
    glover_time_derivative
    make_first_level_design_matrix
    mean_scaling
    run_glm
    spm_dispersion_derivative
    spm_hrf
    spm_time_derivative

:mod:`nilearn.glm.second_level`
-------------------------------

.. automodule:: nilearn.glm.second_level
   :no-members:
   :no-inherited-members:

Classes
^^^^^^^

.. currentmodule:: nilearn.glm.second_level

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SecondLevelModel

Functions
^^^^^^^^^

.. currentmodule:: nilearn.glm.second_level

.. autosummary::
   :toctree: generated/
   :template: function.rst

    make_second_level_design_matrix
    non_parametric_inference


.. autoclasstree:: nilearn.glm.first_level.FirstLevelModel nilearn.glm.second_level.SecondLevelModel
   :full:
