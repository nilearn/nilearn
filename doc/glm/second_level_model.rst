.. _second_level_model:

===================
Second level models
===================

.. topic:: **Page summary**

   Second level models are used to take fMRI analysis to the group level. Once individual subject data has been processed in a common space (e.g. MNI, Talairach, or subject average), the data can be grouped and statistical tests  performed to make broader inferences on fMRI activity.

.. contents:: **Contents**
    :local:
    :depth: 1


Design matrix
=============

As with first level models, a design matrix needs to be defined before fitting a second level model. Some common second level models are one-sample and two-sample t-tests. A quick look into setting up design matrices for these models is provided here.


One-sample testing
------------------

In one-sample testing, activity associated with a predictor variable is tested against some form of baseline activity. Therefore, predictors must be defined in order to set up a one-sample t-test. For example, consider the first level contrast of left-vs-right button press, which can be downloaded using :func:`nilearn.datasets.fetch_localizer_contrasts`::

  from nilearn.datasets import fetch_localizer_contrasts
  n_subjects = 16
  data = fetch_localizer_contrasts(["left vs right button press"], n_subjects,
                               get_tmaps=True)

A first level model has been fit to this data and the contrasts maps for left-vs-right button press activation are available in the `cmaps` parameter. The second level model is fit using these contrast maps thus::

  from nilearn.stats.second_level_model import make_second_level_design_matrix
  second_level_input = data['cmaps']
  design_matrix = pd.DataFrame([1] * len(second_level_input), columns=['intercept'])


Two-sample testing
------------------

In two-sample testing, the activity attributed to two predictors is contrasted against each other, and so, two predictors need to be defined to fit the model. Consider the dataset that has activation corresponding to the presentation of horizonal and vertical checkerboards. This dataset can also be downloaded using :func:`nilearn.datasets.fetch_localizer_contrasts`::

  from nilearn.datasets import fetch_localizer_contrasts
  n_subject = 16
  sample_vertical = fetch_localizer_contrasts(
      ["vertical checkerboard"], n_subjects, get_tmaps=True)
  sample_horizontal = fetch_localizer_contrasts(
      ["horizontal checkerboard"], n_subjects, get_tmaps=True)
  second_level_input = sample_vertical['cmaps'] + sample_horizontal['cmaps']

Set up the design matrix::

  import numpy as np
  condition_effect = np.hstack(([1] * n_subjects, [- 1] * n_subjects))
  subject_effect = np.vstack((np.eye(n_subjects), np.eye(n_subjects)))
  subjects = ['S%02d' % i for i in range(1, n_subjects + 1)]
  design_matrix = pd.DataFrame(
      np.hstack((condition_effect[:, np.newaxis], subject_effect)),
      columns=['vertical vs horizontal'] + subjects)

As with first level models, the design matrix can be visualized using :func:`nilearn.reporting.plot_design_matrix()`

.. image:: ../auto_examples/05_glm_second_level_models/images/sphx_glr_plot_second_level_two_sample_test_001.png
   :target: ../auto_examples/05_glm_second_level_models/plot_second_level_two_sample_test.html


Fitting a second level model
============================

Once the design matrix has been set up, fitting the model is simple. The class that provides tools for this is :class:`nilearn.stats.second_level_model.SecondLevelModel`. The :func:`nilearn.stats.second_level_model.SecondLevelModel.fit` function fits the model::

  from nilearn.stats.second_level_model import SecondLevelModel
  second_level_model = SecondLevelModel().fit(
      second_level_input, design_matrix=design_matrix)


While contrast maps can be computed using the :func:`nilearn.stats.second_level_model.SecondLevelModel.compute_contrast` function. For instance, the the following  contrast can be obtained from the two-sample model described above::

  z_map = second_level_model.compute_contrast('vertical vs horizontal', output_type='z_score')


For full examples refer to: :ref:`sphx_glr_auto_examples_05_glm_second_level_models_plot_second_level_one_sample_test.py`, :ref:`sphx_glr_auto_examples_05_glm_second_level_models_plot_second_level_two_sample_test.py` and :ref:`sphx_glr_auto_examples_05_glm_second_level_models_plot_second_level_association_test.py`


Thresholding statistical maps
=============================

Nilearn's statistical plotting functions provide simple thresholding functionality. For instance, functions like :func:`nilearn.plotting.plot_glass_brain` or :func:`nilearn.plotting.plot_glass_brain` have an argument called `threshold` that only show voxels with a value that is over the threshold provided. Thresholding examples are available here: :ref:`sphx_glr_auto_examples_05_glm_second_level_models_plot_second_level_one_sample_test.py` and :ref:`sphx_glr_auto_examples_05_glm_second_level_models_plot_thresholding.py`.


Multiple comparisons correction
===============================

As discussed in the :ref:`Multiple comparisons` section, the issue of multiple comparisons is important to address with statistical analysis of fMRI data. Nilearn provides parametric and non-parametric tools to address this issue.

Refer to the example :ref:`sphx_glr_auto_examples_05_glm_second_level_models_plot_thresholding.py` for a guide to applying FPR, FDR and FWER corrections. These corrections are applied using the :func:`nilearn.stats.map_threshold` function.

Within an activated cluster, not all voxels reprepsent true activation. To estimate true positives within a cluster, Nilearn provides the :func:`nilearn.stats.cluster_level_inference` function. An example with usage information is available here: :ref:`sphx_glr_auto_examples_05_glm_second_level_models_plot_proportion_activated_voxels.py`


Voxel based morphometry
=======================

The :class:`nilearn.stats.second_level_model.SecondLevelModel` and its associated functions can also be used to perform voxel based morphometry. An example using the `OASIS <http://www.oasis-brains.org/>`_ dataset to identify the relationship between aging, sex and gray matter density is available ref:here <auto_examples/05_glm_second_level_models/plot_oasis.html#sphx-glr-auto-examples-05-glm-second-level-models-plot-oasis-py>.
