.. _second_level_model:

===================
Second level models
===================

.. topic:: **Page summary**

   Second level models in Nilearn are used to perform group-level analyses on fMRI data. Once individual
   subjects have been processed in a common space (e.g. MNI, Talairach, or subject average), the data can
   be grouped and statistical tests  performed to make broader inferences on fMRI activity. Some common
   second level models are one-sample (unpaired or paired) and two-sample t-tests.

.. contents:: **Contents**
    :local:
    :depth: 1


Fitting a second level model
============================

As with first level models, a design matrix needs to be defined before fitting a second level model.
Again, similar to first level models, Nilearn provides a function
:func:`nilearn.glm.second_level.make_second_level_design_matrix` for this purpose. Once
the design matrix has been setup, it can be visualized using the same function as before,
:func:`nilearn.plotting.plot_design_matrix`.

To fit the second level model, the tools to use are within the class
:class:`nilearn.glm.second_level.SecondLevelModel`. Specifically, the function that
fits the model is :func:`nilearn.glm.second_level.SecondLevelModel.fit`.

Some examples to get you going with second level models are provided below::
  * General design matrix setup: :ref:`sphx_glr_auto_examples_05_glm_second_level_plot_second_level_design_matrix.py`
  * One-sample testing: :ref:`sphx_glr_auto_examples_05_glm_second_level_plot_second_level_one_sample_test.py`
  * Two-sample testing, unpaired and paired: :ref:`sphx_glr_auto_examples_05_glm_second_level_plot_second_level_two_sample_test.py`
  * Complex contrast: :ref:`sphx_glr_auto_examples_05_glm_second_level_plot_second_level_association_test.py`


Thresholding statistical maps
=============================

Nilearn's statistical plotting functions provide simple thresholding functionality. For instance, functions
like :func:`nilearn.plotting.plot_stat_map` or :func:`nilearn.plotting.plot_glass_brain` have an argument
called `threshold` that, when set, will only show voxels with a value that is over the threshold provided.

Thresholding examples are available here: :ref:`sphx_glr_auto_examples_05_glm_second_level_plot_second_level_one_sample_test.py`
and :ref:`sphx_glr_auto_examples_05_glm_second_level_plot_thresholding.py`.


Multiple comparisons correction
===============================

As discussed in the :ref:`Multiple comparisons` section of the introduction, the issue of multiple comparisons is
important to address with statistical analysis of fMRI data. Nilearn provides parametric and non-parametric tools
to address this issue.

Refer to the example :ref:`sphx_glr_auto_examples_05_glm_second_level_plot_thresholding.py` for a guide
to applying FPR, FDR and FWER corrections. These corrections are applied using the :func:`nilearn.glm.threshold_stats_img` function.

Within an activated cluster, not all voxels represent true activation. To estimate true positives within a cluster,
Nilearn provides the :func:`nilearn.glm.cluster_level_inference` function. An example with usage information is available
here: :ref:`sphx_glr_auto_examples_05_glm_second_level_plot_proportion_activated_voxels.py`.


Voxel based morphometry
=======================

The :class:`nilearn.glm.second_level.SecondLevelModel` and its associated functions can also be used
to perform voxel based morphometry. An example using the `OASIS <http://www.oasis-brains.org/>`_ dataset to
identify the relationship between aging, sex and gray matter density is available here
:ref:`sphx_glr_auto_examples_05_glm_second_level_plot_oasis.py`.
