.. _first_level_model:

==================
First level models
==================

.. topic:: **Page summary**

  First level models are, in essence, linear regression models run at the level of a single
  session or single subject. The model is applied on a voxel-wise basis, either on the whole
  brain or within a region of interest. The timecourse of each voxel is regressed against a
  predicted BOLD response created by convolving the haemodynamic response function (HRF) with
  a set of predictors defined within the design matrix.


.. contents:: **Contents**
    :local:
    :depth: 1


HRF models
==========

Nilearn offers a few different HRF models including the commonly used double-gamma SPM model ('spm')
and the model shape proposed by G. Glover ('glover'), both allowing the option of adding time and
dispersion derivatives. The addition of these derivatives allows to better model any uncertainty in
timing information. In addition, an FIR (finite impulse response, 'fir') model of the HRF is also available.

In order to visualize the predicted regressor prior to plugging it into the linear model, use the
function :func:`nilearn.glm.first_level.compute_regressor`, or explore the HRF plotting
example :ref:`sphx_glr_auto_examples_04_glm_first_level_plot_hrf.py`.


Design matrix: event-based and time series-based
================================================

Event-based
-----------

To create an event-based design matrix, information about the trial type, onset time and duration of the
events in the experiment are necessary. This can be provided by the user, or be part of the dataset if
using a :term:`BIDS`-compatible dataset or one of the nilearn dataset fetcher functions like
:func:`nilearn.datasets.fetch_spm_multimodal_fmri`,
:func:`nilearn.datasets.fetch_language_localizer_demo_dataset`, etc.

Refer to the examples below for usage under the different scenarios:
  * User-defined: :ref:`sphx_glr_auto_examples_04_glm_first_level_plot_design_matrix.py`
  * Using an OpenNEURO dataset: :ref:`sphx_glr_auto_examples_04_glm_first_level_plot_bids_features.py`
  * Uing nilearn fetcher functions: :ref:`sphx_glr_auto_examples_04_glm_first_level_plot_spm_multimodal_faces.py`

To ascertain that the sequence of events provided to the first level model is accurate, Nilearn provides an
event visualization function called :func:`nilearn.plotting.plot_event()`. Sample usage for this is available
in :ref:`sphx_glr_auto_examples_02_decoding_plot_haxby_glm_decoding.py`.

Once the events are defined, the design matrix is created using the
:func:`nilearn.glm.first_level.make_first_level_design_matrix` function::

  from nilearn.glm.first_level import make_first_level_design_matrix
  design_matrices = make_first_level_design_matrix(frame_times, events,
                            drift_model='polynomial', drift_order=3)

.. note:: Additional predictors, like subject motion, can be specified using the add_reg parameter. Look at the function definition for available arguments.

A handy function called :func:`nilearn.plotting.plot_design_matrix()` can be used to visualize the design matrix.
This is generally a good practice to follow before proceeding with the analysis::

  from nilearn.plotting import plot_design_matrix
  plot_design_matrix(design_matrices)

.. image:: ../auto_examples/04_glm_first_level/images/sphx_glr_plot_design_matrix_001.png
   :target: ../auto_examples/04_glm_first_level/plot_design_matrix.html#sphx-glr-auto-examples-04-glm-first-level-models-plot-design-matrix-py


Time series-based
-----------------

The time series of a seed region can also be used as the predictor for a first level model. This approach would help
identify brain areas co-activating with the seed region. The time series is extracted using
:class:`nilearn.input_data.NiftiSpheresMasker`. For instance, if the seed region is the posterior
cingulate cortex with coordinate [pcc_coords]::

  from nilearn.input_data import NiftiSpheresMasker
  seed_masker = NiftiSpheresMasker([pcc_coords], radius=10)
  seed_time_series = seed_masker.fit_transform(adhd_dataset.func[0])

The seed_time_series is then passed into the design matrix using the add_reg argument mentioned in the note
above. Code for this approach is in :ref:`sphx_glr_auto_examples_04_glm_first_level_plot_adhd_dmn.py`.


Fitting a first level model
===========================

The :class:`nilearn.glm.first_level.FirstLevelModel` class provides the tools to fit the linear model to
the fMRI data. The :func:`nilearn.glm.first_level.FirstLevelModel.fit()` function takes the fMRI data
and design matrix as input and fits the GLM. Like other Nilearn functions,
:func:`nilearn.glm.first_level.FirstLevelModel.fit()` accepts file names as input, but can also
work with `NiftiImage objects <https://nipy.org/nibabel/nibabel_images.html>`_. More information about
input formats is available `here <http://nilearn.github.io/manipulating_images/input_output.html#inputing-data-file-names-or-image-objects>`_ ::

  from nilearn.glm.first_level import FirstLevelModel
  fmri_glm = FirstLevelModel()
  fmri_glm = fmri_glm.fit(subject_data, design_matrices=design_matrices)


Computing contrasts
-------------------

To get more interesting results out of the GLM model, contrasts can be computed between regressors of interest.
The :func:`nilearn.glm.first_level.FirstLevelModel.compute_contrast` function can be used for that. First,
the contrasts of interest must be defined. In the spm_multimodal_fmri dataset referenced above, subjects are
presented with 'normal' and 'scrambled' faces. The basic contrasts that can be constructed are the main effects
of 'normal faces' and 'scrambled faces'. Once the basic_contrasts have been set up, we can construct more
interesting contrasts like 'normal faces - scrambled faces'.

.. note:: The compute_contrast function can work with both numeric and symbolic arguments. See :func:`nilearn.glm.first_level.FirstLevelModel.compute_contrast` for more information.

And finally we can compute the contrasts using the compute_contrast function.
Refer to :ref:`sphx_glr_auto_examples_04_glm_first_level_plot_spm_multimodal_faces.py` for the full example.

The activation maps from these 3 contrasts is presented below:

.. image:: ../auto_examples/04_glm_first_level/images/sphx_glr_plot_spm_multimodal_faces_001.png
     :target: ../auto_examples/04_glm_first_level/plot_spm_multimodal_faces.html
     :scale: 60

.. image:: ../auto_examples/04_glm_first_level/images/sphx_glr_plot_spm_multimodal_faces_002.png
    :target: ../auto_examples/04_glm_first_level/plot_spm_multimodal_faces.html
    :scale: 60

.. image:: ../auto_examples/04_glm_first_level/images/sphx_glr_plot_spm_multimodal_faces_003.png
     :target: ../auto_examples/04_glm_first_level/plot_spm_multimodal_faces.html
     :scale: 60


Additional example: :ref:`sphx_glr_auto_examples_04_glm_first_level_plot_fiac_analysis.py`


Extracting predicted time series and residuals
==============================================

One way to assess the quality of the fit is to compare the observed and predicted time series of voxels.
Nilearn makes the predicted time series easily accessible via a parameter called `predicted` that is part
of the :class:`nilearn.glm.first_level.FirstLevelModel`. This parameter is populated when
FistLevelModel is initialized with the `minimize_memory` flag set to `False`. ::

  observed_timeseries = masker.fit_transform(fmri_img)
  predicted_timeseries = masker.fit_transform(fmri_glm.predicted[0])

Here, masker is an object of :class:`nilearn.input_data.NiftiSpheresMasker`. In the figure below,
predicted (red) and observed (not red) timecourses of 6 voxels are shown.

  .. image:: ../auto_examples/04_glm_first_level/images/sphx_glr_plot_predictions_residuals_002.png
     :target: ../auto_examples/04_glm_first_level/plot_predictions_residuals.html

In addition to the predicted timecourses, this flag also yields the residuals of the GLM. The residuals are
useful to calculate the F and R-squared statistic. For more information refer to
:ref:`sphx_glr_auto_examples_04_glm_first_level_plot_predictions_residuals.py`



Surface-based analysis
======================

fMRI analyses can also be performed on the cortical surface instead of a volumetric brain. Nilearn
provides functions to map subject brains on to a cortical mesh, which can be either a standard surface as
provided by, for e.g. Freesurfer, or a user-defined one. Freesurfer meshes can be accessed using
:func:`nilearn.datasets.fetch_surf_fsaverage`, while the function :func:`nilearn.surface.vol_to_surf`
does the projection from volumetric to surface space. Surface plotting functions like :func:`nilearn.plotting.plot_surf`
and :func:`nilearn.plotting.plot_surf_stat_map` allow for easy visualization of surface-based data.

For a complete example refer to :ref:`sphx_glr_auto_examples_04_glm_first_level_plot_localizer_surface_analysis.py`
