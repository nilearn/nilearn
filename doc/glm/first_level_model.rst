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

Nilearn offers a few different HRF models including the commonly used double-gamma SPM model
('spm') and the model shape proposed by G. Glover ('glover'), both allowing the option of adding
time and dispersion derivatives. The addition of these derivatives allows to better model any
uncertainty in timing information. In addition, an FIR (finite impulse response, 'fir') model
of the HRF is also available.

In order to visualize the predicted regressor prior to plugging it into the linear model, use
the function :func:`nilearn.stats.first_level_model.compute_regressor`, or explore the HRF plotting
example: :ref:`sphx_glr_auto_examples_04_glm_first_level_models_plot_hrf.py`


Design matrix: event-based and time series-based
================================================

Event-based
-----------

To create an event-based design matrix, information about the trial type, onset time and
duration of the events in the experiment are necessary. This can be provided by the user, or
be part of the dataset if using one of the nilearn dataset fetcher functions like
:func:`nilearn.datasets.fetch_spm_multimodal_fmri`,
:func:`nilearn.datasets.fetch_language_localizer_demo_dataset`, etc.
Using a nilearn fetcher function, e.g. :func:`nilearn.datasets.fetch_spm_multimodal_fmri`,
first download the data (for one subject)::

  from nilearn.datasets import fetch_spm_multimodal_fmri
  subject_data = fetch_spm_multimodal_fmri()

Here, each subject has 2 sessions of data; events for the first session can be accessed using::

  import pandas as pd
  events = pd.read_table(subject_data['events{}'.format(1)])

In addition, some scan attributes also need to be specified::

  import numpy as np
  tr = 1.0  # repetition time is 1 second
  n_scans = 128  # the acquisition comprises 128 scans
  frame_times = np.arange(n_scans) * tr  # here are the corresponding frame times

If the events details are provided by the user, then details about the experiment should
be provided. This information can be stored in a DataFrame and used instead of the
events file::

  import pandas as pd
  conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c3', 'c3', 'c3']
  duration = [1., 1., 1., 1., 1., 1., 1., 1., 1.]
  onsets = [30., 70., 100., 10., 30., 90., 30., 40., 60.]
  events = pd.DataFrame({'trial_type': conditions, 'onset': onsets, 'duration': duration})

In either case, the design matrix is created using the
:func:`nilearn.stats.first_level_model.make_first_level_design_matrix` function::

  from nilearn.stats.first_level_model import make_first_level_design_matrix
  design_matrices = make_first_level_design_matrix(frame_times, events,
                            drift_model='polynomial', drift_order=3)

A handy function called :func:`nilearn.reporting.plot_design_matrix()` can be used
to visualize the design matrix::

  from nilearn.reporting import plot_design_matrix
  plot_design_matrix(design_matrices)

.. image:: ../auto_examples/04_glm_first_level_models/images/sphx_glr_plot_design_matrix_001.png
   :target: ../auto_examples/04_glm_first_level_models/plot_design_matrix.html#sphx-glr-auto-examples-04-glm-first-level-models-plot-design-matrix-py

.. note:: Additional predictors, like subject motion, can be specified using the add_reg parameter. Look at the function definition for available arguments.


Time series-based
-----------------

The time series of a seed region can also be used as the predictor for a first level
model. This would be used to identify brain areas co-activating with the seed
region. The time series is extracted using the NiftiSpheresMasker
function. For instance, if the seed region is the posterior cingulate cortex::

  from nilearn.input_data import NiftiSpheresMasker
  seed_masker = NiftiSpheresMasker([pcc_coords], radius=10, detrend=True,
                                 standardize=True, low_pass=0.1,
                                 high_pass=0.01, t_r=2.,
                                 memory='nilearn_cache',
                                 memory_level=1, verbose=0)
  seed_time_series = seed_masker.fit_transform(adhd_dataset.func[0])

The seed_time_series is then passed into the design matrix using the same add_reg
argument used above for motion parameters::

  from nilearn.stats.first_level_model import make_first_level_design_matrix
  design_matrices = make_first_level_design_matrix(frametimes,
                                               add_regs=seed_time_series,
                                               add_reg_names=["pcc_seed"])



Fitting a first level model
===========================

The :class:`nilearn.stats.first_level_model.FirstLevelModel` class provides the tools
to fit the linear model to the fMRI data. The :func:`nilearn.stats.first_level_model.FirstLevelModel.fit()` function
takes the fMRI data and design matrix as input and fits the GLM. Like other Nilearn
functions, :func:`nilearn.stats.first_level_model.FirstLevelModel.fit()` accepts file names as input, but can also
work with NiftiImage objects ref:`https://nipy.org/nibabel/nibabel_images.html`.
More information about input formats is available here:
ref:`http://nilearn.github.io/manipulating_images/input_output.html#inputing-data-file-names-or-image-objects` ::

  from nilearn.stats.first_level_model import FirstLevelModel
  fmri_glm = FirstLevelModel()
  fmri_glm = fmri_glm.fit(subject_data, design_matrices=design_matrices)


Computing contrasts
-------------------

To get more interesting results out of the GLM model, contrasts can be computed
between regressors of interest. The :func:`nilearn.stats.first_level_model.FirstLevelModel.compute_contrast` can be
used for that. First, the contrasts of interest must be defined. In the spm_multimodal_fmri
dataset referenced above, subjects are presented with normal and scrambled faces. The basic
contrasts that can be constructed are::

  contrast_matrix = np.eye(design_matrix.shape[1])
  basic_contrasts = dict([(column, contrast_matrix[i])
                for i, column in enumerate(design_matrix.columns)])

Using basic_contrasts, we can construct more interesting contrasts::

  contrasts = {
    'faces-scrambled': basic_contrasts['faces'] - basic_contrasts['scrambled'],
    'scrambled-faces': -basic_contrasts['faces'] + basic_contrasts['scrambled'],
    'effects_of_interest': np.vstack((basic_contrasts['faces'],
                                      basic_contrasts['scrambled']))
  }

And compute the contrasts as follows::

  for contrast_id, contrast_val in contrasts.items():
    z_map = fmri_glm.compute_contrast(
        contrast_val, output_type='z_score')

.. image:: ../auto_examples/04_glm_first_level_models/images/sphx_glr_plot_spm_multimodal_faces_001.png
     :target: ../auto_examples/04_glm_first_level_models/plot_spm_multimodal_faces.html#sphx-glr-auto-examples-04-glm-first-level-models-plot-spm-multimodal-faces-py
     :scale: 60

.. image:: ../auto_examples/04_glm_first_level_models/images/sphx_glr_plot_spm_multimodal_faces_002.png
    :target: ../auto_examples/04_glm_first_level_models/plot_spm_multimodal_faces.html#sphx-glr-auto-examples-04-glm-first-level-models-plot-spm-multimodal-faces-py
    :scale: 60

.. image:: ../auto_examples/04_glm_first_level_models/images/sphx_glr_plot_spm_multimodal_faces_003.png
     :target: ../auto_examples/04_glm_first_level_models/plot_spm_multimodal_faces.html#sphx-glr-auto-examples-04-glm-first-level-models-plot-spm-multimodal-faces-py
     :scale: 60


For full examples on fitting a first level model, look at the following examples:

ref:`sphx-glr-auto-examples-04-glm-first-level-models-plot-spm-multimodal-faces.py`

ref:`sphx-glr-auto-examples-04-glm-first-level-models-plot-fiac-analysis.py`



Extracting predicted time series and residuals
==============================================

One way to assess the quality of the fit is to compare the observed and predicted time series of voxels.
Nilearn makes the predicted time series easily accessible via a parameter called
`predicted` that is part of the :class:`nilearn.stats.first_level_model.FirstLevelModel`. This parameter
is populated the when FistLevelModel is initialized with the `minimize_memory` flag set to `False`. ::

  observed_timeseries = masker.fit_transform(fmri_img)
  predicted_timeseries = masker.fit_transform(fmri_glm.predicted[0])

Here, masker is an object of :class:`nilearn.input_data.NiftiSpheresMasker`. In the figure below,
predicted (red) and observed (not red) timecourses of 6 voxels are shown.

  .. image:: ../auto_examples/04_glm_first_level_models/images/sphx_glr_plot_predictions_residuals_002.png
     :target: ../auto_examples/04_glm_first_level_models/plot_predictions_residuals.html#sphx-glr-auto-examples-04-glm-first-level-models-plot-predictions-residuals-py

In addition to the predicted timecourses, this flag also yields the residuals of the GLM.
The residuals are useful to calculate the F and R-squared statistic. For more information
refer to ref:`../auto_examples/04_glm_first_level_models/plot_predictions_residuals.html#sphx-glr-auto-examples-04-glm-first-level-models-plot-predictions-residuals-py`



Surface-based analysis
======================

fMRI analyses are also performed on the cortical surface instead of a volumetric brain.
Nilearn provides functions to map subject brains on to a cortical mesh, either a standard
surface as provided by Freesurfer, for e.g., or a user-defined one. Freesurfer meshes can
be accessed using :func:`nilearn.datasets.fetch_surf_fsaverage`, while the function
:func:`nilearn.surface.vol_to_surf` does the projection from volumetric to surface space.
Surface plotting functions like :func:`nilearn.plotting.plot_surf` and
:func:`nilearn.plotting.plot_surf_stat_map` allow for easy visualization of surface-based
data.

For a complete example refer to ref:`../auto_examples/04_glm_first_level_models/plot_localizer_surface_analysis.html`
