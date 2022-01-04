===============================================
 Reference documentation: all nilearn functions
===============================================

This is the class and function reference of nilearn. Please refer to
the :ref:`user guide <user_guide>` for more information and usage examples.

.. contents:: **List of modules**
   :local:


.. _connectome_ref:

:mod:`nilearn.connectome`: Functional Connectivity
====================================================

.. automodule:: nilearn.connectome
   :no-members:
   :no-inherited-members:

**Classes**:

.. currentmodule:: nilearn.connectome

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ConnectivityMeasure
   GroupSparseCovariance
   GroupSparseCovarianceCV

**Functions**:

.. currentmodule:: nilearn.connectome

.. autosummary::
   :toctree: generated/
   :template: function.rst

   sym_matrix_to_vec
   vec_to_sym_matrix
   group_sparse_covariance
   cov_to_corr
   prec_to_partial

.. _datasets_ref:

:mod:`nilearn.datasets`: Automatic Dataset Fetching
===================================================

.. automodule:: nilearn.datasets
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`datasets` section for further details.

**Functions**:

.. currentmodule:: nilearn.datasets

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fetch_atlas_craddock_2012
   fetch_atlas_destrieux_2009
   fetch_atlas_harvard_oxford
   fetch_atlas_juelich
   fetch_atlas_msdl
   fetch_atlas_difumo
   fetch_coords_power_2011
   fetch_coords_seitzman_2018
   fetch_atlas_smith_2009
   fetch_atlas_yeo_2011
   fetch_atlas_aal
   fetch_atlas_basc_multiscale_2015
   fetch_atlas_allen_2011
   fetch_atlas_pauli_2017
   fetch_coords_dosenbach_2010
   fetch_abide_pcp
   fetch_adhd
   fetch_development_fmri
   fetch_haxby
   fetch_icbm152_2009
   fetch_icbm152_brain_gm_mask
   fetch_localizer_button_task
   fetch_localizer_contrasts
   fetch_localizer_calculation_task
   fetch_miyawaki2008
   fetch_surf_nki_enhanced
   fetch_surf_fsaverage
   fetch_atlas_surf_destrieux
   fetch_atlas_talairach
   fetch_atlas_schaefer_2018
   fetch_oasis_vbm
   fetch_megatrawls_netmats
   fetch_neurovault
   fetch_neurovault_ids
   fetch_neurovault_auditory_computation_task
   fetch_neurovault_motor_task
   get_data_dirs
   load_mni152_template
   load_mni152_gm_template
   load_mni152_wm_template
   load_mni152_brain_mask
   load_mni152_gm_mask
   load_mni152_wm_mask
   fetch_language_localizer_demo_dataset
   fetch_bids_langloc_dataset
   fetch_openneuro_dataset_index
   select_from_index
   patch_openneuro_dataset
   fetch_openneuro_dataset
   fetch_localizer_first_level
   fetch_spm_auditory
   fetch_spm_multimodal_fmri
   fetch_fiac_first_level

.. _decoding_ref:

:mod:`nilearn.decoding`: Decoding
=================================

.. automodule:: nilearn.decoding
   :no-members:
   :no-inherited-members:

**Classes**:

.. currentmodule:: nilearn.decoding

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Decoder
   DecoderRegressor
   FREMClassifier
   FREMRegressor
   SpaceNetClassifier
   SpaceNetRegressor
   SearchLight

.. _decomposition_ref:

:mod:`nilearn.decomposition`: Multivariate Decompositions
=========================================================

.. automodule:: nilearn.decomposition
   :no-members:
   :no-inherited-members:

**Classes**:

.. currentmodule:: nilearn.decomposition

.. autosummary::
   :toctree: generated/
   :template: class.rst

   CanICA
   DictLearning

.. _image_ref:

:mod:`nilearn.image`: Image Processing and Resampling Utilities
===============================================================

.. automodule:: nilearn.image
   :no-members:
   :no-inherited-members:

**Functions**:

.. currentmodule:: nilearn.image

.. autosummary::
   :toctree: generated/
   :template: function.rst

   binarize_img
   clean_img
   concat_imgs
   coord_transform
   copy_img
   crop_img
   get_data
   high_variance_confounds
   index_img
   iter_img
   largest_connected_component_img
   load_img
   math_img
   mean_img
   new_img_like
   resample_img
   resample_to_img
   reorder_img
   smooth_img
   swap_img_hemispheres
   threshold_img

.. _interfaces_ref:

:mod:`nilearn.interfaces`: Loading components from interfaces
=============================================================

.. automodule:: nilearn.interfaces
   :no-members:
   :no-inherited-members:

:mod:`nilearn.interfaces.fmriprep`
----------------------------------

.. automodule:: nilearn.interfaces.fmriprep
   :no-members:
   :no-inherited-members:

**Functions**:

.. currentmodule:: nilearn.interfaces.fmriprep

.. autosummary::
   :toctree: generated/
   :template: function.rst

   load_confounds
   load_confounds_strategy

.. _io_ref:

:mod:`nilearn.input_data`: Loading and Processing Files Easily
==============================================================

.. automodule:: nilearn.input_data
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`nifti_masker` section for further details.

**Classes**:

.. currentmodule:: nilearn.input_data

.. autosummary::
   :toctree: generated/
   :template: class.rst

   NiftiMasker
   MultiNiftiMasker
   NiftiLabelsMasker
   NiftiMapsMasker
   NiftiSpheresMasker

.. _masking_ref:

:mod:`nilearn.masking`: Data Masking Utilities
==============================================

.. automodule:: nilearn.masking
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`masking` section for further details.

**Functions**:

.. currentmodule:: nilearn.masking

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_epi_mask
   compute_multi_epi_mask
   compute_brain_mask
   compute_multi_brain_mask
   compute_background_mask
   compute_multi_background_mask
   intersect_masks
   apply_mask
   unmask

:mod:`nilearn.regions`: Operating on Regions
============================================

.. automodule:: nilearn.regions
   :no-members:
   :no-inherited-members:

**Functions**:

.. currentmodule:: nilearn.regions

.. autosummary::
   :toctree: generated/
   :template: function.rst

   connected_regions
   connected_label_regions
   img_to_signals_labels
   signals_to_img_labels
   img_to_signals_maps
   signals_to_img_maps

**Classes**:

.. currentmodule:: nilearn.regions

.. autosummary::
   :toctree: generated/
   :template: class.rst

   RegionExtractor
   Parcellations
   ReNA


:mod:`nilearn.mass_univariate`: Mass-Univariate Analysis
=========================================================

.. automodule:: nilearn.mass_univariate
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.

**Functions**:

.. currentmodule:: nilearn.mass_univariate

.. autosummary::
   :toctree: generated/
   :template: function.rst

   permuted_ols

.. _plotting_ref:


:mod:`nilearn.plotting`: Plotting Brain Data
================================================

.. automodule:: nilearn.plotting
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.

**Functions**:

.. currentmodule:: nilearn.plotting

.. autosummary::
   :toctree: generated/
   :template: function.rst

   find_cut_slices
   find_xyz_cut_coords
   find_parcellation_cut_coords
   find_probabilistic_atlas_cut_coords
   plot_anat
   plot_img
   plot_epi
   plot_matrix
   plot_roi
   plot_stat_map
   plot_glass_brain
   plot_connectome
   plot_markers
   plot_prob_atlas
   plot_carpet
   plot_surf
   plot_surf_roi
   plot_surf_contours
   plot_surf_stat_map
   plot_img_on_surf
   plot_img_comparison
   plot_design_matrix
   plot_event
   plot_contrast_matrix
   view_surf
   view_img_on_surf
   view_connectome
   view_markers
   view_img
   show

:mod:`nilearn.plotting.displays`: Interacting with figures
----------------------------------------------------------

.. automodule:: nilearn.plotting.displays
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.

**Functions**:

.. currentmodule:: nilearn.plotting.displays

.. autosummary::
   :toctree: generated/
   :template: function.rst

    get_projector
    get_slicer

**Classes**:

.. currentmodule:: nilearn.plotting.displays

.. autosummary::
   :toctree: generated/
   :template: class.rst

   OrthoProjector
   XZProjector
   YZProjector
   YXProjector
   XProjector
   YProjector
   ZProjector
   LZRYProjector
   LYRZProjector
   LYRProjector
   LZRProjector
   LRProjector
   LProjector
   RProjector
   BaseAxes
   CutAxes
   GlassBrainAxes
   BaseSlicer
   OrthoSlicer
   PlotlySurfaceFigure
   TiledSlicer
   MosaicSlicer
   XZSlicer
   YZSlicer
   YXSlicer
   XSlicer
   YSlicer
   ZSlicer



.. _signal_ref:


:mod:`nilearn.signal`: Preprocessing Time Series
================================================

.. automodule:: nilearn.signal
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.

**Functions**:

.. currentmodule:: nilearn.signal

.. autosummary::
   :toctree: generated/
   :template: function.rst

   butterworth
   clean
   high_variance_confounds

.. _stats_ref:


:mod:`nilearn.glm`: Generalized Linear Models
================================================

.. automodule:: nilearn.glm
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.

**Classes**:

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

**Functions**:

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
--------------------------------------

.. automodule:: nilearn.glm.first_level
   :no-members:
   :no-inherited-members:

**Classes**:

.. currentmodule:: nilearn.glm.first_level

.. autosummary::
   :toctree: generated/
   :template: class.rst

   FirstLevelModel

**Functions**:

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
---------------------------------------

.. automodule:: nilearn.glm.second_level
   :no-members:
   :no-inherited-members:

**Classes**:

.. currentmodule:: nilearn.glm.second_level

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SecondLevelModel

**Functions**:

.. currentmodule:: nilearn.glm.second_level

.. autosummary::
   :toctree: generated/
   :template: function.rst

    make_second_level_design_matrix
    non_parametric_inference

.. _reporting_ref:


:mod:`nilearn.reporting`: Reporting Functions
=============================================

.. automodule:: nilearn.reporting
   :no-members:
   :no-inherited-members:

**Functions**:

.. currentmodule:: nilearn.reporting

.. autosummary::
   :toctree: generated/
   :template: function.rst

   get_clusters_table
   make_glm_report


:mod:`nilearn.surface`: Manipulating Surface Data
===================================================

.. automodule:: nilearn.surface
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.

**Functions**:

.. currentmodule:: nilearn.surface

.. autosummary::
   :toctree: generated/
   :template: function.rst

   load_surf_data
   load_surf_mesh
   vol_to_surf
