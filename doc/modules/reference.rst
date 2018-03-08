================================================
Reference documentation: all nilearn functions
================================================

This is the class and function reference of nilearn. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.

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

   sym_to_vec
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
   fetch_atlas_msdl
   fetch_coords_power_2011
   fetch_atlas_smith_2009
   fetch_atlas_yeo_2011
   fetch_atlas_aal
   fetch_atlas_basc_multiscale_2015
   fetch_atlas_allen_2011
   fetch_coords_dosenbach_2010
   fetch_abide_pcp
   fetch_adhd
   fetch_haxby
   fetch_icbm152_2009
   fetch_icbm152_brain_gm_mask
   fetch_localizer_button_task
   fetch_localizer_contrasts
   fetch_localizer_calculation_task
   fetch_miyawaki2008
   fetch_nyu_rest
   fetch_surf_nki_enhanced
   fetch_surf_fsaverage5
   fetch_atlas_surf_destrieux
   fetch_atlas_talairach
   fetch_oasis_vbm
   fetch_megatrawls_netmats
   fetch_cobre
   fetch_neurovault
   fetch_neurovault_ids
   get_data_dirs
   load_mni152_template
   load_mni152_brain_mask

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

   SpaceNetClassifier
   SpaceNetRegressor
   SearchLight

.. _decomposition_ref:

:mod:`nilearn.decomposition`: Multivariate decompositions
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

:mod:`nilearn.image`: Image processing and resampling utilities
===============================================================

.. automodule:: nilearn.image
   :no-members:
   :no-inherited-members:

**Functions**:

.. currentmodule:: nilearn.image

.. autosummary::
   :toctree: generated/
   :template: function.rst

   clean_img
   concat_imgs
   coord_transform
   copy_img
   crop_img
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

.. _io_ref:

:mod:`nilearn.input_data`: Loading and Processing files easily
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
   compute_background_mask
   compute_multi_background_mask
   intersect_masks
   apply_mask
   unmask

:mod:`nilearn.regions`: Operating on regions
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


:mod:`nilearn.mass_univariate`: Mass-univariate analysis
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


:mod:`nilearn.plotting`: Plotting brain data
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
   plot_anat
   plot_img
   plot_epi
   plot_matrix
   plot_roi
   plot_stat_map
   plot_glass_brain
   plot_connectome
   plot_prob_atlas
   plot_surf
   plot_surf_roi
   plot_surf_stat_map
   show

**Classes**:

.. currentmodule:: nilearn.plotting.displays

.. autosummary::
   :toctree: generated/
   :template: class.rst

   OrthoSlicer


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

   clean
   high_variance_confounds


:mod:`nilearn.surface`: Manipulating surface data
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
