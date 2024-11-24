.. _datasets_ref:

:mod:`nilearn.datasets`: Automatic Dataset Fetching
===================================================

.. automodule:: nilearn.datasets
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`datasets` section for further details.

Templates
---------

Functions
^^^^^^^^^

.. currentmodule:: nilearn.datasets

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fetch_icbm152_2009
   fetch_icbm152_brain_gm_mask
   fetch_surf_fsaverage
   load_fsaverage
   load_fsaverage_data
   load_mni152_brain_mask
   load_mni152_gm_mask
   load_mni152_gm_template
   load_mni152_template
   load_mni152_wm_mask
   load_mni152_wm_template

Templates descriptions
^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :titlesonly:

    description/icbm152_2009.rst
    description/fsaverage.rst
    description/fsaverage3.rst
    description/fsaverage4.rst
    description/fsaverage5.rst
    description/fsaverage6.rst

Atlases
-------

Deterministic atlases
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: nilearn.datasets

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fetch_atlas_aal
   fetch_atlas_basc_multiscale_2015
   fetch_atlas_destrieux_2009
   fetch_atlas_harvard_oxford
   fetch_atlas_juelich
   fetch_atlas_pauli_2017
   fetch_atlas_schaefer_2018
   fetch_atlas_surf_destrieux
   fetch_atlas_talairach
   fetch_atlas_yeo_2011
   fetch_coords_dosenbach_2010
   fetch_coords_power_2011
   fetch_coords_seitzman_2018

Probabilistic atlases
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: nilearn.datasets

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fetch_atlas_allen_2011
   fetch_atlas_craddock_2012
   fetch_atlas_difumo
   fetch_atlas_harvard_oxford
   fetch_atlas_juelich
   fetch_atlas_msdl
   fetch_atlas_pauli_2017
   fetch_atlas_smith_2009

Atlases descriptions
^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :titlesonly:

    description/aal.rst
    description/allen_rsn_2011.rst
    description/basc_multiscale_2015.rst
    description/craddock_2012.rst
    description/destrieux_surface.rst
    description/difumo_atlases.rst
    description/harvard_oxford.rst
    description/juelich.rst
    description/msdl_atlas.rst
    description/pauli_2017.rst
    description/schaefer_2018.rst
    description/smith_2009.rst
    description/talairach_atlas.rst
    description/yeo_2011.rst
    description/dosenbach_2010.rst
    description/power_2011.rst
    description/seitzman_2018.rst

Preprocessed datasets
---------------------

Functions
^^^^^^^^^

.. currentmodule:: nilearn.datasets

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fetch_abide_pcp
   fetch_adhd
   fetch_bids_langloc_dataset
   fetch_development_fmri
   fetch_ds000030_urls
   fetch_fiac_first_level
   fetch_haxby
   fetch_language_localizer_demo_dataset
   fetch_localizer_first_level
   fetch_miyawaki2008
   fetch_spm_auditory
   fetch_spm_multimodal_fmri
   fetch_surf_nki_enhanced
   load_nki

Datasets descriptions
^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :titlesonly:

    description/ABIDE_pcp.rst
    description/adhd.rst
    description/bids_langloc.rst
    description/development_fmri.rst
    description/fiac.rst
    description/haxby2001.rst
    description/language_localizer_demo.rst
    description/localizer_first_level.rst
    description/miyawaki2008.rst
    description/spm_auditory.rst
    description/spm_multimodal.rst
    description/nki_enhanced_surface.rst
    description/brainomics_localizer.rst

Statistical maps/derivatives
----------------------------

Functions
^^^^^^^^^

.. currentmodule:: nilearn.datasets

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fetch_localizer_button_task
   fetch_localizer_calculation_task
   fetch_localizer_contrasts
   fetch_megatrawls_netmats
   fetch_mixed_gambles
   fetch_oasis_vbm
   fetch_neurovault_auditory_computation_task
   fetch_neurovault_motor_task

Statistical maps/derivatives descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :titlesonly:

    description/Megatrawls.rst
    description/mixed_gambles.rst
    description/oasis1.rst

General functions
-----------------

Functions
^^^^^^^^^

.. currentmodule:: nilearn.datasets

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fetch_neurovault
   fetch_neurovault_ids
   fetch_openneuro_dataset
   get_data_dirs
   patch_openneuro_dataset
   select_from_index
   load_sample_motor_activation_image

General functions descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :titlesonly:

    description/neurovault.rst
