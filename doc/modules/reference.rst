================================================
Reference documentation: all nilearn functions
================================================

This is the class and function reference of nilearn. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.

.. contents:: **List of modules**
   :local:

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

   fetch_adhd
   fetch_craddock_2011_atlas
   fetch_haxby
   fetch_haxby_simple
   fetch_nyu_rest
   fetch_icbm152_2009
   fetch_msdl_atlas
   fetch_yeo_2011_atlas
   fetch_harvard_oxford

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
   
   SearchLight
   SpaceNet

.. _decomposition_ref:

:mod:`nilearn.decompositon`: Multivariate decompositions
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

   high_variance_confounds
   smooth_img
   resample_img
   reorder_img
   crop_img
   mean_img
   swap_img_hemispheres


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

.. _masking_ref:

:mod:`nilearn.masking`: Data Masking Utilities
==============================================

.. automodule:: nilearn.masking
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`masking` section for further details.

**Functions**:

.. currentmodule:: nilearn.region

.. autosummary::
   :toctree: generated/
   :template: function.rst

   img_to_signals_labels
   signals_to_img_labels
   img_to_signals_maps
   signals_to_img_maps

.. seealso::

   :func:`nilearn.masking.apply_mask`,
   :func:`nilearn.masking.unmask`


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

   plot_anat
   plot_img
   plot_epi
   plot_roi
   plot_stat_map
   find_xyz_cut_coords

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


