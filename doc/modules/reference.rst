=========
Reference
=========

This is the class and function reference of nilearn. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.

.. contents:: List of modules
   :local:

.. _datasets_ref:

:mod:`nilearn.datasets`: Automatic Dataset Fetching
===================================================

.. automodule:: nilearn.datasets
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`datasets` section for further details.

Functions
---------
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
   load_harvard_oxford

.. _image_ref:

:mod:`nilearn.image`: Image processing and resampling utilities
===============================================================

.. automodule:: nilearn.image
   :no-members:
   :no-inherited-members:

Functions
---------
.. currentmodule:: nilearn.image

.. autosummary::
   :toctree: generated/
   :template: function.rst

   high_variance_confounds
   smooth
   resample_img


.. _io_ref:

:mod:`nilearn.input_data`: Loading and Processing files easily
==============================================================

.. automodule:: nilearn.input_data
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`nifti_masker` section for further details.

Classes
-------
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

Functions
---------
.. currentmodule:: nilearn.masking

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_epi_mask
   compute_multi_epi_mask
   intersect_masks
   apply_mask
   unmask

.. _region_ref:

:mod:`nilearn.region`: Regions Handling Utilities
=================================================

.. automodule:: nilearn.region
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`region` section for further details.

Functions
---------
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

.. _decoding_ref:

:mod:`nilearn.decoding`: Decoding
=================================

.. automodule:: nilearn.decoding
   :no-members:
   :no-inherited-members:

Classes
-------
.. currentmodule:: nilearn.decoding

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SearchLight

.. _signal_ref:

:mod:`nilearn.signal`: Preprocessing Time Series
================================================

.. automodule:: nilearn.signal
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.

Functions
---------
.. currentmodule:: nilearn.signal

.. autosummary::
   :toctree: generated/
   :template: function.rst

   clean
   high_variance_confounds
   butterworth

.. _utils_ref:

:mod:`nilearn._utils`: Manipulating Niimgs
==========================================

.. automodule:: nilearn._utils
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.

Functions
---------
.. currentmodule:: nilearn._utils

.. autosummary::
   :toctree: generated/
   :template: function.rst

   is_a_niimg
   check_niimg
   check_niimgs
   concat_niimgs
   copy_niimg

