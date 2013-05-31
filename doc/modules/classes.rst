=========
Reference
=========

This is the class and function reference of nisl. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.

.. contents:: List of modules
   :local:

.. _datasets_ref:

:mod:`nisl.datasets`: Automatic Dataset Fetching
================================================

.. automodule:: nisl.datasets
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`datasets` section for further details.

Functions
---------
.. currentmodule:: nisl.datasets

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

:mod:`nisl.image`: Image processing utilities
=============================================

.. automodule:: nisl.image
   :no-members:
   :no-inherited-members:

Functions
---------
.. currentmodule:: nisl.image

.. autosummary::
   :toctree: generated/
   :template: function.rst

   high_variance_confounds
   smooth

.. _io_ref:

:mod:`nisl.io`: Loading and Processing files easily
======================================================

.. automodule:: nisl.io
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`nifti_masker` section for further details.

Classes
-------
.. currentmodule:: nisl.io

.. autosummary::
   :toctree: generated/
   :template: class.rst

   NiftiMasker
   NiftiMultiMasker
   NiftiLabelsMasker
   NiftiMapsMasker

.. _masking_ref:

:mod:`nisl.masking`: Data Masking Utilities
===========================================

.. automodule:: nisl.masking
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`masking` section for further details.

Functions
---------
.. currentmodule:: nisl.masking

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_epi_mask
   compute_multi_epi_mask
   intersect_masks
   apply_mask
   unmask

.. _region_ref:

:mod:`nisl.region`: Regions Handling Utilities
==============================================

.. automodule:: nisl.region
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`region` section for further details.

Functions
---------
.. currentmodule:: nisl.region

.. autosummary::
   :toctree: generated/
   :template: function.rst

   img_to_signals_labels
   signals_to_img_labels
   img_to_signals_maps
   signals_to_img_maps

.. seealso::

   :func:`nisl.masking.apply_mask`,
   :func:`nisl.masking.unmask`

.. _resampling_ref:

:mod:`nisl.resampling`: Data Resampling Utilities
=================================================

.. automodule:: nisl.resampling
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`resampling` section for further details.

Functions
---------
.. currentmodule:: nisl.resampling

.. autosummary::
   :toctree: generated/
   :template: function.rst

   to_matrix_vector
   from_matrix_vector
   get_bounds
   resample_img

.. _decoding_ref:

:mod:`nisl.decoding`: Decoding
==============================

.. automodule:: nisl.decoding
   :no-members:
   :no-inherited-members:

Classes
-------
.. currentmodule:: nisl.decoding

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SearchLight

.. _signal_ref:

:mod:`nisl.signal`: Preprocessing Time Series
==============================================

.. automodule:: nisl.signal
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.

Functions
---------
.. currentmodule:: nisl.signal

.. autosummary::
   :toctree: generated/
   :template: function.rst

   clean
   high_variance_confounds
   butterworth

.. _utils_ref:

:mod:`nisl.utils`: Manipulating Niimgs
======================================

.. automodule:: nisl.utils
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.

Functions
---------
.. currentmodule:: nisl.utils

.. autosummary::
   :toctree: generated/
   :template: function.rst

   is_a_niimg
   check_niimg
   check_niimgs
   concat_niimgs
   copy_niimg

