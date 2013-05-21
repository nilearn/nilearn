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
.. currentmodule:: nisl

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.fetch_haxby
   datasets.fetch_haxby_simple
   datasets.fetch_nyu_rest
   datasets.fetch_adhd

:mod:`nisl.image`: Image processing utilities
=============================================

.. automodule:: nisl.image
   :no-members:
   :no-inherited-members:

Functions
---------
.. currentmodule:: nisl

.. autosummary::
   :toctree: generated/
   :template: function.rst

   image.high_variance_confounds
   image.smooth

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

   nifti_masker.NiftiMasker
   nifti_multi_masker.NiftiMultiMasker
   nifti_region.NiftiLabelsMasker

.. _masking_ref:

:mod:`nisl.masking`: Data Masking Utilities
===========================================

.. automodule:: nisl.masking
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`masking` section for further details.

Functions
---------
.. currentmodule:: nisl

.. autosummary::
   :toctree: generated/
   :template: function.rst

   masking.compute_epi_mask
   masking.compute_multi_epi_mask
   masking.intersect_masks
   masking.apply_mask
   masking.unmask

.. _region_ref:

:mod:`nisl.region`: Regions Handling Utilities
==============================================

.. automodule:: nisl.region
   :no-members:
   :no-inherited-members:

Functions
---------
.. currentmodule:: nisl

.. autosummary::
   :toctree: generated/
   :template: function.rst

   region.img_to_signals_labels
   region.signals_to_img_labels
   region.img_to_signals_maps
   region.signals_to_img_maps

.. seealso::
   
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
.. currentmodule:: nisl

.. autosummary::
   :toctree: generated/
   :template: function.rst

   resampling.to_matrix_vector
   resampling.from_matrix_vector
   resampling.get_bounds
   resampling.resample_img

.. _signal_ref:

:mod:`nisl.signal`: Preprocessing Time Series
==============================================

.. automodule:: nisl.signal
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`signal` section for further details.

Functions
---------
.. currentmodule:: nisl

.. autosummary::
   :toctree: generated/
   :template: function.rst

   signal.clean

.. _utils_ref:

:mod:`nisl.utils`: Manipulating Niimgs
======================================

.. automodule:: nisl.utils
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`utils` section for further details.

Functions
---------
.. currentmodule:: nisl

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.is_a_niimg
   utils.check_niimg
   utils.check_niimgs
   utils.concat_niimgs


