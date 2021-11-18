"""
The :mod:`nilearn.input_data` module used to include masker objects.
It is deprecated since release 0.8.2 in favor of the
:mod:`~nilearn.maskers` module.

Please consider updating your code:

.. code-blocks::python
    from nilearn.input_data import NiftiMasker

becomes:

.. code-blocks::python
    from nilearn.maskers import NiftiMasker

Note that all imports that used to work will continue to do so with
a simple warning.
"""
import warnings


from nilearn.maskers import (
    NiftiMasker, MultiNiftiMasker, NiftiLabelsMasker,
    NiftiMapsMasker, NiftiSpheresMasker
)

warnings.warn("The module 'input_data' is deprecated since 0.8.2. "
              "Please import maskers from the 'maskers' module.")


__all__ = ['NiftiMasker', 'MultiNiftiMasker', 'NiftiLabelsMasker',
           'NiftiMapsMasker', 'NiftiSpheresMasker']
