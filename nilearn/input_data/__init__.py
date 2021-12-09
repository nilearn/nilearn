"""
The :mod:`nilearn.input_data` module used to include masker objects.
It is deprecated since release 0.9.0 in favor of the
:mod:`~nilearn.maskers` module.

Please consider updating your code:

.. code-blocks::python
    from nilearn.input_data import NiftiMasker

becomes:

.. code-blocks::python
    from nilearn.maskers import NiftiMasker

Note that all imports that used to work will continue to do so with
a simple warning at least until release 0.13.0.
"""
import sys
import warnings

from nilearn import maskers  # noqa:F401

warnings.warn("The module 'input_data' is deprecated since 0.9.0, "
              "Importing maskers from 'input_data' will be possible "
              "at least until release 0.13.0. Please import maskers "
              "from the 'maskers' module instead.",
              FutureWarning)

sys.modules[__name__] = sys.modules['nilearn.maskers']

