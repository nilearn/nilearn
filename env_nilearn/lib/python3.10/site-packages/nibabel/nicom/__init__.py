# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""DICOM reader

.. currentmodule:: nibabel.nicom

.. autosummary::
   :toctree: ../generated

   csareader
   dicomreaders
   dicomwrappers
   dwiparams
   structreader
"""

import warnings

warnings.warn(
    'The DICOM readers are highly experimental, unstable,'
    ' and only work for Siemens time-series at the moment\n'
    'Please use with caution.  We would be grateful for your '
    'help in improving them',
    UserWarning,
    stacklevel=2,
)
