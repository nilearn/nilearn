import unittest

from nibabel.optpkg import optional_package

pydicom, have_dicom, _ = optional_package('pydicom')

dicom_test = unittest.skipUnless(have_dicom, 'Could not import pydicom')
