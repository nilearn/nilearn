"""Testing reading DICOM files"""

from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from nibabel.optpkg import optional_package

from .. import dicomreaders as didr
from .test_dicomwrappers import DATA, EXPECTED_AFFINE, EXPECTED_PARAMS, IO_DATA_PATH

pydicom, _, setup_module = optional_package('pydicom')


def test_read_dwi():
    img = didr.mosaic_to_nii(DATA)
    arr = img.get_fdata()
    assert arr.shape == (128, 128, 48)
    assert_array_almost_equal(img.affine, EXPECTED_AFFINE)


def test_read_dwis():
    data, aff, bs, gs = didr.read_mosaic_dwi_dir(IO_DATA_PATH, 'siemens_dwi_*.dcm.gz')
    assert data.ndim == 4
    assert_array_almost_equal(aff, EXPECTED_AFFINE)
    assert_array_almost_equal(bs, (0, EXPECTED_PARAMS[0]))
    assert_array_almost_equal(gs, (np.zeros((3,)), EXPECTED_PARAMS[1]))
    with pytest.raises(OSError):
        didr.read_mosaic_dwi_dir('improbable')


def test_passing_kwds():
    # Check that we correctly pass keywords to dicom
    dwi_glob = 'siemens_dwi_*.dcm.gz'
    csa_glob = 'csa*.bin'
    for func in (didr.read_mosaic_dwi_dir, didr.read_mosaic_dir):
        data, aff, bs, gs = func(IO_DATA_PATH, dwi_glob)
        # This should not raise an error
        data2, aff2, bs2, gs2 = func(IO_DATA_PATH, dwi_glob, dicom_kwargs=dict(force=True))
        assert_array_equal(data, data2)
        # This should raise an error in pydicom.filereader.dcmread
        with pytest.raises(TypeError):
            func(IO_DATA_PATH, dwi_glob, dicom_kwargs=dict(not_a_parameter=True))
        # These are invalid dicoms, so will raise an error unless force=True
        with pytest.raises(pydicom.filereader.InvalidDicomError):
            func(IO_DATA_PATH, csa_glob)
        # But here, we catch the error because the dicoms are in the wrong
        # format
        with pytest.raises(didr.DicomReadError):
            func(IO_DATA_PATH, csa_glob, dicom_kwargs=dict(force=True))


def test_slices_to_series():
    dicom_files = (pjoin(IO_DATA_PATH, f'{i}.dcm') for i in range(2))
    wrappers = [didr.wrapper_from_file(f) for f in dicom_files]
    series = didr.slices_to_series(wrappers)
    assert len(series) == 1
    assert len(series[0]) == 2
