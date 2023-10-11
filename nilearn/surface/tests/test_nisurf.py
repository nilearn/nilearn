# Tests for functionality in nisurf.py

import os
import tempfile

import nibabel as nb
import numpy as np
import pytest

from nilearn.surface.nisurf import (check_nisurf_1d, check_nisurf_2d,
                                    _repr_nisurfs_data, _load_surf_mask,
                                    _safe_get_data)
from numpy.testing import assert_array_equal
from nilearn._utils.exceptions import DimensionError


def test_load_with_check_nisurf_1d():
    # Test load from fake nifti file
    filename_nii = tempfile.mktemp(suffix='.nii')
    nii = nb.Nifti1Image(np.zeros((20, )), affine=None)
    nb.save(nii, filename_nii)
    assert_array_equal(check_nisurf_1d(filename_nii), np.zeros((20, )))
    os.remove(filename_nii)

    # Test load from array
    data = np.zeros((20, ))
    assert_array_equal(check_nisurf_1d(data), np.zeros((20, )))


def test_check_nisurf_1d_dimension_error():
    data = np.zeros((20, 20))

    with pytest.raises(DimensionError,
                       match="Input data has incompatible dimensionality: "
                             "Expected dimension is 1D and you provided "
                             "a 2D image."):
        check_nisurf_1d(data)


def test_proc_dtype_nisurf():

    data = np.zeros(20).astype(np.int8)
    assert 'int32' in check_nisurf_1d(data, dtype='auto').dtype.name
    assert 'bool' in check_nisurf_1d(data, dtype=np.bool).dtype.name

    data = np.ones(20).astype(float)
    assert 'float32' in check_nisurf_1d(data, dtype='auto').dtype.name


def test_load_files_with_check_nisurf_2d():
    # Test load from fake nifti files
    files = [tempfile.mktemp(suffix='.nii') for _ in range(3)]
    for f in files:
        nii = nb.Nifti1Image(np.zeros((10, )), affine=None)
        nb.save(nii, f)

    data = check_nisurf_2d(files)
    assert_array_equal(data, np.zeros((10, 3)))
    for f in files:
        os.remove(f)


def test_load_arrays_with_check_nisurf_2d():
    # Already 2D array case
    data = np.ones((10, 3))
    assert_array_equal(check_nisurf_2d(data), np.ones((10, 3)))

    # Check list of arrays cases
    data = [np.ones(10) for _ in range(3)]
    assert_array_equal(check_nisurf_2d(data), np.ones((10, 3)))


def test_check_nisurf_2d_different_sizes_error():
    data = [np.ones(10) for _ in range(3)]
    data.append(np.ones(5))

    with pytest.raises(ValueError,
                       match='When more than one file is input, all '
                              'files must contain data with the same '
                              'shape in axis=0'):
        check_nisurf_2d(data)


def test_check_bad_input_nisurf_2d():

    def fake_func():
        pass

    class fake_class():
        pass

    # Make sure bad input yields error
    bad_data = [19, fake_func, fake_class]
    for data in bad_data:
        with pytest.raises(ValueError,
                           match='The input type is not recognized. '
                                 'Valid inputs are a 2D Numpy array '
                                 'a list of 1D Numpy arrays, a valid file '
                                 'with a 2D surface or a list of valid files '
                                 'with 1D surfaces.'):
            check_nisurf_2d(data)


def test_atleast_2d():
    data = np.ones(10)
    assert_array_equal(check_nisurf_2d(data, atleast_2d=True),
                       np.ones((10, 1)))


def test_check_nisurf_2d_dimension_error():
    data = np.zeros((20, 2, 8))
    with pytest.raises(DimensionError,
                       match="Input data has incompatible dimensionality: "
                             "Expected dimension is 2D and you provided "
                             "a 3D image."):
        check_nisurf_2d(data)

    data = np.zeros((20))
    with pytest.raises(DimensionError,
                       match="Input data has incompatible dimensionality: "
                             "Expected dimension is 2D and you provided "
                             "a 1D image."):
        check_nisurf_2d(data)


def test_repr_nisurf_data():
    assert _repr_nisurfs_data('test') == 'test'
    assert _repr_nisurfs_data(np.ones(5)) == 'Surface(shape=(5,))'
    assert _repr_nisurfs_data([np.ones(5), np.ones(5)]) ==\
        "[Surface(shape=(5,)), Surface(shape=(5,))]"
    assert _repr_nisurfs_data(5) == '5'


def test_load_surf_mask():
    data = np.ones(10)
    data[:5] = 0
    assert_array_equal(_load_surf_mask(data), data.astype(bool))


def test_surf_mask_errors():
    data = np.zeros(10)
    with pytest.raises(ValueError,
                       match='The mask is invalid as it is empty: '
                             'it masks all data.'):
        _load_surf_mask(data)

    data = np.ones(10)
    data[:5] = 2
    with pytest.raises(ValueError,
                       match='Background of the mask must be represented with'
                             '0. Given mask contains:'):
        _load_surf_mask(data)

    data[-1] = 50
    with pytest.raises(ValueError,
                       match='Given mask is not made of 2 values'):
        _load_surf_mask(data)


def test_get_safe_data():

    # No copy
    original = np.ones(10)
    data = _safe_get_data(original, copy=False, ensure_finite=False)
    assert_array_equal(original, data)

    data[0] = 5
    assert(original[0] == 5)

    # Copy
    original = np.ones(10)
    data = _safe_get_data(original, copy=True, ensure_finite=False)
    assert_array_equal(original, data)

    data[0] = 5
    assert(original[0] != 5)

    # Assert finite
    original[5] = np.nan
    original[6] = np.inf
    data = _safe_get_data(original, copy=True, ensure_finite=True)
    assert np.all(np.isfinite(data))

    # Check assert finite behavior w/ copy=False
    original_copy = original.copy()
    data = _safe_get_data(original, copy=False, ensure_finite=True)
    assert np.all(np.isfinite(data))
    assert_array_equal(original, original_copy)















