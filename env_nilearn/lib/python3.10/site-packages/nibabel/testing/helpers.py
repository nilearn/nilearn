"""Helper functions for tests"""

from io import BytesIO

import numpy as np

from ..optpkg import optional_package

have_scipy = optional_package('scipy.io')[1]

from numpy.testing import assert_array_equal


def bytesio_filemap(klass):
    """Return bytes io filemap for this image class `klass`"""
    file_map = klass.make_file_map()
    for fileholder in file_map.values():
        fileholder.fileobj = BytesIO()
        fileholder.pos = 0
    return file_map


def bytesio_round_trip(img):
    """Save then load image from bytesio"""
    klass = img.__class__
    bytes_map = bytesio_filemap(klass)
    img.to_file_map(bytes_map)
    return klass.from_file_map(bytes_map)


def assert_data_similar(arr, params):
    """Check data is the same if recorded, otherwise check summaries

    Helper function to test image array data `arr` against record in `params`,
    where record can be the array itself, or summary values from the array.

    Parameters
    ----------
    arr : array-like
        Something that results in an array after ``np.asarry(arr)``
    params : mapping
        Mapping that has either key ``data`` with value that is array-like, or
        key ``data_summary`` with value a dict having keys ``min``, ``max``,
        ``mean``
    """
    if 'data' in params:
        assert_array_equal(arr, params['data'])
        return
    summary = params['data_summary']
    real_arr = np.asarray(arr)
    assert np.allclose(
        (real_arr.min(), real_arr.max(), real_arr.mean()),
        (summary['min'], summary['max'], summary['mean']),
    )
