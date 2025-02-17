"""Testing utilities for nilearn.image functions."""

import numpy as np
from numpy.testing import assert_array_equal


def match_headers_keys(source, target, except_keys):
    """Check if header fields of two Nifti images match, except for some keys.

    Parameters
    ----------
    source : Nifti1Image
        Source image to compare headers with.
    target : Nifti1Image
        Target image to compare headers from.
    except_keys : list of str
        List of keys that should from comparison.
    """
    for key in source.header:
        if key in except_keys:
            assert (target.header[key] != source.header[key]).any()
        elif isinstance(target.header[key], np.ndarray):
            assert_array_equal(
                target.header[key],
                source.header[key],
            )
        else:
            assert target.header[key] == source.header[key]
