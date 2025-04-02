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


def pad_array(array, pad_sizes):
    """Pad an array with zeros.

    Pads an array with zeros as specified in `pad_sizes`.

    Parameters
    ----------
    array : :class:`numpy.ndarray`
        Array to pad.

    pad_sizes : :obj:`list`
        Padding quantity specified as
        *[x1minpad, x1maxpad, x2minpad,x2maxpad, x3minpad, ...]*.

    Returns
    -------
    :class:`numpy.ndarray`
        Padded array.

    Raises
    ------
    ValueError
        Inconsistent min/max padding quantities.

    """
    if len(pad_sizes) % 2 != 0:
        raise ValueError(
            "Please specify as many max paddings as min"
            f" paddings. You have specified {len(pad_sizes)} arguments"
        )

    all_paddings = np.zeros([array.ndim, 2], dtype=np.int64)
    all_paddings[: len(pad_sizes) // 2] = np.array(pad_sizes).reshape(-1, 2)

    lower_paddings, upper_paddings = all_paddings.T
    new_shape = np.array(array.shape) + upper_paddings + lower_paddings

    padded = np.zeros(new_shape, dtype=array.dtype)
    source_slices = [
        slice(max(-lp, 0), min(s + up, s))
        for lp, up, s in zip(lower_paddings, upper_paddings, array.shape)
    ]
    target_slices = [
        slice(max(lp, 0), min(s - up, s))
        for lp, up, s in zip(lower_paddings, upper_paddings, new_shape)
    ]

    padded[tuple(target_slices)] = array[tuple(source_slices)].copy()
    return padded
