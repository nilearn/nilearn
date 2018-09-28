"""
Validation and conversion utilities for numpy.
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD

import csv
import numpy as np
from .compat import _basestring


def _asarray(arr, dtype=None, order=None):
    # np.asarray does not take "K" and "A" orders in version 1.3.0
    if order in ("K", "A", None):
        if (arr.itemsize == 1 and dtype == np.bool) \
                or (arr.dtype == np.bool and np.dtype(dtype).itemsize == 1):
            ret = arr.view(dtype=dtype)
        else:
            ret = np.asarray(arr, dtype=dtype)
    else:
        if (((arr.itemsize == 1 and dtype == np.bool) or
            (arr.dtype == np.bool and np.dtype(dtype).itemsize == 1))
            and (order == "F" and arr.flags["F_CONTIGUOUS"]
                 or order == "C" and arr.flags["C_CONTIGUOUS"])):
            ret = arr.view(dtype=dtype)
        else:
            ret = np.asarray(arr, dtype=dtype, order=order)

    return ret


def as_ndarray(arr, copy=False, dtype=None, order='K'):
    """Starting with an arbitrary array, convert to numpy.ndarray.

    In the case of a memmap array, a copy is automatically made to break the
    link with the underlying file (whatever the value of the "copy" keyword).

    The purpose of this function is mainly to get rid of memmap objects, but
    it can be used for other purposes. In particular, combining copying and
    casting can lead to performance improvements in some cases, by avoiding
    unnecessary copies.

    If not specified, input array order is preserved, in all cases, even when
    a copy is requested.

    Caveat: this function does not copy during bool to/from 1-byte dtype
    conversions. This can lead to some surprising results in some rare cases.
    Example:

        a = numpy.asarray([0, 1, 2], dtype=numpy.int8)
        b = as_ndarray(a, dtype=bool)  # array([False, True, True], dtype=bool)
        c = as_ndarray(b, dtype=numpy.int8)  # array([0, 1, 2], dtype=numpy.int8)

    The usually expected result for the last line would be array([0, 1, 1])
    because True evaluates to 1. Since there is no copy made here, the original
    array is recovered.

    Parameters
    ----------
    arr: array-like
        input array. Any value accepted by numpy.asarray is valid.

    copy: bool
        if True, force a copy of the array. Always True when arr is a memmap.

    dtype: any numpy dtype
        dtype of the returned array. Performing copy and type conversion at the
        same time can in some cases avoid an additional copy.

    order: string
        gives the order of the returned array.
        Valid values are: "C", "F", "A", "K", None.
        default is "K". See ndarray.copy() for more information.

    Returns
    -------
    ret: numpy.ndarray
        Numpy array containing the same data as arr, always of class
        numpy.ndarray, and with no link to any underlying file.
    """
    # This function should work on numpy 1.3
    # in this version, astype() and copy() have no "order" keyword.
    # and asarray() does not accept the "K" and "A" values for order.

    # numpy.asarray never copies a subclass of numpy.ndarray (even for
    #     memmaps) when dtype is unchanged.
    # .astype() always copies

    if order not in ("C", "F", "A", "K", None):
        raise ValueError("Invalid value for 'order': %s" % str(order))

    if isinstance(arr, np.memmap):
        if dtype is None:
            if order in ("K", "A", None):
                ret = np.array(np.asarray(arr), copy=True)
            else:
                ret = np.array(np.asarray(arr), copy=True, order=order)
        else:
            if order in ("K", "A", None):
                # always copy (even when dtype does not change)
                ret = np.asarray(arr).astype(dtype)
            else:
                # First load data from disk without changing order
                # Changing order while reading through a memmap is incredibly
                # inefficient.
                ret = np.array(arr, copy=True)
                ret = _asarray(arr, dtype=dtype, order=order)

    elif isinstance(arr, np.ndarray):
        ret = _asarray(arr, dtype=dtype, order=order)
        # In the present cas, np.may_share_memory result is always reliable.
        if np.may_share_memory(ret, arr) and copy:
            # order-preserving copy
            if ret.flags["F_CONTIGUOUS"]:
                ret = ret.T.copy().T
            else:
                ret = ret.copy()

    elif isinstance(arr, (list, tuple)):
        if order in ("A", "K"):
            ret = np.asarray(arr, dtype=dtype)
        else:
            ret = np.asarray(arr, dtype=dtype, order=order)

    else:
        raise ValueError("Type not handled: %s" % arr.__class__)

    return ret


def csv_to_array(csv_path, delimiters=' \t,;', **kwargs):
    """Read a CSV file by trying to guess its delimiter

    Parameters
    ----------
    csv_path: string
        Path of the CSV file to load.

    delimiters: string
        Each character of the delimiters string is a potential delimiters for
        the CSV file.

    kwargs: keyword arguments
        The additional keyword arguments are passed to numpy.genfromtxt when
        loading the CSV.

    Returns
    -------
    array: numpy.ndarray
        An array containing the data loaded from the CSV file.
    """
    if not isinstance(csv_path, _basestring):
        raise TypeError('CSV must be a file path. Got a CSV of type: %s' %
                        type(csv_path))

    try:
        # First, we try genfromtxt which works in most cases.
        array = np.genfromtxt(csv_path, loose=False, **kwargs)
    except ValueError:
        # There was an error during the conversion to numpy array, probably
        # because the delimiter is wrong.
        # In that case, we try to guess the delimiter.
        try:
            with open(csv_path, 'r') as csv_file:
                dialect = csv.Sniffer().sniff(csv_file.readline(), delimiters)
        except csv.Error as e:
            raise TypeError(
                'Could not read CSV file [%s]: %s' % (csv_path, e.args[0]))

        array = np.genfromtxt(csv_path, delimiter=dialect.delimiter, **kwargs)

    return array
