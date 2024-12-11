"""Validation and conversion utilities for numpy."""

# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais

import csv
from pathlib import Path

import numpy as np


def as_ndarray(arr, copy=False, dtype=None, order="K"):
    """Convert to numpy.ndarray starting with an arbitrary array.

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

    Parameters
    ----------
    arr : array-like
        input array. Any value accepted by numpy.asarray is valid.

    copy : bool
        if True, force a copy of the array. Always True when arr is a memmap.

    dtype : any numpy dtype
        dtype of the returned array. Performing copy and type conversion at the
        same time can in some cases avoid an additional copy.

    order : :obj:`str`, default='K'
        gives the order of the returned array.
        Valid values are: "C", "F", "A", "K", None.
        See ndarray.copy() for more information.

    Returns
    -------
    ret : numpy.ndarray
        Numpy array containing the same data as arr, always of class
        numpy.ndarray, and with no link to any underlying file.

    Examples
    --------
    >>> import numpy
    >>> a = numpy.asarray([0, 1, 2], dtype=numpy.int8)
    >>> b = as_ndarray(a, dtype=bool)
    >>> b
    array([False,  True,  True])
    >>> c = as_ndarray(b, dtype=numpy.int8)
    >>> c
    array([0, 1, 2], dtype=int8)

    The usually expected result for the last line would be array([0, 1, 1])
    because True evaluates to 1. Since there is no copy made here, the original
    array is recovered.
    """
    if order not in ("C", "F", "A", "K", None):
        raise ValueError(f"Invalid value for 'order': {order!s}")

    if not isinstance(arr, (np.memmap, np.ndarray, list, tuple)):
        raise ValueError(f"Type not handled: {arr.__class__}")

    # the cases where we have to create a copy of the underlying array
    if isinstance(arr, (np.memmap, list, tuple)) or (
        isinstance(arr, np.ndarray) and copy
    ):
        return np.array(arr, copy=True, dtype=dtype, order=order)
    # if the order does not change and dtype does not change or
    # bool to/from 1-byte dtype,
    # no need to create a copy
    elif (
        (arr.itemsize == 1 and dtype in (bool, np.bool_))
        or (arr.dtype in (bool, np.bool_) and np.dtype(dtype).itemsize == 1)
        or arr.dtype == dtype
    ) and (
        (order == "F" and arr.flags["F_CONTIGUOUS"])
        or (order == "C" and arr.flags["C_CONTIGUOUS"])
        or order in ("K", "A", None)
    ):
        return arr.view(dtype=dtype)
    else:
        return np.asarray(arr, dtype=dtype, order=order)


def csv_to_array(csv_path, delimiters=" \t,;", **kwargs):
    """Read a CSV file by trying to guess its delimiter.

    Parameters
    ----------
    csv_path : string or pathlib.Path
        Path of the CSV file to load.

    delimiters : string
        Each character of the delimiters string is a potential delimiters for
        the CSV file.

    kwargs : keyword arguments
        The additional keyword arguments are passed to numpy.genfromtxt when
        loading the CSV.

    Returns
    -------
    array : numpy.ndarray
        An array containing the data loaded from the CSV file.
    """
    try:
        # First, we try genfromtxt which works in most cases.
        array = np.genfromtxt(csv_path, loose=False, encoding=None, **kwargs)
    except ValueError:
        # There was an error during the conversion to numpy array, probably
        # because the delimiter is wrong.
        # In that case, we try to guess the delimiter.
        try:
            with Path(csv_path).open() as csv_file:
                dialect = csv.Sniffer().sniff(csv_file.readline(), delimiters)
        except csv.Error as e:
            raise TypeError(
                f"Could not read CSV file [{csv_path}]: {e.args[0]}"
            )

        array = np.genfromtxt(
            csv_path, delimiter=dialect.delimiter, encoding=None, **kwargs
        )

    return array
