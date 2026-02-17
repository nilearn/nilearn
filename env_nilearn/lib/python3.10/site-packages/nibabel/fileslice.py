"""Utilities for getting array slices out of file-like objects"""

import operator
from functools import reduce
from mmap import mmap
from numbers import Integral

import numpy as np

# Threshold for memory gap above which we always skip, to save memory
# This value came from trying various values and looking at the timing with
# ``bench_fileslice``
SKIP_THRESH = 2**8


class _NullLock:
    """Can be used as no-function dummy object in place of ``threading.lock``.

    The ``_NullLock`` is an object which can be used in place of a
    ``threading.Lock`` object, but doesn't actually do anything.

    It is used by the ``read_segments`` function in the event that a
    ``Lock`` is not provided by the caller.
    """

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def is_fancy(sliceobj):
    """Returns True if sliceobj is attempting fancy indexing

    Parameters
    ----------
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``

    Returns
    -------
    tf: bool
        True if sliceobj represents fancy indexing, False for basic indexing
    """
    if not isinstance(sliceobj, tuple):
        sliceobj = (sliceobj,)
    for slicer in sliceobj:
        if getattr(slicer, 'ndim', 0) > 0:  # ndarray always fancy, but scalars are safe
            return True
        # slice or Ellipsis or None OK for  basic
        if isinstance(slicer, slice) or slicer in (None, Ellipsis):
            continue
        try:
            int(slicer)
        except TypeError:
            return True
    return False


def canonical_slicers(sliceobj, shape, check_inds=True):
    """Return canonical version of `sliceobj` for array shape `shape`

    `sliceobj` is a slicer for an array ``A`` implied by `shape`.

    * Expand `sliceobj` with ``slice(None)`` to add any missing (implied) axes
      in `sliceobj`
    * Find any slicers in `sliceobj` that do a full axis slice and replace by
      ``slice(None)``
    * Replace any floating point values for slicing with integers
    * Replace negative integer slice values with equivalent positive integers.

    Does not handle fancy indexing (indexing with arrays or array-like indices)

    Parameters
    ----------
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``
    shape : sequence
        shape of array that will be indexed by `sliceobj`
    check_inds : {True, False}, optional
        Whether to check if integer indices are out of bounds

    Returns
    -------
    can_slicers : tuple
        version of `sliceobj` for which Ellipses have been expanded, missing
        (implied) dimensions have been appended, and slice objects equivalent
        to ``slice(None)`` have been replaced by ``slice(None)``, integer axes
        have been checked, and negative indices set to positive equivalent
    """
    if not isinstance(sliceobj, tuple):
        sliceobj = (sliceobj,)
    if is_fancy(sliceobj):
        raise ValueError('Cannot handle fancy indexing')
    can_slicers = []
    n_dim = len(shape)
    n_real = 0
    for i, slicer in enumerate(sliceobj):
        if slicer is None:
            can_slicers.append(None)
            continue
        if slicer == Ellipsis:
            remaining = sliceobj[i + 1 :]
            if Ellipsis in remaining:
                raise ValueError('More than one Ellipsis in slicing expression')
            real_remaining = [r for r in remaining if r is not None]
            n_ellided = n_dim - n_real - len(real_remaining)
            can_slicers.extend((slice(None),) * n_ellided)
            n_real += n_ellided
            continue
        # int / slice indexing cases
        dim_len = shape[n_real]
        n_real += 1
        try:  # test for integer indexing
            slicer = int(slicer)
        except TypeError:  # should be slice object
            if slicer != slice(None):
                # Could this be full slice?
                if (
                    slicer.stop == dim_len
                    and slicer.start in (None, 0)
                    and slicer.step in (None, 1)
                ):
                    slicer = slice(None)
        else:
            if slicer < 0:
                slicer = dim_len + slicer
            elif check_inds and slicer >= dim_len:
                raise ValueError(f'Integer index {slicer} too large')
        can_slicers.append(slicer)
    # Fill out any missing dimensions
    if n_real < n_dim:
        can_slicers.extend((slice(None),) * (n_dim - n_real))
    return tuple(can_slicers)


def slice2outax(ndim, sliceobj):
    """Matching output axes for input array ndim `ndim` and slice `sliceobj`

    Parameters
    ----------
    ndim : int
        number of axes in input array
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``

    Returns
    -------
    out_ax_inds : tuple
        Say ``A` is a (pretend) input array of `ndim` dimensions. Say ``B =
        A[sliceobj]``.  `out_ax_inds` has one value per axis in ``A`` giving
        corresponding axis in ``B``.
    """
    sliceobj = canonical_slicers(sliceobj, [1] * ndim, check_inds=False)
    out_ax_no = 0
    out_ax_inds = []
    for obj in sliceobj:
        if isinstance(obj, Integral):
            out_ax_inds.append(None)
            continue
        if obj is not None:
            out_ax_inds.append(out_ax_no)
        out_ax_no += 1
    return tuple(out_ax_inds)


def slice2len(slicer, in_len):
    """Output length after slicing original length `in_len` with `slicer`
    Parameters
    ----------
    slicer : slice object
    in_len : int

    Returns
    -------
    out_len : int
        Length after slicing

    Notes
    -----
    Returns same as ``len(np.arange(in_len)[slicer])``
    """
    if slicer == slice(None):
        return in_len
    full_slicer = fill_slicer(slicer, in_len)
    return _full_slicer_len(full_slicer)


def _full_slicer_len(full_slicer):
    """Return length of slicer processed by ``fill_slicer``"""
    start, stop, step = full_slicer.start, full_slicer.stop, full_slicer.step
    if stop is None:  # case of negative step
        stop = -1
    gap = stop - start
    if (step > 0 and gap <= 0) or (step < 0 and gap >= 0):
        return 0
    return int(np.ceil(gap / step))


def fill_slicer(slicer, in_len):
    """Return slice object with Nones filled out to match `in_len`

    Also fixes too large stop / start values according to slice() slicing
    rules.

    The returned slicer can have a None as `slicer.stop` if `slicer.step` is
    negative and the input `slicer.stop` is None. This is because we can't
    represent the ``stop`` as an integer, because -1 has a different meaning.

    Parameters
    ----------
    slicer : slice object
    in_len : int
        length of axis on which `slicer` will be applied

    Returns
    -------
    can_slicer : slice object
        slice with start, stop, step set to explicit values, with the exception
        of ``stop`` for negative step, which is None for the case of slicing
        down through the first element
    """
    start, stop, step = slicer.start, slicer.stop, slicer.step
    if step is None:
        step = 1
    if start is not None and start < 0:
        start = in_len + start
    if stop is not None and stop < 0:
        stop = in_len + stop
    if step > 0:
        if start is None:
            start = 0
        if stop is None:
            stop = in_len
        else:
            stop = min(stop, in_len)
    else:  # step < 0
        if start is None:
            start = in_len - 1
        else:
            start = min(start, in_len - 1)
    return slice(start, stop, step)


def predict_shape(sliceobj, in_shape):
    """Predict shape of array from slicing array shape `shape` with `sliceobj`

    Parameters
    ----------
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``
    in_shape : sequence
        shape of array that could be sliced by `sliceobj`

    Returns
    -------
    out_shape : tuple
        predicted shape arising from slicing array shape `in_shape` with
        `sliceobj`
    """
    if not isinstance(sliceobj, tuple):
        sliceobj = (sliceobj,)
    sliceobj = canonical_slicers(sliceobj, in_shape)
    out_shape = []
    real_no = 0
    for slicer in sliceobj:
        if slicer is None:
            out_shape.append(1)
            continue
        real_no += 1
        try:  # if int - we drop a dim (no append)
            slicer = int(slicer)
        except TypeError:
            out_shape.append(slice2len(slicer, in_shape[real_no - 1]))
    return tuple(out_shape)


def _positive_slice(slicer):
    """Return full slice `slicer` enforcing positive step size

    `slicer` assumed full in the sense of :func:`fill_slicer`
    """
    start, stop, step = slicer.start, slicer.stop, slicer.step
    if step > 0:
        return slicer
    if stop is None:
        stop = -1
    gap = stop - start
    n = gap / step
    n = int(n) - 1 if int(n) == n else int(n)
    end = start + n * step
    return slice(end, start + 1, -step)


def threshold_heuristic(slicer, dim_len, stride, skip_thresh=SKIP_THRESH):
    """Whether to force full axis read or contiguous read of stepped slice

    Allows :func:`fileslice` to sometimes read memory that it will throw away
    in order to get maximum speed.  In other words, trade memory for fewer disk
    reads.

    Parameters
    ----------
    slicer : slice object, or int
        If slice, can be assumed to be full as in ``fill_slicer``
    dim_len : int
        length of axis being sliced
    stride : int
        memory distance between elements on this axis
    skip_thresh : int, optional
        Memory gap threshold in bytes above which to prefer skipping memory
        rather than reading it and later discarding.

    Returns
    -------
    action : {'full', 'contiguous', None}
        Gives the suggested optimization for reading the data

        * 'full' - read whole axis
        * 'contiguous' - read all elements between start and stop
        * None - read only memory needed for output

    Notes
    -----
    Let's say we are in the middle of reading a file at the start of some
    memory length $B$ bytes.  We don't need the memory, and we are considering
    whether to read it anyway (then throw it away) (READ) or stop reading, skip
    $B$ bytes and restart reading from there (SKIP).

    After trying some more fancy algorithms, a hard threshold (`skip_thresh`)
    for the maximum skip distance seemed to work well, as measured by times on
    ``nibabel.benchmarks.bench_fileslice``
    """
    if isinstance(slicer, Integral):
        gap_size = (dim_len - 1) * stride
        return 'full' if gap_size <= skip_thresh else None
    step_size = abs(slicer.step) * stride
    if step_size > skip_thresh:
        return None  # Prefer skip
    # At least contiguous - also full?
    slicer = _positive_slice(slicer)
    start, stop = slicer.start, slicer.stop
    read_len = stop - start
    gap_size = (dim_len - read_len) * stride
    return 'full' if gap_size <= skip_thresh else 'contiguous'


def optimize_slicer(slicer, dim_len, all_full, is_slowest, stride, heuristic=threshold_heuristic):
    """Return maybe modified slice and post-slice slicing for `slicer`

    Parameters
    ----------
    slicer : slice object or int
    dim_len : int
        length of axis along which to slice
    all_full : bool
        Whether dimensions up until now have been full (all elements)
    is_slowest : bool
        Whether this dimension is the slowest changing in memory / on disk
    stride : int
        size of one step along this axis
    heuristic : callable, optional
        function taking slice object, dim_len, stride length as arguments,
        returning one of 'full', 'contiguous', None. See
        :func:`threshold_heuristic` for an example.

    Returns
    -------
    to_read : slice object or int
        maybe modified slice based on `slicer` expressing what data should be
        read from an underlying file or buffer. `to_read` must always have
        positive ``step`` (because we don't want to go backwards in the buffer
        / file)
    post_slice : slice object
        slice to be applied after array has been read.  Applies any
        transformations in `slicer` that have not been applied in `to_read`. If
        axis will be dropped by `to_read` slicing, so no slicing would make
        sense, return string ``dropped``

    Notes
    -----
    This is the heart of the algorithm for making segments from slice objects.

    A contiguous slice is a slice with ``slice.step in (1, -1)``

    A full slice is a continuous slice returning all elements.

    The main question we have to ask is whether we should transform `to_read`,
    `post_slice` to prefer a full read and partial slice.  We only do this in
    the case of all_full==True.  In this case we might benefit from reading a
    continuous chunk of data even if the slice is not continuous, or reading
    all the data even if the slice is not full. Apply a heuristic `heuristic`
    to decide whether to do this, and adapt `to_read` and `post_slice` slice
    accordingly.

    Otherwise (apart from constraint to be positive) return `to_read` unaltered
    and `post_slice` as ``slice(None)``
    """
    # int or slice as input?
    try:  # if int - we drop a dim (no append)
        slicer = int(slicer)  # casts float to int as well
    except TypeError:  # slice
        # Deal with full cases first
        if slicer == slice(None):
            return slicer, slicer
        slicer = fill_slicer(slicer, dim_len)
        # actually equivalent to slice(None)
        if slicer == slice(0, dim_len, 1):
            return slice(None), slice(None)
        # full, but reversed
        if slicer == slice(dim_len - 1, None, -1):
            return slice(None), slice(None, None, -1)
        # Not full, maybe continuous
        is_int = False
    else:  # int
        if slicer < 0:  # make negative offsets positive
            slicer = dim_len + slicer
        is_int = True
    if all_full:
        action = heuristic(slicer, dim_len, stride)
        # Check return values (we may be using a custom function)
        if action not in ('full', 'contiguous', None):
            raise ValueError(f'Unexpected return {action} from heuristic')
        if is_int and action == 'contiguous':
            raise ValueError('int index cannot be contiguous')
        # If this is the slowest changing dimension, never upgrade None or
        # contiguous beyond contiguous (we've already covered the already-full
        # case)
        if is_slowest and action == 'full':
            action = None if is_int else 'contiguous'
        if action == 'full':
            return slice(None), slicer
        elif action == 'contiguous':  # Cannot be int
            # If this is already contiguous, default None behavior handles it
            step = slicer.step
            if step not in (-1, 1):
                if step < 0:
                    slicer = _positive_slice(slicer)
                return (slice(slicer.start, slicer.stop, 1), slice(None, None, step))
    # We only need to be positive
    if is_int:
        return slicer, 'dropped'
    if slicer.step > 0:
        return slicer, slice(None)
    return _positive_slice(slicer), slice(None, None, -1)


def calc_slicedefs(sliceobj, in_shape, itemsize, offset, order, heuristic=threshold_heuristic):
    """Return parameters for slicing array with `sliceobj` given memory layout

    Calculate the best combination of skips / (read + discard) to use for
    reading the data from disk / memory, then generate corresponding
    `segments`, the disk offsets and read lengths to read the memory.  If we
    have chosen some (read + discard) optimization, then we need to discard the
    surplus values from the read array using `post_slicers`, a slicing tuple
    that takes the array as read from a file-like object, and returns the array
    we want.

    Parameters
    ----------
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``
    in_shape : sequence
        shape of underlying array to be sliced
    itemsize : int
        element size in array (in bytes)
    offset : int
        offset of array data in underlying file or memory buffer
    order : {'C', 'F'}
        memory layout of underlying array
    heuristic : callable, optional
        function taking slice object, dim_len, stride length as arguments,
        returning one of 'full', 'contiguous', None.  See
        :func:`optimize_slicer` and :func:`threshold_heuristic`

    Returns
    -------
    segments : list
        list of 2 element lists where lists are (offset, length), giving
        absolute memory offset in bytes and number of bytes to read
    read_shape : tuple
        shape with which to interpret memory as read from `segments`.
        Interpreting the memory read from `segments` with this shape, and a
        dtype, gives an intermediate array - call this ``R``
    post_slicers : tuple
        Any new slicing to be applied to the array ``R`` after reading via
        `segments` and reshaping via `read_shape`.  Slices are in terms of
        `read_shape`.  If empty, no new slicing to apply
    """
    if order not in 'CF':
        raise ValueError("order should be one of 'CF'")
    sliceobj = canonical_slicers(sliceobj, in_shape)
    # order fastest changing first (record reordering)
    if order == 'C':
        sliceobj = sliceobj[::-1]
        in_shape = in_shape[::-1]
    # Analyze sliceobj for new read_slicers and fixup post_slicers
    # read_slicers are the virtual slices; we don't slice with these, but use
    # the slice definitions to read the relevant memory from disk
    read_slicers, post_slicers = optimize_read_slicers(sliceobj, in_shape, itemsize, heuristic)
    # work out segments corresponding to read_slicers
    segments = slicers2segments(read_slicers, in_shape, offset, itemsize)
    # Make post_slicers empty if it is the slicing identity operation
    if all(s == slice(None) for s in post_slicers):
        post_slicers = []
    read_shape = predict_shape(read_slicers, in_shape)
    # If reordered, order shape, post_slicers
    if order == 'C':
        read_shape = read_shape[::-1]
        post_slicers = post_slicers[::-1]
    return list(segments), tuple(read_shape), tuple(post_slicers)


def optimize_read_slicers(sliceobj, in_shape, itemsize, heuristic):
    """Calculates slices to read from disk, and apply after reading

    Parameters
    ----------
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``.
        Can be assumed to be canonical in the sense of ``canonical_slicers``
    in_shape : sequence
        shape of underlying array to be sliced.  Array for `in_shape` assumed
        to be already in 'F' order. Reorder shape / sliceobj for slicing a 'C'
        array before passing to this function.
    itemsize : int
        element size in array (bytes)
    heuristic : callable
        function taking slice object, axis length, and stride length as
        arguments, returning one of 'full', 'contiguous', None.  See
        :func:`optimize_slicer`; see :func:`threshold_heuristic` for an
        example.

    Returns
    -------
    read_slicers : tuple
        `sliceobj` maybe rephrased to fill out dimensions that are better read
        from disk and later trimmed to their original size with `post_slicers`.
        `read_slicers` implies a block of memory to be read from disk. The
        actual disk positions come from `slicers2segments` run over
        `read_slicers`. Includes any ``newaxis`` dimensions in `sliceobj`
    post_slicers : tuple
        Any new slicing to be applied to the read array after reading.  The
        `post_slicers` discard any memory that we read to save time, but that
        we don't need for the slice.  Include any ``newaxis`` dimension added
        by `sliceobj`
    """
    read_slicers = []
    post_slicers = []
    real_no = 0
    stride = itemsize
    all_full = True
    for slicer in sliceobj:
        if slicer is None:
            read_slicers.append(None)
            post_slicers.append(slice(None))
            continue
        dim_len = in_shape[real_no]
        real_no += 1
        is_last = real_no == len(in_shape)
        # make modified sliceobj (to_read, post_slice)
        read_slicer, post_slicer = optimize_slicer(
            slicer, dim_len, all_full, is_last, stride, heuristic
        )
        read_slicers.append(read_slicer)
        all_full = all_full and read_slicer == slice(None)
        if not isinstance(read_slicer, Integral):
            post_slicers.append(post_slicer)
        stride *= dim_len
    return tuple(read_slicers), tuple(post_slicers)


def slicers2segments(read_slicers, in_shape, offset, itemsize):
    """Get segments from `read_slicers` given `in_shape` and memory steps

    Parameters
    ----------
    read_slicers : object
        something that can be used to slice an array as in ``arr[sliceobj]``
        Slice objects can by be assumed canonical as in ``canonical_slicers``,
        and positive as in ``_positive_slice``
    in_shape : sequence
        shape of underlying array on disk before reading
    offset : int
        offset of array data in underlying file or memory buffer
    itemsize : int
        element size in array (in bytes)

    Returns
    -------
    segments : list
        list of 2 element lists where lists are [offset, length], giving
        absolute memory offset in bytes and number of bytes to read
    """
    all_full = True
    all_segments = [[offset, itemsize]]
    stride = itemsize
    real_no = 0
    for read_slicer in read_slicers:
        if read_slicer is None:
            continue
        dim_len = in_shape[real_no]
        real_no += 1
        is_int = isinstance(read_slicer, Integral)
        if not is_int:  # slicer is (now) a slice
            # make slice full (it will always be positive)
            read_slicer = fill_slicer(read_slicer, dim_len)
            slice_len = _full_slicer_len(read_slicer)
        is_full = read_slicer == slice(0, dim_len, 1)
        is_contiguous = not is_int and read_slicer.step == 1
        if all_full and is_contiguous:  # full or contiguous
            if read_slicer.start != 0:
                all_segments[0][0] += stride * read_slicer.start
            all_segments[0][1] *= slice_len
        else:  # Previous or current stuff is not contiguous
            if is_int:
                for segment in all_segments:
                    segment[0] += stride * read_slicer
            else:  # slice object
                segments = all_segments
                all_segments = []
                for i in range(read_slicer.start, read_slicer.stop, read_slicer.step):
                    for s in segments:
                        all_segments.append([s[0] + stride * i, s[1]])
        all_full = all_full and is_full
        stride *= dim_len
    return all_segments


def read_segments(fileobj, segments, n_bytes, lock=None):
    """Read `n_bytes` byte data implied by `segments` from `fileobj`

    Parameters
    ----------
    fileobj : file-like object
        Implements `seek` and `read`
    segments : sequence
        list of 2 sequences where sequences are (offset, length), giving
        absolute file offset in bytes and number of bytes to read
    n_bytes : int
        total number of bytes that will be read
    lock : {None, threading.Lock, lock-like} optional
        If provided, used to ensure that paired calls to ``seek`` and ``read``
        cannot be interrupted by another thread accessing the same ``fileobj``.
        Each thread which accesses the same file via ``read_segments`` must
        share a lock in order to ensure that the file access is thread-safe.
        A lock does not need to be provided for single-threaded access. The
        default value (``None``) results in a lock-like object  (a
        ``_NullLock``) which does not do anything.

    Returns
    -------
    buffer : buffer object
        object implementing buffer protocol, such as byte string or ndarray or
        mmap or ctypes ``c_char_array``
    """
    # Make a lock-like thing to make the code below a bit nicer
    if lock is None:
        lock = _NullLock()

    if len(segments) == 0:
        if n_bytes != 0:
            raise ValueError('No segments, but non-zero n_bytes')
        return b''
    if len(segments) == 1:
        offset, length = segments[0]
        with lock:
            fileobj.seek(offset)
            bytes = fileobj.read(length)
        if len(bytes) != n_bytes:
            raise ValueError('Whoops, not enough data in file')
        return bytes
    # More than one segment
    bytes = mmap(-1, n_bytes)
    for offset, length in segments:
        with lock:
            fileobj.seek(offset)
            bytes.write(fileobj.read(length))
    if bytes.tell() != n_bytes:
        raise ValueError('Oh dear, n_bytes does not look right')
    return bytes


def _simple_fileslice(fileobj, sliceobj, shape, dtype, offset=0, order='C', heuristic=None):
    """Read all data from `fileobj` into array, then slice with `sliceobj`

    The simplest possible thing; read all the data into the full array, then
    slice the full array.

    Parameters
    ----------
    fileobj : file-like object
        implements ``read`` and ``seek``
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``
    shape : sequence
        shape of full array inside `fileobj`
    dtype : dtype object
        dtype of array inside `fileobj`
    offset : int, optional
        offset of array data within `fileobj`
    order : {'C', 'F'}, optional
        memory layout of array in `fileobj`
    heuristic : optional
        The routine doesn't use `heuristic`; the parameter is for API
        compatibility with :func:`fileslice`

    Returns
    -------
    sliced_arr : array
        Array in `fileobj` as sliced with `sliceobj`
    """
    fileobj.seek(offset)
    nbytes = reduce(operator.mul, shape) * dtype.itemsize
    bytes = fileobj.read(nbytes)
    new_arr = np.ndarray(shape, dtype, buffer=bytes, order=order)
    return new_arr[sliceobj]


def fileslice(
    fileobj, sliceobj, shape, dtype, offset=0, order='C', heuristic=threshold_heuristic, lock=None
):
    """Slice array in `fileobj` using `sliceobj` slicer and array definitions

    `fileobj` contains the contiguous binary data for an array ``A`` of shape,
    dtype, memory layout `shape`, `dtype`, `order`, with the binary data
    starting at file offset `offset`.

    Our job is to return the sliced array ``A[sliceobj]`` in the most efficient
    way in terms of memory and time.

    Sometimes it will be quicker to read memory that we will later throw away,
    to save time we might lose doing short seeks on `fileobj`.  Call these
    alternatives: (read + discard); and skip.  This routine guesses when to
    (read+discard) or skip using the callable `heuristic`, with a default using
    a hard threshold for the memory gap large enough to prefer a skip.

    Parameters
    ----------
    fileobj : file-like object
        file-like object, opened for reading in binary mode. Implements
        ``read`` and ``seek``.
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``.
    shape : sequence
        shape of full array inside `fileobj`.
    dtype : dtype specifier
        dtype of array inside `fileobj`, or input to ``numpy.dtype`` to specify
        array dtype.
    offset : int, optional
        offset of array data within `fileobj`
    order : {'C', 'F'}, optional
        memory layout of array in `fileobj`.
    heuristic : callable, optional
        function taking slice object, axis length, stride length as arguments,
        returning one of 'full', 'contiguous', None.  See
        :func:`optimize_slicer` and see :func:`threshold_heuristic` for an
        example.
    lock : {None, threading.Lock, lock-like} optional
        If provided, used to ensure that paired calls to ``seek`` and ``read``
        cannot be interrupted by another thread accessing the same ``fileobj``.
        Each thread which accesses the same file via ``read_segments`` must
        share a lock in order to ensure that the file access is thread-safe.
        A lock does not need to be provided for single-threaded access. The
        default value (``None``) results in a lock-like object  (a
        ``_NullLock``) which does not do anything.

    Returns
    -------
    sliced_arr : array
        Array in `fileobj` as sliced with `sliceobj`
    """
    if is_fancy(sliceobj):
        raise ValueError('Cannot handle fancy indexing')
    dtype = np.dtype(dtype)
    itemsize = int(dtype.itemsize)
    segments, sliced_shape, post_slicers = calc_slicedefs(sliceobj, shape, itemsize, offset, order)
    n_bytes = reduce(operator.mul, sliced_shape, 1) * itemsize
    arr_data = read_segments(fileobj, segments, n_bytes, lock)
    sliced = np.ndarray(sliced_shape, dtype, buffer=arr_data, order=order)
    return sliced[post_slicers]


def strided_scalar(shape, scalar=0.0):
    """Return array shape `shape` where all entries point to value `scalar`

    Parameters
    ----------
    shape : sequence
        Shape of output array.
    scalar : scalar
        Scalar value with which to fill array.

    Returns
    -------
    strided_arr : array
        Array of shape `shape` for which all values == `scalar`, built by
        setting all strides of `strided_arr` to 0, so the scalar is broadcast
        out to the full array `shape`. `strided_arr` is flagged as not
        `writeable`.

        The array is set read-only to avoid a numpy error when broadcasting -
        see https://github.com/numpy/numpy/issues/6491
    """
    shape = tuple(shape)
    scalar = np.array(scalar)
    strides = [0] * len(shape)
    strided_scalar = np.lib.stride_tricks.as_strided(scalar, shape, strides)
    strided_scalar.flags.writeable = False
    return strided_scalar
