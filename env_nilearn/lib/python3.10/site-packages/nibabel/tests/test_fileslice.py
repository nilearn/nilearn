"""Test slicing of file-like objects"""

import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ..fileslice import (
    _positive_slice,
    _simple_fileslice,
    calc_slicedefs,
    canonical_slicers,
    fileslice,
    fill_slicer,
    is_fancy,
    optimize_read_slicers,
    optimize_slicer,
    predict_shape,
    read_segments,
    slice2len,
    slice2outax,
    slicers2segments,
    strided_scalar,
    threshold_heuristic,
)


def _check_slice(sliceobj):
    # Fancy indexing always returns a copy, basic indexing returns a view
    a = np.arange(100).reshape((10, 10))
    b = a[sliceobj]
    if np.isscalar(b):
        return  # Can't check
    # Check if this is a view
    a[:] = 99
    b_is_view = np.all(b == 99)
    assert (not is_fancy(sliceobj)) == b_is_view


def test_is_fancy():
    slices = (2, [2], [2, 3], Ellipsis, np.array((2, 3)))
    for slice0 in slices:
        _check_slice(slice0)
        _check_slice((slice0,))  # tuple is same
        # Double ellipsis illegal in np 1.12dev - set up check for that case
        maybe_bad = slice0 is Ellipsis
        for slice1 in slices:
            if maybe_bad and slice1 is Ellipsis:
                continue
            _check_slice((slice0, slice1))
    assert not is_fancy((None,))
    assert not is_fancy((None, 1))
    assert not is_fancy((1, None))
    # Check that actual False returned (rather than falsey)
    assert is_fancy(1) is False


def test_canonical_slicers():
    # Check transformation of sliceobj into canonical form
    slicers = (slice(None), slice(9), slice(0, 9), slice(1, 10), slice(1, 10, 2), 2, np.array(2))

    shape = (10, 10)
    for slice0 in slicers:
        assert canonical_slicers((slice0,), shape) == (slice0, slice(None))
        for slice1 in slicers:
            sliceobj = (slice0, slice1)
            assert canonical_slicers(sliceobj, shape) == sliceobj
            assert canonical_slicers(sliceobj, shape + (2, 3, 4)) == sliceobj + (slice(None),) * 3
            assert canonical_slicers(sliceobj * 3, shape * 3) == sliceobj * 3
            # Check None passes through
            assert canonical_slicers(sliceobj + (None,), shape) == sliceobj + (None,)
            assert canonical_slicers((None,) + sliceobj, shape) == (None,) + sliceobj
            assert canonical_slicers((None,) + sliceobj + (None,), shape) == (None,) + sliceobj + (
                None,
            )
    # Check Ellipsis
    assert canonical_slicers((Ellipsis,), shape) == (slice(None), slice(None))
    assert canonical_slicers((Ellipsis, None), shape) == (slice(None), slice(None), None)
    assert canonical_slicers((Ellipsis, 1), shape) == (slice(None), 1)
    assert canonical_slicers((1, Ellipsis), shape) == (1, slice(None))
    # Ellipsis at end does nothing
    assert canonical_slicers((1, 1, Ellipsis), shape) == (1, 1)
    assert canonical_slicers((1, Ellipsis, 2), (10, 1, 2, 3, 11)) == (
        1,
        slice(None),
        slice(None),
        slice(None),
        2,
    )
    with pytest.raises(ValueError):
        canonical_slicers((Ellipsis, 1, Ellipsis), (2, 3, 4, 5))
    # Check full slices get expanded
    for slice0 in (slice(10), slice(0, 10), slice(0, 10, 1)):
        assert canonical_slicers((slice0, 1), shape) == (slice(None), 1)
    for slice0 in (slice(10), slice(0, 10), slice(0, 10, 1)):
        assert canonical_slicers((slice0, 1), shape) == (slice(None), 1)
        assert canonical_slicers((1, slice0), shape) == (1, slice(None))
    # Check ints etc get parsed through to tuples
    assert canonical_slicers(1, shape) == (1, slice(None))
    assert canonical_slicers(slice(None), shape) == (slice(None), slice(None))
    # Check fancy indexing raises error
    with pytest.raises(ValueError):
        canonical_slicers((np.array([1]), 1), shape)
    with pytest.raises(ValueError):
        canonical_slicers((1, np.array([1])), shape)
    # Check out of range integer raises error
    with pytest.raises(ValueError):
        canonical_slicers((10,), shape)
    with pytest.raises(ValueError):
        canonical_slicers((1, 10), shape)
    with pytest.raises(ValueError):
        canonical_slicers((10,), shape, True)
    with pytest.raises(ValueError):
        canonical_slicers((1, 10), shape, True)
    # Unless check_inds is False
    assert canonical_slicers((10,), shape, False) == (10, slice(None))
    assert canonical_slicers((1, 10), shape, False) == (1, 10)
    # Check negative -> positive
    assert canonical_slicers(-1, shape) == (9, slice(None))
    assert canonical_slicers((slice(None), -1), shape) == (slice(None), 9)
    # check numpy integer scalars behave the same as numpy integers
    assert canonical_slicers(np.array(2), shape) == canonical_slicers(2, shape)
    assert canonical_slicers((np.array(2), np.array(1)), shape) == canonical_slicers((2, 1), shape)
    assert canonical_slicers((2, np.array(1)), shape) == canonical_slicers((2, 1), shape)
    assert canonical_slicers((np.array(2), 1), shape) == canonical_slicers((2, 1), shape)


def test_slice2outax():
    # Test function giving output axes from input ndims and slice
    sn = slice(None)
    assert slice2outax(1, (sn,)) == (0,)
    assert slice2outax(1, (1,)) == (None,)
    assert slice2outax(1, (None,)) == (1,)
    assert slice2outax(1, (None, 1)) == (None,)
    assert slice2outax(1, (None, 1, None)) == (None,)
    assert slice2outax(1, (None, sn)) == (1,)
    assert slice2outax(2, (sn,)) == (0, 1)
    assert slice2outax(2, (sn, sn)) == (0, 1)
    assert slice2outax(2, (1,)) == (None, 0)
    assert slice2outax(2, (sn, 1)) == (0, None)
    assert slice2outax(2, (None,)) == (1, 2)
    assert slice2outax(2, (None, 1)) == (None, 1)
    assert slice2outax(2, (None, 1, None)) == (None, 2)
    assert slice2outax(2, (None, 1, None, 2)) == (None, None)
    assert slice2outax(2, (None, sn, None, 1)) == (1, None)
    assert slice2outax(3, (sn,)) == (0, 1, 2)
    assert slice2outax(3, (sn, sn)) == (0, 1, 2)
    assert slice2outax(3, (sn, None, sn)) == (0, 2, 3)
    assert slice2outax(3, (sn, None, sn, None, sn)) == (0, 2, 4)
    assert slice2outax(3, (1,)) == (None, 0, 1)
    assert slice2outax(3, (None, sn, None, 1)) == (1, None, 3)


def _slices_for_len(L):
    # Example slices for a dimension of length L
    if L == 0:
        raise ValueError('Need length > 0')
    sdefs = [0, L // 2, L - 1, -1, slice(None), slice(L - 1)]
    if L > 1:
        sdefs += [
            -2,
            slice(1, L - 1),
            slice(1, L - 1, 2),
            slice(L - 1, 1, -1),
            slice(L - 1, 1, -2),
        ]
    return tuple(sdefs)


def test_slice2len():
    # Test slice length calculation
    assert slice2len(slice(None), 10) == 10
    assert slice2len(slice(11), 10) == 10
    assert slice2len(slice(1, 11), 10) == 9
    assert slice2len(slice(1, 1), 10) == 0
    assert slice2len(slice(1, 11, 2), 10) == 5
    assert slice2len(slice(0, 11, 3), 10) == 4
    assert slice2len(slice(1, 11, 3), 10) == 3
    assert slice2len(slice(None, None, -1), 10) == 10
    assert slice2len(slice(11, None, -1), 10) == 10
    assert slice2len(slice(None, 1, -1), 10) == 8
    assert slice2len(slice(None, None, -2), 10) == 5
    assert slice2len(slice(None, None, -3), 10) == 4
    assert slice2len(slice(None, 0, -3), 10) == 3
    # Start, end are always taken to be relative if negative
    assert slice2len(slice(None, -4, -1), 10) == 3
    assert slice2len(slice(-4, -2, 1), 10) == 2
    # start after stop
    assert slice2len(slice(3, 2, 1), 10) == 0
    assert slice2len(slice(2, 3, -1), 10) == 0


def test_fill_slicer():
    # Test slice length calculation
    assert fill_slicer(slice(None), 10) == slice(0, 10, 1)
    assert fill_slicer(slice(11), 10) == slice(0, 10, 1)
    assert fill_slicer(slice(1, 11), 10) == slice(1, 10, 1)
    assert fill_slicer(slice(1, 1), 10) == slice(1, 1, 1)
    assert fill_slicer(slice(1, 11, 2), 10) == slice(1, 10, 2)
    assert fill_slicer(slice(0, 11, 3), 10) == slice(0, 10, 3)
    assert fill_slicer(slice(1, 11, 3), 10) == slice(1, 10, 3)
    assert fill_slicer(slice(None, None, -1), 10) == slice(9, None, -1)
    assert fill_slicer(slice(11, None, -1), 10) == slice(9, None, -1)
    assert fill_slicer(slice(None, 1, -1), 10) == slice(9, 1, -1)
    assert fill_slicer(slice(None, None, -2), 10) == slice(9, None, -2)
    assert fill_slicer(slice(None, None, -3), 10) == slice(9, None, -3)
    assert fill_slicer(slice(None, 0, -3), 10) == slice(9, 0, -3)
    # Start, end are always taken to be relative if negative
    assert fill_slicer(slice(None, -4, -1), 10) == slice(9, 6, -1)
    assert fill_slicer(slice(-4, -2, 1), 10) == slice(6, 8, 1)
    # start after stop
    assert fill_slicer(slice(3, 2, 1), 10) == slice(3, 2, 1)
    assert fill_slicer(slice(2, 3, -1), 10) == slice(2, 3, -1)


def test__positive_slice():
    # Reverse slice direction to be positive
    assert _positive_slice(slice(0, 5, 1)) == slice(0, 5, 1)
    assert _positive_slice(slice(1, 5, 3)) == slice(1, 5, 3)
    assert _positive_slice(slice(4, None, -2)) == slice(0, 5, 2)
    assert _positive_slice(slice(4, None, -1)) == slice(0, 5, 1)
    assert _positive_slice(slice(4, 1, -1)) == slice(2, 5, 1)
    assert _positive_slice(slice(4, 1, -2)) == slice(2, 5, 2)


def test_threshold_heuristic():
    # Test for default skip / read heuristic
    # int
    assert threshold_heuristic(1, 9, 1, skip_thresh=8) == 'full'
    assert threshold_heuristic(1, 9, 1, skip_thresh=7) is None
    assert threshold_heuristic(1, 9, 2, skip_thresh=16) == 'full'
    assert threshold_heuristic(1, 9, 2, skip_thresh=15) is None
    # full slice, smallest step size
    assert threshold_heuristic(slice(0, 9, 1), 9, 2, skip_thresh=2) == 'full'
    # Dropping skip thresh below step size gives None
    assert threshold_heuristic(slice(0, 9, 1), 9, 2, skip_thresh=1) == None
    # As does increasing step size
    assert threshold_heuristic(slice(0, 9, 2), 9, 2, skip_thresh=3) == None
    # Negative step size same as positive
    assert threshold_heuristic(slice(9, None, -1), 9, 2, skip_thresh=2) == 'full'
    # Add a gap between start and end. Now contiguous because of step size
    assert threshold_heuristic(slice(2, 9, 1), 9, 2, skip_thresh=2) == 'contiguous'
    # To not-contiguous, even with step size 1
    assert threshold_heuristic(slice(2, 9, 1), 9, 2, skip_thresh=1) == None
    # Back to full when skip covers gap
    assert threshold_heuristic(slice(2, 9, 1), 9, 2, skip_thresh=4) == 'full'
    # Until it doesn't cover the gap
    assert threshold_heuristic(slice(2, 9, 1), 9, 2, skip_thresh=3) == 'contiguous'


# Some dummy heuristics for optimize_slicer
def _always(slicer, dim_len, stride):
    return 'full'


def _partial(slicer, dim_len, stride):
    return 'contiguous'


def _never(slicer, dim_len, stride):
    return None


def test_optimize_slicer():
    # Analyze slice for fullness, contiguity, direction
    #
    # If all_full:
    # - make positive slicer
    # - decide if worth reading continuous block
    # - if so, modify as_read, as_returned accordingly, set contiguous / full
    # - if not, fill as_read for non-contiguous case
    # If not all_full
    # - make positive slicer
    for all_full in (True, False):
        for heuristic in (_always, _never, _partial):
            for is_slowest in (True, False):
                # following tests not affected by all_full or optimization
                # full - always passes through
                assert optimize_slicer(slice(None), 10, all_full, is_slowest, 4, heuristic) == (
                    slice(None),
                    slice(None),
                )
                # Even if full specified with explicit values
                assert optimize_slicer(slice(10), 10, all_full, is_slowest, 4, heuristic) == (
                    slice(None),
                    slice(None),
                )
                assert optimize_slicer(slice(0, 10), 10, all_full, is_slowest, 4, heuristic) == (
                    slice(None),
                    slice(None),
                )
                assert optimize_slicer(
                    slice(0, 10, 1), 10, all_full, is_slowest, 4, heuristic
                ) == (slice(None), slice(None))
                # Reversed full is still full, but with reversed post_slice
                assert optimize_slicer(
                    slice(None, None, -1), 10, all_full, is_slowest, 4, heuristic
                ) == (slice(None), slice(None, None, -1))
    # Contiguous is contiguous unless heuristic kicks in, in which case it may
    # be 'full'
    assert optimize_slicer(slice(9), 10, False, False, 4, _always) == (slice(0, 9, 1), slice(None))
    assert optimize_slicer(slice(9), 10, True, False, 4, _always) == (slice(None), slice(0, 9, 1))
    # Unless this is the slowest dimension, and all_true is True, in which case
    # we don't update to full
    assert optimize_slicer(slice(9), 10, True, True, 4, _always) == (slice(0, 9, 1), slice(None))
    # Nor if the heuristic won't update
    assert optimize_slicer(slice(9), 10, True, False, 4, _never) == (slice(0, 9, 1), slice(None))
    assert optimize_slicer(slice(1, 10), 10, True, False, 4, _never) == (
        slice(1, 10, 1),
        slice(None),
    )
    # Reversed contiguous still contiguous
    assert optimize_slicer(slice(8, None, -1), 10, False, False, 4, _never) == (
        slice(0, 9, 1),
        slice(None, None, -1),
    )
    assert optimize_slicer(slice(8, None, -1), 10, True, False, 4, _always) == (
        slice(None),
        slice(8, None, -1),
    )
    assert optimize_slicer(slice(8, None, -1), 10, False, False, 4, _never) == (
        slice(0, 9, 1),
        slice(None, None, -1),
    )
    assert optimize_slicer(slice(9, 0, -1), 10, False, False, 4, _never) == (
        slice(1, 10, 1),
        slice(None, None, -1),
    )
    # Non-contiguous
    assert optimize_slicer(slice(0, 10, 2), 10, False, False, 4, _never) == (
        slice(0, 10, 2),
        slice(None),
    )
    # all_full triggers optimization, but optimization does nothing
    assert optimize_slicer(slice(0, 10, 2), 10, True, False, 4, _never) == (
        slice(0, 10, 2),
        slice(None),
    )
    # all_full triggers optimization, optimization does something
    assert optimize_slicer(slice(0, 10, 2), 10, True, False, 4, _always) == (
        slice(None),
        slice(0, 10, 2),
    )
    # all_full disables optimization, optimization does something
    assert optimize_slicer(slice(0, 10, 2), 10, False, False, 4, _always) == (
        slice(0, 10, 2),
        slice(None),
    )
    # Non contiguous, reversed
    assert optimize_slicer(slice(10, None, -2), 10, False, False, 4, _never) == (
        slice(1, 10, 2),
        slice(None, None, -1),
    )
    assert optimize_slicer(slice(10, None, -2), 10, True, False, 4, _always) == (
        slice(None),
        slice(9, None, -2),
    )
    # Short non-contiguous
    assert optimize_slicer(slice(2, 8, 2), 10, False, False, 4, _never) == (
        slice(2, 8, 2),
        slice(None),
    )
    # with partial read
    assert optimize_slicer(slice(2, 8, 2), 10, True, False, 4, _partial) == (
        slice(2, 8, 1),
        slice(None, None, 2),
    )
    # If this is the slowest changing dimension, heuristic can upgrade None to
    # contiguous, but not (None, contiguous) to full
    # we've done this one already
    assert optimize_slicer(slice(0, 10, 2), 10, True, False, 4, _always) == (
        slice(None),
        slice(0, 10, 2),
    )
    # if slowest, just upgrade to contiguous
    assert optimize_slicer(slice(0, 10, 2), 10, True, True, 4, _always) == (
        slice(0, 10, 1),
        slice(None, None, 2),
    )
    # contiguous does not upgrade to full
    assert optimize_slicer(slice(9), 10, True, True, 4, _always) == (slice(0, 9, 1), slice(None))
    # integer
    assert optimize_slicer(0, 10, True, False, 4, _never) == (0, 'dropped')
    # can be negative
    assert optimize_slicer(-1, 10, True, False, 4, _never) == (9, 'dropped')
    # or float
    assert optimize_slicer(0.9, 10, True, False, 4, _never) == (0, 'dropped')
    # should never get 'contiguous'
    with pytest.raises(ValueError):
        optimize_slicer(0, 10, True, False, 4, _partial)
    # full can be forced with heuristic
    assert optimize_slicer(0, 10, True, False, 4, _always) == (slice(None), 0)
    # but disabled for slowest changing dimension
    assert optimize_slicer(0, 10, True, True, 4, _always) == (0, 'dropped')


def test_optimize_read_slicers():
    # Test function to optimize read slicers
    assert optimize_read_slicers((1,), (10,), 4, _never) == ((1,), ())
    assert optimize_read_slicers((slice(None),), (10,), 4, _never) == (
        (slice(None),),
        (slice(None),),
    )
    assert optimize_read_slicers((slice(9),), (10,), 4, _never) == (
        (slice(0, 9, 1),),
        (slice(None),),
    )
    # optimize cannot update a continuous to a full if last
    assert optimize_read_slicers((slice(9),), (10,), 4, _always) == (
        (slice(0, 9, 1),),
        (slice(None),),
    )
    # optimize can update non-contiguous to continuous even if last
    # not optimizing
    assert optimize_read_slicers((slice(0, 9, 2),), (10,), 4, _never) == (
        (slice(0, 9, 2),),
        (slice(None),),
    )
    # optimizing
    assert optimize_read_slicers((slice(0, 9, 2),), (10,), 4, _always) == (
        (slice(0, 9, 1),),
        (slice(None, None, 2),),
    )
    # Optimize does nothing for integer when last
    assert optimize_read_slicers((1,), (10,), 4, _always) == ((1,), ())
    # 2D
    assert optimize_read_slicers((slice(None), slice(None)), (10, 6), 4, _never) == (
        (slice(None), slice(None)),
        (slice(None), slice(None)),
    )
    assert optimize_read_slicers((slice(None), 1), (10, 6), 4, _never) == (
        (slice(None), 1),
        (slice(None),),
    )
    assert optimize_read_slicers((1, slice(None)), (10, 6), 4, _never) == (
        (1, slice(None)),
        (slice(None),),
    )
    # Not optimizing a partial slice
    assert optimize_read_slicers((slice(9), slice(None)), (10, 6), 4, _never) == (
        (slice(0, 9, 1), slice(None)),
        (slice(None), slice(None)),
    )
    # Optimizing a partial slice
    assert optimize_read_slicers((slice(9), slice(None)), (10, 6), 4, _always) == (
        (slice(None), slice(None)),
        (slice(0, 9, 1), slice(None)),
    )
    # Optimize cannot update a continuous to a full if last
    assert optimize_read_slicers((slice(None), slice(5)), (10, 6), 4, _always) == (
        (slice(None), slice(0, 5, 1)),
        (slice(None), slice(None)),
    )
    # optimize can update non-contiguous to full if not last
    # not optimizing
    assert optimize_read_slicers((slice(0, 9, 3), slice(None)), (10, 6), 4, _never) == (
        (slice(0, 9, 3), slice(None)),
        (slice(None), slice(None)),
    )
    # optimizing full
    assert optimize_read_slicers((slice(0, 9, 3), slice(None)), (10, 6), 4, _always) == (
        (slice(None), slice(None)),
        (slice(0, 9, 3), slice(None)),
    )
    # optimizing partial
    assert optimize_read_slicers((slice(0, 9, 3), slice(None)), (10, 6), 4, _partial) == (
        (slice(0, 9, 1), slice(None)),
        (slice(None, None, 3), slice(None)),
    )
    # optimize can update non-contiguous to continuous even if last
    # not optimizing
    assert optimize_read_slicers((slice(None), slice(0, 5, 2)), (10, 6), 4, _never) == (
        (slice(None), slice(0, 5, 2)),
        (slice(None), slice(None)),
    )
    # optimizing
    assert optimize_read_slicers((slice(None), slice(0, 5, 2)), (10, 6), 4, _always) == (
        (slice(None), slice(0, 5, 1)),
        (slice(None), slice(None, None, 2)),
    )
    # Optimize does nothing for integer when last
    assert optimize_read_slicers((slice(None), 1), (10, 6), 4, _always) == (
        (slice(None), 1),
        (slice(None),),
    )
    # Check gap threshold with 3D
    _depends0 = partial(threshold_heuristic, skip_thresh=10 * 4 - 1)
    _depends1 = partial(threshold_heuristic, skip_thresh=10 * 4)
    assert optimize_read_slicers(
        (slice(9), slice(None), slice(None)), (10, 6, 2), 4, _depends0
    ) == ((slice(None), slice(None), slice(None)), (slice(0, 9, 1), slice(None), slice(None)))
    assert optimize_read_slicers(
        (slice(None), slice(5), slice(None)), (10, 6, 2), 4, _depends0
    ) == ((slice(None), slice(0, 5, 1), slice(None)), (slice(None), slice(None), slice(None)))
    assert optimize_read_slicers(
        (slice(None), slice(5), slice(None)), (10, 6, 2), 4, _depends1
    ) == ((slice(None), slice(None), slice(None)), (slice(None), slice(0, 5, 1), slice(None)))
    # Check longs as integer slices
    sn = slice(None)
    assert optimize_read_slicers((1, 2, 3), (2, 3, 4), 4, _always) == ((sn, sn, 3), (1, 2))


def test_slicers2segments():
    # Test function to construct segments from slice objects
    assert slicers2segments((0,), (10,), 7, 4) == [[7, 4]]
    assert slicers2segments((0, 1), (10, 6), 7, 4) == [[7 + 10 * 4, 4]]
    assert slicers2segments((0, 1, 2), (10, 6, 4), 7, 4) == [[7 + 10 * 4 + 10 * 6 * 2 * 4, 4]]
    assert slicers2segments((slice(None),), (10,), 7, 4) == [[7, 10 * 4]]
    assert slicers2segments((0, slice(None)), (10, 6), 7, 4) == [
        [7 + 10 * 4 * i, 4] for i in range(6)
    ]
    assert slicers2segments((slice(None), 0), (10, 6), 7, 4) == [[7, 10 * 4]]
    assert slicers2segments((slice(None), slice(None)), (10, 6), 7, 4) == [[7, 10 * 6 * 4]]
    assert slicers2segments((slice(None), slice(None), 2), (10, 6, 4), 7, 4) == [
        [7 + 10 * 6 * 2 * 4, 10 * 6 * 4]
    ]


def test_calc_slicedefs():
    # Check get_segments routine.  The tests aren't well organized because I
    # wrote them after the code.  We live and (fail to) learn
    segments, out_shape, new_slicing = calc_slicedefs((1,), (10,), 4, 7, 'F', _never)
    assert segments == [[11, 4]]
    assert new_slicing == ()
    assert out_shape == ()
    assert calc_slicedefs((slice(None),), (10,), 4, 7, 'F', _never) == (
        [[7, 40]],
        (10,),
        (),
    )
    assert calc_slicedefs((slice(9),), (10,), 4, 7, 'F', _never) == (
        [[7, 36]],
        (9,),
        (),
    )
    assert calc_slicedefs((slice(1, 9),), (10,), 4, 7, 'F', _never) == (
        [[11, 32]],
        (8,),
        (),
    )
    # Two dimensions, single slice
    assert calc_slicedefs((0,), (10, 6), 4, 7, 'F', _never) == (
        [[7, 4], [47, 4], [87, 4], [127, 4], [167, 4], [207, 4]],
        (6,),
        (),
    )
    assert calc_slicedefs((0,), (10, 6), 4, 7, 'C', _never) == (
        [[7, 6 * 4]],
        (6,),
        (),
    )
    # Two dimensions, contiguous not full
    assert calc_slicedefs((1, slice(1, 5)), (10, 6), 4, 7, 'F', _never) == (
        [[51, 4], [91, 4], [131, 4], [171, 4]],
        (4,),
        (),
    )
    assert calc_slicedefs((1, slice(1, 5)), (10, 6), 4, 7, 'C', _never) == (
        [[7 + 7 * 4, 16]],
        (4,),
        (),
    )
    # With full slice first
    assert calc_slicedefs((slice(None), slice(1, 5)), (10, 6), 4, 7, 'F', _never) == (
        [[47, 160]],
        (10, 4),
        (),
    )
    # Check effect of heuristic on calc_slicedefs
    # Even integer slices can generate full when heuristic says so
    assert calc_slicedefs((1, slice(None)), (10, 6), 4, 7, 'F', _always) == (
        [[7, 10 * 6 * 4]],
        (10, 6),
        (1, slice(None)),
    )
    # Except when last
    assert calc_slicedefs((slice(None), 1), (10, 6), 4, 7, 'F', _always) == (
        [[7 + 10 * 4, 10 * 4]],
        (10,),
        (),
    )


def test_predict_shape():
    shapes = (15, 16, 17, 18)
    for n_dim in range(len(shapes)):
        shape = shapes[: n_dim + 1]
        arr = np.arange(np.prod(shape)).reshape(shape)
        slicers_list = []
        for i in range(n_dim):
            slicers_list.append(_slices_for_len(shape[i]))
            for sliceobj in product(*slicers_list):
                assert predict_shape(sliceobj, shape) == arr[sliceobj].shape
    # Try some Nones and ellipses
    assert predict_shape((Ellipsis,), (2, 3)) == (2, 3)
    assert predict_shape((Ellipsis, 1), (2, 3)) == (2,)
    assert predict_shape((1, Ellipsis), (2, 3)) == (3,)
    assert predict_shape((1, slice(None), Ellipsis), (2, 3)) == (3,)
    assert predict_shape((None,), (2, 3)) == (1, 2, 3)
    assert predict_shape((None, 1), (2, 3)) == (1, 3)
    assert predict_shape((1, None, slice(None)), (2, 3)) == (1, 3)
    assert predict_shape((1, slice(None), None), (2, 3)) == (3, 1)


def test_strided_scalar():
    # Utility to make numpy array of given shape from scalar using striding
    for shape, scalar in product(
        ((2,), (2, 3), (2, 3, 4)),
        (1, 2, np.int16(3)),
    ):
        expected = np.zeros(shape, dtype=np.array(scalar).dtype) + scalar
        observed = strided_scalar(shape, scalar)
        assert_array_equal(observed, expected)
        assert observed.shape == shape
        assert observed.dtype == expected.dtype
        assert_array_equal(observed.strides, 0)
        # Strided scalars are set as not writeable
        # This addresses a numpy 1.10 breakage of broadcasting a strided
        # array without resizing (see GitHub PR #358)
        assert not observed.flags.writeable

        def setval(x):
            x[..., 0] = 99

        # RuntimeError for numpy < 1.10
        with pytest.raises((RuntimeError, ValueError)):
            setval(observed)
    # Default scalar value is 0
    assert_array_equal(strided_scalar((2, 3, 4)), np.zeros((2, 3, 4)))


def _check_bytes(bytes, arr):
    barr = np.ndarray(arr.shape, arr.dtype, buffer=bytes)
    assert_array_equal(barr, arr)


def test_read_segments():
    # Test segment reading
    fobj = BytesIO()
    arr = np.arange(100, dtype=np.int16)
    fobj.write(arr.tobytes())
    _check_bytes(read_segments(fobj, [(0, 200)], 200), arr)
    _check_bytes(read_segments(fobj, [(0, 100), (100, 100)], 200), arr)
    _check_bytes(read_segments(fobj, [(0, 50), (100, 50)], 100), np.r_[arr[:25], arr[50:75]])
    _check_bytes(read_segments(fobj, [(10, 40), (100, 50)], 90), np.r_[arr[5:25], arr[50:75]])
    _check_bytes(read_segments(fobj, [], 0), arr[0:0])
    # Error conditions
    with pytest.raises(ValueError):
        read_segments(fobj, [], 1)
    with pytest.raises(ValueError):
        read_segments(fobj, [(0, 200)], 199)
    with pytest.raises(Exception):
        read_segments(fobj, [(0, 100), (100, 200)], 199)


def test_read_segments_lock():
    # Test read_segment locking with multiple threads
    fobj = BytesIO()
    arr = np.array(np.random.randint(0, 256, 1000), dtype=np.uint8)
    fobj.write(arr.tobytes())

    # Encourage the interpreter to switch threads between a seek/read pair
    def yielding_read(*args, **kwargs):
        time.sleep(0.001)
        return fobj._real_read(*args, **kwargs)

    fobj._real_read = fobj.read
    fobj.read = yielding_read

    # Generate some random array segments to read from the file
    def random_segments(nsegs):
        segs = []
        nbytes = 0

        for i in range(nsegs):
            seglo = np.random.randint(0, 998)
            seghi = np.random.randint(seglo + 1, 1000)
            seglen = seghi - seglo
            nbytes += seglen
            segs.append([seglo, seglen])

        return segs, nbytes

    # Get the data that should be returned for the given segments
    def get_expected(segs):
        segs = [arr[off : off + length] for off, length in segs]
        return np.concatenate(segs)

    # Read from the file, check the result. We do this task simultaneously in
    # many threads. Each thread that passes adds 1 to numpassed[0]
    numpassed = [0]
    lock = Lock()

    def runtest():
        seg, nbytes = random_segments(1)
        expected = get_expected(seg)
        _check_bytes(read_segments(fobj, seg, nbytes, lock), expected)

        seg, nbytes = random_segments(10)
        expected = get_expected(seg)
        _check_bytes(read_segments(fobj, seg, nbytes, lock), expected)

        with lock:
            numpassed[0] += 1

    threads = [Thread(target=runtest) for i in range(100)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert numpassed[0] == len(threads)


def _check_slicer(sliceobj, arr, fobj, offset, order, heuristic=threshold_heuristic):
    new_slice = fileslice(fobj, sliceobj, arr.shape, arr.dtype, offset, order, heuristic)
    assert_array_equal(arr[sliceobj], new_slice)


def slicer_samples(shape):
    """Generator returns slice samples for given `shape`"""
    ndim = len(shape)
    slicers_list = []
    for i in range(ndim):
        slicers_list.append(_slices_for_len(shape[i]))
        yield from product(*slicers_list)
    # Nones and ellipses
    yield (None,)
    if ndim == 0:
        return
    yield (None, 0)
    yield (None, np.array(0))
    yield (0, None)
    yield (np.array(0), None)
    yield (Ellipsis, -1)
    yield (Ellipsis, np.array(-1))
    yield (-1, Ellipsis)
    yield (np.array(-1), Ellipsis)
    yield (None, Ellipsis)
    yield (Ellipsis, None)
    yield (Ellipsis, None, None)
    if ndim == 1:
        return
    yield (0, None, slice(None))
    yield (np.array(0), None, slice(None))
    yield (Ellipsis, -1, None)
    yield (Ellipsis, np.array(-1), None)
    yield (0, Ellipsis, None)
    yield (np.array(0), Ellipsis, None)
    if ndim == 2:
        return
    yield (slice(None), 0, -1, None)
    yield (slice(None), np.array(0), np.array(-1), None)
    yield (np.array(0), slice(None), np.array(-1), None)


def test_fileslice():
    shapes = (15, 16, 17)
    for n_dim in range(1, len(shapes) + 1):
        shape = shapes[:n_dim]
        arr = np.arange(np.prod(shape)).reshape(shape)
        for order in 'FC':
            for offset in (0, 20):
                fobj = BytesIO()
                fobj.write(b'\0' * offset)
                fobj.write(arr.tobytes(order=order))
                for sliceobj in slicer_samples(shape):
                    _check_slicer(sliceobj, arr, fobj, offset, order)


def test_fileslice_dtype():
    # Test that any valid dtype specifier works for fileslice
    sliceobj = (slice(None), slice(2))
    for dt in (np.dtype('int32'), np.int32, 'i4', 'int32', '>i4', '<i4'):
        arr = np.arange(24, dtype=dt).reshape((2, 3, 4))
        fobj = BytesIO(arr.tobytes())
        new_slice = fileslice(fobj, sliceobj, arr.shape, dt)
        assert_array_equal(arr[sliceobj], new_slice)


def test_fileslice_errors():
    # Test fileslice causes error on fancy indexing
    arr = np.arange(24).reshape((2, 3, 4))
    fobj = BytesIO(arr.tobytes())
    _check_slicer((1,), arr, fobj, 0, 'C')
    # Fancy indexing raises error
    with pytest.raises(ValueError):
        fileslice(fobj, (np.array([1]),), (2, 3, 4), arr.dtype)


def test_fileslice_heuristic():
    # Just check that any of several heuristics gives the right answer
    shape = (15, 16, 17)
    arr = np.arange(np.prod(shape)).reshape(shape)
    for heuristic in (_always, _never, _partial, threshold_heuristic):
        for order in 'FC':
            fobj = BytesIO()
            fobj.write(arr.tobytes(order=order))
            sliceobj = (1, slice(0, 15, 2), slice(None))
            _check_slicer(sliceobj, arr, fobj, 0, order, heuristic)
            # Check _simple_fileslice while we're at it - si como no?
            new_slice = _simple_fileslice(
                fobj, sliceobj, arr.shape, arr.dtype, 0, order, heuristic
            )
            assert_array_equal(arr[sliceobj], new_slice)
