"""Utilities for casting numpy values in various ways

Most routines work round some numpy oddities in floating point precision and
casting.  Others work round numpy casting to and from python ints
"""

from __future__ import annotations

import warnings
from platform import machine, processor

import numpy as np

from .deprecated import deprecate_with_version


class CastingError(Exception):
    pass


# Test for VC truncation when casting floats to uint64
# Christoph Gohlke says this is so for MSVC <= 2010 because VC is using x87
# instructions; see:
# https://github.com/scipy/scipy/blob/99bb8411f6391d921cb3f4e56619291e91ddf43b/scipy/ndimage/tests/test_datatypes.py#L51
_test_val = 2**63 + 2**11  # Should be exactly representable in float64
TRUNC_UINT64 = np.float64(_test_val).astype(np.uint64) != _test_val

# np.sctypes is deprecated in numpy 2.0 and np.core.sctypes should not be used instead.
sctypes = {
    'int': [
        getattr(np, dtype) for dtype in ('int8', 'int16', 'int32', 'int64') if hasattr(np, dtype)
    ],
    'uint': [
        getattr(np, dtype)
        for dtype in ('uint8', 'uint16', 'uint32', 'uint64')
        if hasattr(np, dtype)
    ],
    'float': [
        getattr(np, dtype)
        for dtype in ('float16', 'float32', 'float64', 'float96', 'float128')
        if hasattr(np, dtype)
    ],
    'complex': [
        getattr(np, dtype)
        for dtype in ('complex64', 'complex128', 'complex192', 'complex256')
        if hasattr(np, dtype)
    ],
    'others': [bool, object, bytes, str, np.void],
}
sctypes_aliases = {
    getattr(np, dtype)
    for dtype in (
        'int8', 'byte', 'int16', 'short', 'int32', 'intc', 'int_', 'int64', 'longlong',
        'uint8', 'ubyte', 'uint16', 'ushort', 'uint32', 'uintc', 'uint', 'uint64', 'ulonglong',
        'float16', 'half', 'float32', 'single', 'float64', 'double', 'float96', 'float128', 'longdouble',
        'complex64', 'csingle', 'complex128', 'cdouble', 'complex192', 'complex256', 'clongdouble',
        # other names of the built-in scalar types
        'int_', 'float_', 'complex_', 'bytes_', 'str_', 'bool_', 'datetime64', 'timedelta64',
        # other
        'object_', 'void',
    )
    if hasattr(np, dtype)
}  # fmt:skip


def float_to_int(arr, int_type, nan2zero=True, infmax=False):
    """Convert floating point array `arr` to type `int_type`

    * Rounds numbers to nearest integer
    * Clips values to prevent overflows when casting
    * Converts NaN to 0 (for `nan2zero` == True)

    Casting floats to integers is delicate because the result is undefined
    and platform specific for float values outside the range of `int_type`.
    Define ``shared_min`` to be the minimum value that can be exactly
    represented in both the float type of `arr` and `int_type`. Define
    `shared_max` to be the equivalent maximum value.  To avoid undefined
    results we threshold `arr` at ``shared_min`` and ``shared_max``.

    Parameters
    ----------
    arr : array-like
        Array of floating point type
    int_type : object
        Numpy integer type
    nan2zero : {True, False, None}
        Whether to convert NaN value to zero.  Default is True.  If False, and
        NaNs are present, raise CastingError. If None, do not check for NaN
        values and pass through directly to the ``astype`` casting mechanism.
        In this last case, the resulting value is undefined.
    infmax : {False, True}
        If True, set np.inf values in `arr` to be `int_type` integer maximum
        value, -np.inf as `int_type` integer minimum.  If False, set +/- infs
        to be ``shared_min``, ``shared_max`` as defined above.  Therefore False
        gives faster conversion at the expense of infs that are further from
        infinity.

    Returns
    -------
    iarr : ndarray
        of type `int_type`

    Examples
    --------
    >>> float_to_int([np.nan, np.inf, -np.inf, 1.1, 6.6], np.int16)
    array([     0,  32767, -32768,      1,      7], dtype=int16)

    Notes
    -----
    Numpy relies on the C library to cast from float to int using the standard
    ``astype`` method of the array.

    Quoting from section F4 of the C99 standard:

        If the floating value is infinite or NaN or if the integral part of the
        floating value exceeds the range of the integer type, then the
        "invalid" floating-point exception is raised and the resulting value
        is unspecified.

    Hence we threshold at ``shared_min`` and ``shared_max`` to avoid casting to
    values that are undefined.

    See: https://en.wikipedia.org/wiki/C99 . There are links to the C99
    standard from that page.
    """
    arr = np.asarray(arr)
    flt_type = arr.dtype.type
    int_type = np.dtype(int_type).type
    # Deal with scalar as input; fancy indexing needs 1D
    shape = arr.shape
    arr = np.atleast_1d(arr)
    mn, mx = shared_range(flt_type, int_type)
    if nan2zero is None:
        seen_nans = False
    else:
        nans = np.isnan(arr)
        seen_nans = np.any(nans)
        if not nan2zero and seen_nans:
            raise CastingError('NaNs in array, nan2zero is False')
    iarr = np.clip(np.rint(arr), mn, mx).astype(int_type)
    if seen_nans:
        iarr[nans] = 0
    if not infmax:
        return iarr.reshape(shape)
    ii = np.iinfo(int_type)
    iarr[arr == np.inf] = ii.max
    if ii.min != int(mn):
        iarr[arr == -np.inf] = ii.min
    return iarr.reshape(shape)


# Cache range values
_SHARED_RANGES: dict[tuple[type, type], tuple[np.number, np.number]] = {}


def shared_range(flt_type, int_type):
    """Min and max in float type that are >=min, <=max in integer type

    This is not as easy as it sounds, because the float type may not be able to
    exactly represent the max or min integer values, so we have to find the
    next exactly representable floating point value to do the thresholding.

    Parameters
    ----------
    flt_type : dtype specifier
        A dtype specifier referring to a numpy floating point type.  For
        example, ``f4``, ``np.dtype('f4')``, ``np.float32`` are equivalent.
    int_type : dtype specifier
        A dtype specifier referring to a numpy integer type.  For example,
        ``i4``, ``np.dtype('i4')``, ``np.int32`` are equivalent

    Returns
    -------
    mn : object
        Number of type `flt_type` that is the minimum value in the range of
        `int_type`, such that ``mn.astype(int_type)`` >= min of `int_type`
    mx : object
        Number of type `flt_type` that is the maximum value in the range of
        `int_type`, such that ``mx.astype(int_type)`` <= max of `int_type`

    Examples
    --------
    >>> shared_range(np.float32, np.int32) == (-2147483648.0, 2147483520.0)
    True
    >>> shared_range('f4', 'i4') == (-2147483648.0, 2147483520.0)
    True
    """
    flt_type = np.dtype(flt_type).type
    int_type = np.dtype(int_type).type
    key = (flt_type, int_type)
    # Used cached value if present
    try:
        return _SHARED_RANGES[key]
    except KeyError:
        pass
    ii = np.iinfo(int_type)
    fi = np.finfo(flt_type)
    mn = ceil_exact(ii.min, flt_type)
    if mn == -np.inf:
        mn = fi.min
    mx = floor_exact(ii.max, flt_type)
    if mx == np.inf:
        mx = fi.max
    elif TRUNC_UINT64 and int_type == np.uint64:
        mx = min(mx, flt_type(2**63))
    _SHARED_RANGES[key] = (mn, mx)
    return mn, mx


# ----------------------------------------------------------------------------
# Routines to work out the next lowest representable integer in floating point
# types.
# ----------------------------------------------------------------------------


class FloatingError(Exception):
    pass


def on_powerpc():
    """True if we are running on a Power PC platform

    Has to deal with older Macs and IBM POWER7 series among others
    """
    return processor() == 'powerpc' or machine().startswith('ppc')


def type_info(np_type):
    """Return dict with min, max, nexp, nmant, width for numpy type `np_type`

    Type can be integer in which case nexp and nmant are None.

    Parameters
    ----------
    np_type : numpy type specifier
        Any specifier for a numpy dtype

    Returns
    -------
    info : dict
        with fields ``min`` (minimum value), ``max`` (maximum value), ``nexp``
        (exponent width), ``nmant`` (significand precision not including
        implicit first digit), ``minexp`` (minimum exponent), ``maxexp``
        (maximum exponent), ``width`` (width in bytes). (``nexp``, ``nmant``,
        ``minexp``, ``maxexp``) are None for integer types. Both ``min`` and
        ``max`` are of type `np_type`.

    Raises
    ------
    FloatingError
        for floating point types we don't recognize

    Notes
    -----
    You might be thinking that ``np.finfo`` does this job, and it does, except
    for PPC long doubles (https://github.com/numpy/numpy/issues/2669) and
    float96 on Windows compiled with Mingw. This routine protects against such
    errors in ``np.finfo`` by only accepting values that we know are likely to
    be correct.
    """
    dt = np.dtype(np_type)
    np_type = dt.type
    width = dt.itemsize
    try:  # integer type
        info = np.iinfo(dt)
    except ValueError:
        pass
    else:
        return dict(
            min=np_type(info.min),
            max=np_type(info.max),
            minexp=None,
            maxexp=None,
            nmant=None,
            nexp=None,
            width=width,
        )
    # Mitigate warning from WSL1 when checking `np.longdouble` (#1309)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore', category=UserWarning, message='Signature.*numpy.longdouble'
        )
        info = np.finfo(dt)

    # Trust the standard IEEE types
    nmant, nexp = info.nmant, info.nexp
    ret = dict(
        min=np_type(info.min),
        max=np_type(info.max),
        nmant=nmant,
        nexp=nexp,
        minexp=info.minexp,
        maxexp=info.maxexp,
        width=width,
    )
    if np_type in (np.float16, np.float32, np.float64, np.complex64, np.complex128):
        return ret
    info_64 = np.finfo(np.float64)
    if dt.kind == 'c':
        assert np_type is np.clongdouble
        vals = (nmant, nexp, width / 2)
    else:
        assert np_type is np.longdouble
        vals = (nmant, nexp, width)
    if vals in (
        (112, 15, 16),  # binary128
        (info_64.nmant, info_64.nexp, 8),  # float64
        (63, 15, 12),  # Intel extended 80
        (63, 15, 16),  # Intel extended 80
    ):
        return ret  # these are OK without modification
    # The remaining types are longdoubles with bad finfo values.  Some we
    # correct, others we wait to hear of errors.
    # We start with float64 as basis
    ret = type_info(np.float64)
    if vals in ((52, 15, 12), (52, 15, 16)):  # windows float96 / windows float128?
        # On windows 32 bit at least, float96 is Intel 80 storage but operating
        # at float64 precision. The finfo values give nexp == 15 (as for intel
        # 80) but in calculations nexp in fact appears to be 11 as for float64
        ret.update(dict(width=width))
        return ret
    if vals == (105, 11, 16):  # correctly detected double double
        ret.update(dict(nmant=nmant, nexp=nexp, width=width))
        return ret
    # Oh dear, we don't recognize the type information.  Try some known types
    # and then give up. At this stage we're expecting exotic longdouble or
    # their complex equivalent.
    if np_type not in (np.longdouble, np.clongdouble) or width not in (16, 32):
        raise FloatingError(f'We had not expected type {np_type}')
    if vals == (1, 1, 16) and on_powerpc() and _check_maxexp(np.longdouble, 1024):
        # double pair on PPC.  The _check_nmant routine does not work for this
        # type, hence the powerpc platform check instead
        ret.update(dict(nmant=106, width=width))
    elif _check_nmant(np.longdouble, 52) and _check_maxexp(np.longdouble, 11):
        # Got float64 despite everything
        pass
    elif _check_nmant(np.longdouble, 112) and _check_maxexp(np.longdouble, 16384):
        # binary 128, but with some busted type information. np.clongdouble
        # seems to break here too, so we need to use np.longdouble and
        # complexify
        two = np.longdouble(2)
        # See: https://matthew-brett.github.io/pydagogue/floating_point.html
        max_val = (two**113 - 1) / (two**112) * two**16383
        if np_type is np.clongdouble:
            max_val += 0j
        ret = dict(
            min=-max_val,
            max=max_val,
            nmant=112,
            nexp=15,
            minexp=-16382,
            maxexp=16384,
            width=width,
        )
    else:  # don't recognize the type
        raise FloatingError(f'We had not expected long double type {np_type} with info {info}')
    return ret


def _check_nmant(np_type, nmant):
    """True if fp type `np_type` seems to have `nmant` significand digits

    Note 'digits' does not include implicit digits.  And in fact if there are
    no implicit digits, the `nmant` number is one less than the actual digits.
    Assumes base 2 representation.

    Parameters
    ----------
    np_type : numpy type specifier
        Any specifier for a numpy dtype
    nmant : int
        Number of digits to test against

    Returns
    -------
    tf : bool
        True if `nmant` is the correct number of significand digits, false
        otherwise
    """
    np_type = np.dtype(np_type).type
    max_contig = np_type(2 ** (nmant + 1))  # maximum of contiguous integers
    tests = max_contig + np.array([-2, -1, 0, 1, 2], dtype=np_type)
    return np.all(tests - max_contig == [-2, -1, 0, 0, 2])


def _check_maxexp(np_type, maxexp):
    """True if fp type `np_type` seems to have `maxexp` maximum exponent

    We're testing "maxexp" as returned by numpy. This value is set to one
    greater than the maximum power of 2 that `np_type` can represent.

    Assumes base 2 representation.  Very crude check

    Parameters
    ----------
    np_type : numpy type specifier
        Any specifier for a numpy dtype
    maxexp : int
        Maximum exponent to test against

    Returns
    -------
    tf : bool
        True if `maxexp` is the correct maximum exponent, False otherwise.
    """
    dt = np.dtype(np_type)
    np_type = dt.type
    two = np_type(2).reshape((1,))  # to avoid upcasting
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # Expected overflow warning
        return np.isfinite(two ** (maxexp - 1)) and not np.isfinite(two**maxexp)


@deprecate_with_version('as_int() is deprecated. Use int() instead.', '5.2.0', '7.0.0')
def as_int(x, check=True):
    """Return python integer representation of number

    This is useful because the numpy int(val) mechanism is broken for large
    values in np.longdouble.

    It is also useful to work around a numpy 1.4.1 bug in conversion of uints
    to python ints.

    Parameters
    ----------
    x : object
        integer, unsigned integer or floating point value
    check : {True, False}
        If True, raise error for values that are not integers

    Returns
    -------
    i : int
        Python integer

    Examples
    --------
    >>> as_int(2.0)
    2
    >>> as_int(-2.0)
    -2
    >>> as_int(2.1) #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    FloatingError: Not an integer: 2.1
    >>> as_int(2.1, check=False)
    2
    """
    ix = int(x)
    if check and ix != x:
        raise FloatingError(f'Not an integer: {x}')
    return ix


@deprecate_with_version('int_to_float(..., dt) is deprecated. Use dt() instead.', '5.2.0', '7.0.0')
def int_to_float(val, flt_type):
    """Convert integer `val` to floating point type `flt_type`

    Why is this so complicated?

    At least in numpy <= 1.6.1, numpy longdoubles do not correctly convert to
    ints, and ints do not correctly convert to longdoubles.  Specifically, in
    both cases, the values seem to go through float64 conversion on the way, so
    to convert better, we need to split into float64s and sum up the result.

    Parameters
    ----------
    val : int
        Integer value
    flt_type : object
        numpy floating point type

    Returns
    -------
    f : numpy scalar
        of type `flt_type`

    Examples
    --------
    >>> int_to_float(1, np.float32)
    1.0
    """
    return flt_type(val)


def floor_exact(val, flt_type):
    """Return nearest exact integer <= `val` in float type `flt_type`

    Parameters
    ----------
    val : int
        We have to pass val as an int rather than the floating point type
        because large integers cast as floating point may be rounded by the
        casting process.
    flt_type : numpy type
        numpy float type.

    Returns
    -------
    floor_val : object
        value of same floating point type as `val`, that is the nearest exact
        integer in this type such that `floor_val` <= `val`.  Thus if `val` is
        exact in `flt_type`, `floor_val` == `val`.

    Examples
    --------
    Obviously 2 is within the range of representable integers for float32

    >>> floor_exact(2, np.float32)
    2.0

    As is 2**24-1 (the number of significand digits is 23 + 1 implicit)

    >>> floor_exact(2**24-1, np.float32) == 2**24-1
    True

    But 2**24+1 gives a number that float32 can't represent exactly

    >>> floor_exact(2**24+1, np.float32) == 2**24
    True

    As for the numpy floor function, negatives floor towards -inf

    >>> floor_exact(-2**24-1, np.float32) == -2**24-2
    True
    """
    val = int(val)
    flt_type = np.dtype(flt_type).type
    sign = 1 if val > 0 else -1
    try:
        fval = flt_type(val)
    except OverflowError:
        return sign * np.inf
    if not np.isfinite(fval):
        return fval
    info = type_info(flt_type)
    diff = val - int(fval)
    if diff >= 0:  # floating point value <= val
        return fval
    # Float casting made the value go up
    biggest_gap = 2 ** (floor_log2(val) - info['nmant'])
    assert biggest_gap > 1
    fval -= flt_type(biggest_gap)
    return fval


def ceil_exact(val, flt_type):
    """Return nearest exact integer >= `val` in float type `flt_type`

    Parameters
    ----------
    val : int
        We have to pass val as an int rather than the floating point type
        because large integers cast as floating point may be rounded by the
        casting process.
    flt_type : numpy type
        numpy float type.

    Returns
    -------
    ceil_val : object
        value of same floating point type as `val`, that is the nearest exact
        integer in this type such that `floor_val` >= `val`.  Thus if `val` is
        exact in `flt_type`, `ceil_val` == `val`.

    Examples
    --------
    Obviously 2 is within the range of representable integers for float32

    >>> ceil_exact(2, np.float32)
    2.0

    As is 2**24-1 (the number of significand digits is 23 + 1 implicit)

    >>> ceil_exact(2**24-1, np.float32) == 2**24-1
    True

    But 2**24+1 gives a number that float32 can't represent exactly

    >>> ceil_exact(2**24+1, np.float32) == 2**24+2
    True

    As for the numpy ceil function, negatives ceil towards inf

    >>> ceil_exact(-2**24-1, np.float32) == -2**24
    True
    """
    return -floor_exact(-val, flt_type)


def int_abs(arr):
    """Absolute values of array taking care of max negative int values

    Parameters
    ----------
    arr : array-like

    Returns
    -------
    abs_arr : array
        array the same shape as `arr` in which all negative numbers have been
        changed to positive numbers with the magnitude.

    Examples
    --------
    This kind of thing is confusing in base numpy:

    >>> import numpy as np
    >>> np.abs(np.int8(-128))
    -128

    ``int_abs`` fixes that:

    >>> int_abs(np.int8(-128))
    128
    >>> int_abs(np.array([-128, 127], dtype=np.int8))
    array([128, 127], dtype=uint8)
    >>> int_abs(np.array([-128, 127], dtype=np.float32))
    array([128., 127.], dtype=float32)
    """
    arr = np.asarray(arr)
    dt = arr.dtype
    if dt.kind == 'u':
        return arr
    if dt.kind != 'i':
        return np.absolute(arr)
    out = arr.astype(np.dtype(dt.str.replace('i', 'u')))
    return np.choose(arr < 0, (arr, arr * -1), out=out)


def floor_log2(x):
    """floor of log2 of abs(`x`)

    Embarrassingly, from https://en.wikipedia.org/wiki/Binary_logarithm

    Parameters
    ----------
    x : int

    Returns
    -------
    L : None or int
        floor of base 2 log of `x`.  None if `x` == 0.

    Examples
    --------
    >>> floor_log2(2**9+1)
    9
    >>> floor_log2(-2**9+1)
    8
    >>> floor_log2(0.5)
    -1
    >>> floor_log2(0) is None
    True
    """
    ip = 0
    rem = abs(x)
    if rem > 1:
        while rem >= 2:
            ip += 1
            rem //= 2
        return ip
    elif rem == 0:
        return None
    while rem < 1:
        ip -= 1
        rem *= 2
    return ip


def best_float():
    """Floating point type with best precision

    This is nearly always np.longdouble, except on Windows, where np.longdouble
    is Intel80 storage, but with float64 precision for calculations.  In that
    case we return float64 on the basis it's the fastest and smallest at the
    highest precision.

    SPARC float128 also proved so slow that we prefer float64.

    Returns
    -------
    best_type : numpy type
        floating point type with highest precision

    Notes
    -----
    Needs to run without error for module import, because it is called in
    ``ok_floats`` below, and therefore in setting module global ``OK_FLOATS``.
    """
    try:
        long_info = type_info(np.longdouble)
    except FloatingError:
        return np.float64
    if (
        long_info['nmant'] > type_info(np.float64)['nmant'] and machine() != 'sparc64'
    ):  # sparc has crazy-slow float128
        return np.longdouble
    return np.float64


def longdouble_lte_float64():
    """Return True if longdouble appears to have the same precision as float64"""
    return np.longdouble(2**53) == np.longdouble(2**53) + 1


# Record longdouble precision at import because it can change on Windows
_LD_LTE_FLOAT64 = longdouble_lte_float64()


def longdouble_precision_improved():
    """True if longdouble precision increased since initial import

    This can happen on Windows compiled with MSVC.  It may be because libraries
    compiled with mingw (longdouble is Intel80) get linked to numpy compiled
    with MSVC (longdouble is Float64)
    """
    return not longdouble_lte_float64() and _LD_LTE_FLOAT64


def have_binary128():
    """True if we have a binary128 IEEE longdouble"""
    try:
        ti = type_info(np.longdouble)
    except FloatingError:
        return False
    return (ti['nmant'], ti['maxexp']) == (112, 16384)


def ok_floats():
    """Return floating point types sorted by precision

    Remove longdouble if it has no higher precision than float64
    """
    # copy float list so we don't change the numpy global
    floats = sctypes['float'][:]
    if best_float() != np.longdouble and np.longdouble in floats:
        floats.remove(np.longdouble)
    return sorted(floats, key=lambda f: type_info(f)['nmant'])


OK_FLOATS = ok_floats()


def able_int_type(values):
    """Find the smallest integer numpy type to contain sequence `values`

    Prefers uint to int if minimum is >= 0

    Parameters
    ----------
    values : sequence
        sequence of integer values

    Returns
    -------
    itype : None or numpy type
        numpy integer type or None if no integer type holds all `values`

    Examples
    --------
    >>> able_int_type([0, 1]) == np.uint8
    True
    >>> able_int_type([-1, 1]) == np.int8
    True
    """
    if any(v % 1 for v in values):
        return None
    mn = min(values)
    mx = max(values)
    if mn >= 0:
        for ityp in sctypes['uint']:
            if mx <= np.iinfo(ityp).max:
                return ityp
    for ityp in sctypes['int']:
        info = np.iinfo(ityp)
        if mn >= info.min and mx <= info.max:
            return ityp
    return None


def ulp(val=np.float64(1.0)):
    """Return gap between `val` and nearest representable number of same type

    This is the value of a unit in the last place (ULP), and is similar in
    meaning to the MATLAB eps function.

    Parameters
    ----------
    val : scalar, optional
        scalar value of any numpy type.  Default is 1.0 (float64)

    Returns
    -------
    ulp_val : scalar
        gap between `val` and nearest representable number of same type

    Notes
    -----
    The wikipedia article on machine epsilon points out that the term *epsilon*
    can be used in the sense of a unit in the last place (ULP), or as the
    maximum relative rounding error.  The MATLAB ``eps`` function uses the ULP
    meaning, but this function is ``ulp`` rather than ``eps`` to avoid
    confusion between different meanings of *eps*.
    """
    val = np.array(val)
    if not np.isfinite(val):
        return np.nan
    if val.dtype.kind in 'iu':
        return 1
    aval = np.abs(val)
    info = type_info(val.dtype)
    fl2 = floor_log2(aval)
    if fl2 is None or fl2 < info['minexp']:  # subnormal
        fl2 = info['minexp']
    # 'nmant' value does not include implicit first bit
    return 2 ** (fl2 - info['nmant'])
