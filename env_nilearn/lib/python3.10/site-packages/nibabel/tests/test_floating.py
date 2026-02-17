"""Test floating point deconstructions and floor methods"""

import sys

import numpy as np
from packaging.version import Version

from ..casting import (
    FloatingError,
    _check_maxexp,
    _check_nmant,
    ceil_exact,
    floor_exact,
    have_binary128,
    longdouble_precision_improved,
    ok_floats,
    on_powerpc,
    sctypes,
    type_info,
)
from ..testing import suppress_warnings

IEEE_floats = [np.float16, np.float32, np.float64]

LD_INFO = type_info(np.longdouble)

FP_OVERFLOW_WARN = Version(np.__version__) < Version('2.0.0.dev0')


def dtt2dict(dtt):
    """Create info dictionary from numpy type"""
    info = np.finfo(dtt)
    return dict(
        min=info.min,
        max=info.max,
        nexp=info.nexp,
        nmant=info.nmant,
        minexp=info.minexp,
        maxexp=info.maxexp,
        width=np.dtype(dtt).itemsize,
    )


def test_type_info():
    # Test routine to get min, max, nmant, nexp
    for dtt in sctypes['int'] + sctypes['uint']:
        info = np.iinfo(dtt)
        infod = type_info(dtt)
        assert infod == dict(
            min=info.min,
            max=info.max,
            nexp=None,
            nmant=None,
            minexp=None,
            maxexp=None,
            width=np.dtype(dtt).itemsize,
        )
        assert infod['min'].dtype.type == dtt
        assert infod['max'].dtype.type == dtt
    for dtt in IEEE_floats + [np.complex64, np.complex64]:
        infod = type_info(dtt)
        assert dtt2dict(dtt) == infod
        assert infod['min'].dtype.type == dtt
        assert infod['max'].dtype.type == dtt
    # What is longdouble?
    ld_dict = dtt2dict(np.longdouble)
    dbl_dict = dtt2dict(np.float64)
    infod = type_info(np.longdouble)
    vals = tuple(ld_dict[k] for k in ('nmant', 'nexp', 'width'))
    # Information for PPC head / tail doubles from:
    # https://developer.apple.com/library/mac/#documentation/Darwin/Reference/Manpages/man3/float.3.html
    if vals in (
        (52, 11, 8),  # longdouble is same as double
        (63, 15, 12),  # intel 80 bit
        (63, 15, 16),  # intel 80 bit
        (112, 15, 16),  # real float128
        (106, 11, 16),  # PPC head, tail doubles, expected values
    ):
        pass
    elif vals == (105, 11, 16):  # bust info for PPC head / tail longdoubles
        # min and max broken, copy from infod
        ld_dict.update({k: infod[k] for k in ('min', 'max')})
    elif vals == (1, 1, 16):  # another bust info for PPC head / tail longdoubles
        ld_dict = dbl_dict.copy()
        ld_dict.update(dict(nmant=106, width=16))
    elif vals == (52, 15, 12):
        width = ld_dict['width']
        ld_dict = dbl_dict.copy()
        ld_dict['width'] = width
    else:
        raise ValueError(f'Unexpected float type {np.longdouble} to test')
    assert ld_dict == infod


def test_nmant():
    for t in IEEE_floats:
        assert type_info(t)['nmant'] == np.finfo(t).nmant
    if (LD_INFO['nmant'], LD_INFO['nexp']) == (63, 15):
        assert type_info(np.longdouble)['nmant'] == 63


def test_check_nmant_nexp():
    # Routine for checking number of sigificand digits and exponent
    for t in IEEE_floats:
        nmant = np.finfo(t).nmant
        maxexp = np.finfo(t).maxexp
        assert _check_nmant(t, nmant)
        assert not _check_nmant(t, nmant - 1)
        assert not _check_nmant(t, nmant + 1)
        with suppress_warnings():  # overflow
            assert _check_maxexp(t, maxexp)
        assert not _check_maxexp(t, maxexp - 1)
        with suppress_warnings():
            assert not _check_maxexp(t, maxexp + 1)
    # Check against type_info
    for t in ok_floats():
        ti = type_info(t)
        if ti['nmant'] not in (105, 106):  # This check does not work for PPC double pair
            assert _check_nmant(t, ti['nmant'])
        # Test fails for longdouble after blacklisting of OSX powl as of numpy
        # 1.12 - see https://github.com/numpy/numpy/issues/8307
        if t != np.longdouble or sys.platform != 'darwin':
            assert _check_maxexp(t, ti['maxexp'])


def test_int_longdouble_np_regression():
    # Test longdouble conversion from int works as expected
    # Previous versions of numpy would fail, and we used a custom int_to_float()
    # function. This test remains to ensure we don't need to bring it back.
    nmant = type_info(np.float64)['nmant']
    # test we recover precision just above nmant
    i = 2 ** (nmant + 1) - 1
    assert int(np.longdouble(i)) == i
    assert int(np.longdouble(-i)) == -i
    # If longdouble can cope with 2**64, test
    if nmant >= 63:
        # Check conversion to int; the line below causes an error subtracting
        # ints / uint64 values, at least for Python 3.3 and numpy dev 1.8
        big_int = np.uint64(2**64 - 1)
        assert int(np.longdouble(big_int)) == big_int


def test_int_np_regression():
    # Test int works as expected for integers.
    # We previously used a custom as_int() for integers because of a
    # numpy 1.4.1 bug such that int(np.uint32(2**32-1) == -1
    for t in sctypes['int'] + sctypes['uint']:
        info = np.iinfo(t)
        mn, mx = np.array([info.min, info.max], dtype=t)
        assert (mn, mx) == (int(mn), int(mx))


def test_floor_exact_16():
    # A normal integer can generate an inf in float16
    assert floor_exact(2**31, np.float16) == np.inf
    assert floor_exact(-(2**31), np.float16) == -np.inf


def test_floor_exact_64():
    # float64
    for e in range(53, 63):
        start = np.float64(2**e)
        across = start + np.arange(2048, dtype=np.float64)
        gaps = set(np.diff(across)).difference([0])
        assert len(gaps) == 1
        gap = gaps.pop()
        assert gap == int(gap)
        test_val = 2 ** (e + 1) - 1
        assert floor_exact(test_val, np.float64) == 2 ** (e + 1) - int(gap)


def test_floor_exact(max_digits):
    max_digits(4950)  # max longdouble is ~10**4932

    to_test = IEEE_floats + [float]
    try:
        type_info(np.longdouble)['nmant']
    except FloatingError:
        # Significand bit count not reliable, don't test long double
        pass
    else:
        to_test.append(np.longdouble)
    # When numbers go above int64 - I believe, numpy comparisons break down,
    # so we have to cast to int before comparison
    int_flex = lambda x, t: int(floor_exact(x, t))
    int_ceex = lambda x, t: int(ceil_exact(x, t))
    for t in to_test:
        # A number bigger than the range returns the max
        info = type_info(t)
        assert floor_exact(10**4933, t) == np.inf
        assert ceil_exact(10**4933, t) == np.inf
        # A number more negative returns -inf
        assert floor_exact(-(10**4933), t) == -np.inf
        assert ceil_exact(-(10**4933), t) == -np.inf
        # Check around end of integer precision
        nmant = info['nmant']
        for i in range(nmant + 1):
            iv = 2**i
            # up to 2**nmant should be exactly representable
            for func in (int_flex, int_ceex):
                assert func(iv, t) == iv
                assert func(-iv, t) == -iv
                assert func(iv - 1, t) == iv - 1
                assert func(-iv + 1, t) == -iv + 1
        if t is np.longdouble and (on_powerpc() or longdouble_precision_improved()):
            # The nmant value for longdouble on PPC appears to be conservative,
            # so that the tests for behavior above the nmant range fail.
            # windows longdouble can change from float64 to Intel80 in some
            # situations, in which case nmant will not be correct
            continue
        # Confirm to ourselves that 2**(nmant+1) can't be exactly represented
        iv = 2 ** (nmant + 1)
        assert int_flex(iv + 1, t) == iv
        assert int_ceex(iv + 1, t) == iv + 2
        # negatives
        assert int_flex(-iv - 1, t) == -iv - 2
        assert int_ceex(-iv - 1, t) == -iv
        # The gap in representable numbers is 2 above 2**(nmant+1), 4 above
        # 2**(nmant+2), and so on.
        for i in range(5):
            iv = 2 ** (nmant + 1 + i)
            gap = 2 ** (i + 1)
            assert int(t(iv) + t(gap)) == iv + gap
            for j in range(1, gap):
                assert int_flex(iv + j, t) == iv
                assert int_flex(iv + gap + j, t) == iv + gap
                assert int_ceex(iv + j, t) == iv + gap
                assert int_ceex(iv + gap + j, t) == iv + 2 * gap
            # negatives
            for j in range(1, gap):
                assert int_flex(-iv - j, t) == -iv - gap
                assert int_flex(-iv - gap - j, t) == -iv - 2 * gap
                assert int_ceex(-iv - j, t) == -iv
                assert int_ceex(-iv - gap - j, t) == -iv - gap


def test_usable_binary128():
    # Check for usable binary128
    yes = have_binary128()
    with np.errstate(over='ignore'):
        exp_test = np.longdouble(2) ** 16383
    assert yes == (
        exp_test.dtype.itemsize == 16
        and np.isfinite(exp_test)
        and _check_nmant(np.longdouble, 112)
    )
