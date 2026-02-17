"""Test numerical errors introduced by writing then reading images

Test arrays with a range of numerical values, integer and floating point.
"""

from io import BytesIO

import numpy as np
from numpy.testing import assert_array_equal

from .. import Nifti1Header, Nifti1Image
from ..arraywriters import ScalingError
from ..casting import best_float, sctypes, type_info, ulp
from ..spatialimages import HeaderDataError, supported_np_types

DEBUG = False


def round_trip(arr, out_dtype):
    img = Nifti1Image(arr, np.eye(4), dtype=out_dtype)
    img.file_map['image'].fileobj = BytesIO()
    img.to_file_map()
    back = Nifti1Image.from_file_map(img.file_map)
    # Recover array and calculated scaling from array proxy object
    return back.get_fdata(), back.dataobj.slope, back.dataobj.inter


def check_params(in_arr, in_type, out_type):
    arr = in_arr.astype(in_type)
    # clip infs that can arise from downcasting
    if arr.dtype.kind == 'f':
        info = np.finfo(in_type)
        arr = np.clip(arr, info.min, info.max)
    try:
        arr_dash, slope, inter = round_trip(arr, out_type)
    except (ScalingError, HeaderDataError):
        return arr, None, None, None
    return arr, arr_dash, slope, inter


BFT = best_float()
LOGe2 = np.log(BFT(2))


def big_bad_ulp(arr):
    """Return array of ulp values for values in `arr`

    I haven't thought about whether the vectorized log2 here could lead to
    incorrect rounding; this only needs to be ballpark

    This function might be used in nipy/io/tests/test_image_io.py

    Parameters
    ----------
    arr : array
        floating point array

    Returns
    -------
    ulps : array
        ulp values for each element of arr
    """
    # Assumes array is floating point
    arr = np.asarray(arr)
    info = type_info(arr.dtype)
    working_arr = np.abs(arr.astype(BFT))
    # Log2 for numpy < 1.3
    fl2 = np.zeros_like(working_arr) + info['minexp']
    # Avoid divide by zero error for log of 0
    nzs = working_arr > 0
    fl2[nzs] = np.floor(np.log(working_arr[nzs]) / LOGe2)
    fl2 = np.clip(fl2, info['minexp'], np.inf)
    return 2 ** (fl2 - info['nmant'])


def test_big_bad_ulp():
    for ftype in (np.float32, np.float64):
        ti = type_info(ftype)
        fi = np.finfo(ftype)
        min_ulp = 2 ** (ti['minexp'] - ti['nmant'])
        in_arr = np.zeros((10,), dtype=ftype)
        in_arr = np.array([0, 0, 1, 2, 4, 5, -5, -np.inf, np.inf], dtype=ftype)
        out_arr = [
            min_ulp,
            min_ulp,
            fi.eps,
            fi.eps * 2,
            fi.eps * 4,
            fi.eps * 4,
            fi.eps * 4,
            np.inf,
            np.inf,
        ]
        assert_array_equal(big_bad_ulp(in_arr).astype(ftype), out_arr)


BIG_FLOAT = np.float64


def test_round_trip():
    scaling_type = np.float32
    rng = np.random.RandomState(20111121)
    N = 10000
    sd_10s = range(-20, 51, 5)
    iuint_types = sctypes['int'] + sctypes['uint']
    # Remove types which cannot be set into nifti header datatype
    nifti_supported = supported_np_types(Nifti1Header())
    iuint_types = [t for t in iuint_types if t in nifti_supported]
    f_types = [np.float32, np.float64]
    # Expanding standard deviations
    for sd_10 in sd_10s:
        sd = 10.0**sd_10
        V_in = rng.normal(0, sd, size=(N, 1))
        for in_type in f_types:
            for out_type in iuint_types:
                check_arr(sd_10, V_in, in_type, out_type, scaling_type)
    # Spread integers across range
    for sd in np.linspace(0.05, 0.5, 5):
        for in_type in iuint_types:
            info = np.iinfo(in_type)
            mn, mx = info.min, info.max
            type_range = mx - mn
            center = type_range / 2.0 + mn
            # float(sd) because type_range can be type 'long'
            width = type_range * float(sd)
            V_in = rng.normal(center, width, size=(N, 1))
            for out_type in iuint_types:
                check_arr(sd, V_in, in_type, out_type, scaling_type)


def check_arr(test_id, V_in, in_type, out_type, scaling_type):
    arr, arr_dash, slope, inter = check_params(V_in, in_type, out_type)
    if arr_dash is None:
        # Scaling causes a header or writer error
        return
    nzs = arr != 0  # avoid divide by zero error
    if not np.any(nzs):
        if DEBUG:
            raise ValueError('Array all zero')
        return
    arr = arr[nzs]
    arr_dash_L = arr_dash.astype(BIG_FLOAT)[nzs]
    top = arr - arr_dash_L
    if not np.any(top != 0):
        return
    rel_err = np.abs(top / arr)
    abs_err = np.abs(top)
    if slope == 1:  # integers output, offset only scaling
        if {in_type, out_type} == {np.int64, np.uint64}:
            # Scaling to or from 64 bit ints can go outside range of continuous
            # integers for float64 and thus lose precision; take this into
            # account
            A = arr.astype(float)
            Ai = A - inter
            ulps = [big_bad_ulp(A), big_bad_ulp(Ai)]
            exp_abs_err = np.max(ulps, axis=0)
        else:  # floats can give full precision - no error!
            exp_abs_err = np.zeros_like(abs_err)
        rel_thresh = 0
    else:
        # Error from integer rounding
        inting_err = np.abs(scaling_type(slope) / 2)
        inting_err = inting_err + ulp(inting_err)
        # Error from calculation of inter
        inter_err = ulp(scaling_type(inter))
        # Max abs error from floating point
        with np.errstate(over='ignore'):
            Ai = arr - scaling_type(inter)
        Ais = Ai / scaling_type(slope)
        exp_abs_err = inting_err + inter_err + (big_bad_ulp(Ai) + big_bad_ulp(Ais))
        # Relative scaling error from calculation of slope
        # This threshold needs to be 2 x larger on windows 32 bit and PPC for
        # some reason
        rel_thresh = ulp(scaling_type(1))
    test_vals = (abs_err <= exp_abs_err) | (rel_err <= rel_thresh)
    this_test = np.all(test_vals)
    if DEBUG:
        abs_fails = abs_err > exp_abs_err
        rel_fails = rel_err > rel_thresh
        all_fails = abs_fails & rel_fails
        if np.any(rel_fails):
            abs_mx_e = abs_err[rel_fails].max()
            exp_abs_mx_e = exp_abs_err[rel_fails].max()
        else:
            abs_mx_e = None
            exp_abs_mx_e = None
        if np.any(abs_fails):
            rel_mx_e = rel_err[abs_fails].max()
        else:
            rel_mx_e = None
        print(
            (
                test_id,
                np.dtype(in_type).str,
                np.dtype(out_type).str,
                exp_abs_mx_e,
                abs_mx_e,
                rel_thresh,
                rel_mx_e,
                slope,
                inter,
            )
        )
        # To help debugging failures with --pdb-failure
        np.nonzero(all_fails)
    assert this_test
