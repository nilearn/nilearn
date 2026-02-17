# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import itertools
import unittest
from io import BytesIO

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ..optpkg import optional_package

_, have_scipy, _ = optional_package('scipy')

# Decorator to skip tests requiring save / load if scipy not available for mat
# files
needs_scipy = unittest.skipUnless(have_scipy, 'scipy not available')

from ..casting import sctypes_aliases, shared_range, type_info
from ..spatialimages import HeaderDataError
from ..spm99analyze import HeaderTypeError, Spm99AnalyzeHeader, Spm99AnalyzeImage
from ..testing import (
    assert_allclose_safely,
    bytesio_filemap,
    bytesio_round_trip,
    suppress_warnings,
)
from ..volumeutils import _dt_min_max, apply_read_scaling
from . import test_analyze

# np.core.sctypes values are lists of types with unique sizes
# For testing, we want all concrete classes of a type
# Key on kind, rather than abstract base classes, since timedelta64 is a signedinteger
sctypes = {}
for sctype in sctypes_aliases:
    sctypes.setdefault(np.dtype(sctype).kind, []).append(sctype)

# Sort types to ensure that xdist doesn't complain about test order when we parametrize
FLOAT_TYPES = sorted(sctypes['f'], key=lambda x: x.__name__)
COMPLEX_TYPES = sorted(sctypes['c'], key=lambda x: x.__name__)
INT_TYPES = sorted(sctypes['i'], key=lambda x: x.__name__)
UINT_TYPES = sorted(sctypes['u'], key=lambda x: x.__name__)

# Create combined type lists
CFLOAT_TYPES = FLOAT_TYPES + COMPLEX_TYPES
IUINT_TYPES = INT_TYPES + UINT_TYPES
NUMERIC_TYPES = CFLOAT_TYPES + IUINT_TYPES


class HeaderScalingMixin:
    """Mixin to add scaling tests to header tests

    Needs to be a mixin so nifti tests can use this method without inheriting
    directly from the SPM header tests
    """

    def test_data_scaling(self):
        hdr = self.header_class()
        hdr.set_data_shape((1, 2, 3))
        hdr.set_data_dtype(np.int16)
        S3 = BytesIO()
        data = np.arange(6, dtype=np.float64).reshape((1, 2, 3))
        # This uses scaling
        hdr.data_to_fileobj(data, S3)
        data_back = hdr.data_from_fileobj(S3)
        # almost equal
        assert_array_almost_equal(data, data_back, 4)
        # But not quite
        assert not np.all(data == data_back)
        # This is exactly the same call, just testing it works twice
        data_back2 = hdr.data_from_fileobj(S3)
        assert_array_equal(data_back, data_back2, 4)
        # Rescaling is the default
        hdr.data_to_fileobj(data, S3, rescale=True)
        data_back = hdr.data_from_fileobj(S3)
        assert_array_almost_equal(data, data_back, 4)
        assert not np.all(data == data_back)
        # This doesn't use scaling, and so gets perfect precision
        with np.errstate(invalid='ignore'):
            hdr.data_to_fileobj(data, S3, rescale=False)
        data_back = hdr.data_from_fileobj(S3)
        assert np.all(data == data_back)


class TestSpm99AnalyzeHeader(test_analyze.TestAnalyzeHeader, HeaderScalingMixin):
    header_class = Spm99AnalyzeHeader

    def test_empty(self):
        super().test_empty()
        hdr = self.header_class()
        assert hdr['scl_slope'] == 1

    def test_big_scaling(self):
        # Test that upcasting works for huge scalefactors
        # See tests for apply_read_scaling in test_volumeutils
        hdr = self.header_class()
        hdr.set_data_shape((1, 1, 1))
        hdr.set_data_dtype(np.int16)
        sio = BytesIO()
        dtt = np.float32
        # This will generate a huge scalefactor
        data = np.array([type_info(dtt)['max']], dtype=dtt)[:, None, None]
        hdr.data_to_fileobj(data, sio)
        data_back = hdr.data_from_fileobj(sio)
        assert np.allclose(data, data_back)

    def test_slope_inter(self):
        hdr = self.header_class()
        assert hdr.get_slope_inter() == (1.0, None)
        for in_tup, exp_err, out_tup, raw_slope in (
            ((2.0,), None, (2.0, None), 2.0),
            ((None,), None, (None, None), np.nan),
            ((1.0, None), None, (1.0, None), 1.0),
            # non zero intercept causes error
            ((None, 1.1), HeaderTypeError, (None, None), np.nan),
            ((2.0, 1.1), HeaderTypeError, (None, None), 2.0),
            # null scalings
            ((0.0, None), HeaderDataError, (None, None), 0.0),
            ((np.nan, np.nan), None, (None, None), np.nan),
            ((np.nan, None), None, (None, None), np.nan),
            ((None, np.nan), None, (None, None), np.nan),
            ((np.inf, None), HeaderDataError, (None, None), np.inf),
            ((-np.inf, None), HeaderDataError, (None, None), -np.inf),
            ((None, 0.0), None, (None, None), np.nan),
        ):
            hdr = self.header_class()
            if not exp_err is None:
                with pytest.raises(exp_err):
                    hdr.set_slope_inter(*in_tup)
                # raw set
                if not in_tup[0] is None:
                    hdr['scl_slope'] = in_tup[0]
            else:
                hdr.set_slope_inter(*in_tup)
                assert hdr.get_slope_inter() == out_tup
                # Check set survives through checking
                hdr = Spm99AnalyzeHeader.from_header(hdr, check=True)
                assert hdr.get_slope_inter() == out_tup
            assert_array_equal(hdr['scl_slope'], raw_slope)

    def test_origin_checks(self):
        HC = self.header_class
        # origin
        hdr = HC()
        hdr.data_shape = [1, 1, 1]
        hdr['origin'][0] = 101  # severity 20
        fhdr, message, raiser = self.log_chk(hdr, 20)
        assert fhdr == hdr
        assert (
            message == 'very large origin values '
            'relative to dims; leaving as set, '
            'ignoring for affine'
        )
        pytest.raises(*raiser)
        # diagnose binary block
        dxer = self.header_class.diagnose_binaryblock
        assert dxer(hdr.binaryblock) == 'very large origin values relative to dims'


class ImageScalingMixin:
    # Mixin to add scaling checks to image test class
    # Nifti tests inherits from Analyze tests not Spm Analyze tests.  We need
    # these tests for Nifti scaling, hence the mixin.

    def assert_scaling_equal(self, hdr, slope, inter):
        h_slope, h_inter = self._get_raw_scaling(hdr)
        assert_array_equal(h_slope, slope)
        assert_array_equal(h_inter, inter)

    def assert_scale_me_scaling(self, hdr):
        # Assert that header `hdr` has "scale-me" scaling
        slope, inter = self._get_raw_scaling(hdr)
        if not slope is None:
            assert np.isnan(slope)
        if not inter is None:
            assert np.isnan(inter)

    def _get_raw_scaling(self, hdr):
        return hdr['scl_slope'], None

    def _set_raw_scaling(self, hdr, slope, inter):
        # Brutal set of slope and inter
        hdr['scl_slope'] = slope
        if not inter is None:
            raise ValueError('inter should be None')

    def assert_null_scaling(self, arr, slope, inter):
        # Assert scaling makes no difference to img, load, save
        img_class = self.image_class
        input_hdr = img_class.header_class()
        # Scaling makes no difference to array returned from get_data
        self._set_raw_scaling(input_hdr, slope, inter)
        img = img_class(arr, np.eye(4), input_hdr)
        img_hdr = img.header
        self._set_raw_scaling(input_hdr, slope, inter)
        assert_array_equal(img.get_fdata(), arr)
        # Scaling has no effect on image as written via header (with rescaling
        # turned off).
        fm = bytesio_filemap(img)
        img_fobj = fm['image'].fileobj
        hdr_fobj = img_fobj if not 'header' in fm else fm['header'].fileobj
        img_hdr.write_to(hdr_fobj)
        img_hdr.data_to_fileobj(arr, img_fobj, rescale=False)
        raw_rt_img = img_class.from_file_map(fm)
        assert_array_equal(raw_rt_img.get_fdata(), arr)
        # Scaling makes no difference for image round trip
        fm = bytesio_filemap(img)
        img.to_file_map(fm)
        rt_img = img_class.from_file_map(fm)
        assert_array_equal(rt_img.get_fdata(), arr)

    def test_header_scaling(self):
        # For images that implement scaling, test effect of scaling
        #
        # This tests the affect of creating an image with a header containing
        # the scaling, then writing the image and reading again.  So the
        # scaling can be affected by the processing of the header when creating
        # the image, or by interpretation of the scaling when creating the
        # array.
        #
        # Analyze does not implement any scaling, but this test class is the
        # base class for all Analyze-derived classes, such as NIfTI
        img_class = self.image_class
        hdr_class = img_class.header_class
        if not hdr_class.has_data_slope:
            return
        arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        invalid_slopes = (0, np.nan, np.inf, -np.inf)
        for slope in (1,) + invalid_slopes:
            self.assert_null_scaling(arr, slope, None)
        if not hdr_class.has_data_intercept:
            return
        invalid_inters = (np.nan, np.inf, -np.inf)
        invalid_pairs = tuple(itertools.product(invalid_slopes, invalid_inters))
        bad_slopes_good_inter = tuple(itertools.product(invalid_slopes, (0, 1)))
        good_slope_bad_inters = tuple(itertools.product((1, 2), invalid_inters))
        for slope, inter in invalid_pairs + bad_slopes_good_inter + good_slope_bad_inters:
            self.assert_null_scaling(arr, slope, inter)

    def _check_write_scaling(self, slope, inter, effective_slope, effective_inter):
        # Test that explicit set of slope / inter forces write of data using
        # this slope, inter.  We use this helper function for children of the
        # Analyze header
        img_class = self.image_class
        arr = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
        # We're going to test rounding later
        arr[0, 0, 0] = 0.4
        arr[1, 0, 0] = 0.6
        aff = np.eye(4)
        # Implicit header gives scale-me scaling
        img = img_class(arr, aff)
        self.assert_scale_me_scaling(img.header)
        # Input header scaling reset when creating image
        hdr = img.header
        self._set_raw_scaling(hdr, slope, inter)
        img = img_class(arr, aff)
        self.assert_scale_me_scaling(img.header)
        # Array from image unchanged by scaling
        assert_array_equal(img.get_fdata(), arr)
        # As does round trip
        img_rt = bytesio_round_trip(img)
        self.assert_scale_me_scaling(img_rt.header)
        # Round trip array is not scaled
        assert_array_equal(img_rt.get_fdata(), arr)
        # Explicit scaling causes scaling after round trip
        self._set_raw_scaling(img.header, slope, inter)
        self.assert_scaling_equal(img.header, slope, inter)
        # Array from image unchanged by scaling
        assert_array_equal(img.get_fdata(), arr)
        # But the array scaled after round trip
        img_rt = bytesio_round_trip(img)
        assert_array_equal(
            img_rt.get_fdata(), apply_read_scaling(arr, effective_slope, effective_inter)
        )
        # The scaling set into the array proxy
        do_slope, do_inter = img.header.get_slope_inter()
        assert_array_equal(img_rt.dataobj.slope, 1 if do_slope is None else do_slope)
        assert_array_equal(img_rt.dataobj.inter, 0 if do_inter is None else do_inter)
        # The new header scaling has been reset
        self.assert_scale_me_scaling(img_rt.header)
        # But the original is the same as it was when we set it
        self.assert_scaling_equal(img.header, slope, inter)
        # The data gets rounded nicely if we need to do conversion
        img.header.set_data_dtype(np.uint8)
        with np.errstate(invalid='ignore'):
            img_rt = bytesio_round_trip(img)
        assert_array_equal(
            img_rt.get_fdata(), apply_read_scaling(np.round(arr), effective_slope, effective_inter)
        )
        # But we have to clip too
        arr[-1, -1, -1] = 256
        arr[-2, -1, -1] = -1
        with np.errstate(invalid='ignore'):
            img_rt = bytesio_round_trip(img)
        exp_unscaled_arr = np.clip(np.round(arr), 0, 255)
        assert_array_equal(
            img_rt.get_fdata(),
            apply_read_scaling(exp_unscaled_arr, effective_slope, effective_inter),
        )

    def test_int_int_scaling(self):
        # Check int to int conversion without slope, inter
        img_class = self.image_class
        arr = np.array([-1, 0, 256], dtype=np.int16)[:, None, None]
        img = img_class(arr, np.eye(4))
        hdr = img.header
        img.set_data_dtype(np.uint8)
        self._set_raw_scaling(hdr, 1, 0 if hdr.has_data_intercept else None)
        img_rt = bytesio_round_trip(img)
        assert_array_equal(img_rt.get_fdata(), np.clip(arr, 0, 255))

    # NOTE: Need to check complex scaling
    @pytest.mark.parametrize('in_dtype', FLOAT_TYPES + IUINT_TYPES)
    def test_no_scaling(self, in_dtype, supported_dtype):
        # Test writing image converting types when not calculating scaling
        img_class = self.image_class
        hdr_class = img_class.header_class
        hdr = hdr_class()
        # Any old non-default slope and intercept
        slope = 2
        inter = 10 if hdr.has_data_intercept else 0

        mn_in, mx_in = _dt_min_max(in_dtype)
        mn = -1 if np.dtype(in_dtype).kind != 'u' else 0
        arr = np.array([mn_in, mn, 0, 1, 10, mx_in], dtype=in_dtype)
        img = img_class(arr, np.eye(4), hdr)
        img.set_data_dtype(supported_dtype)
        # Setting the scaling means we don't calculate it later
        img.header.set_slope_inter(slope, inter)
        with np.errstate(invalid='ignore'):
            rt_img = bytesio_round_trip(img)
        with suppress_warnings():  # invalid mult
            back_arr = np.asanyarray(rt_img.dataobj)
        exp_back = arr.copy()
        # If converting to floating point type, casting is direct.
        # Otherwise we will need to do float-(u)int casting at some point
        if supported_dtype in IUINT_TYPES:
            if in_dtype in FLOAT_TYPES:
                # Working precision is (at least) float
                exp_back = exp_back.astype(float)
                # Float to iu conversion will always round, clip
                with np.errstate(invalid='ignore'):
                    exp_back = np.round(exp_back)
                if in_dtype in FLOAT_TYPES:
                    # Clip to shared range of working precision
                    exp_back = np.clip(exp_back, *shared_range(float, supported_dtype))
            else:  # iu input and output type
                # No scaling, never gets converted to float.
                # Does get clipped to range of output type
                mn_out, mx_out = _dt_min_max(supported_dtype)
                if (mn_in, mx_in) != (mn_out, mx_out):
                    # Use smaller of input, output range to avoid np.clip
                    # upcasting the array because of large clip limits.
                    exp_back = np.clip(exp_back, max(mn_in, mn_out), min(mx_in, mx_out))
        if supported_dtype in COMPLEX_TYPES:
            # always cast to real from complex
            exp_back = exp_back.astype(supported_dtype)
        else:
            # Cast to working precision
            exp_back = exp_back.astype(float)
        # Allow for small differences in large numbers
        with suppress_warnings():  # invalid value
            assert_allclose_safely(back_arr, exp_back * slope + inter)

    def test_write_scaling(self):
        # Check writes with scaling set
        for slope, inter, e_slope, e_inter in (
            (1, None, 1, None),
            (0, None, 1, None),
            (np.inf, None, 1, None),
            (2, None, 2, None),
        ):
            self._check_write_scaling(slope, inter, e_slope, e_inter)

    def test_nan2zero_range_ok(self):
        # Check that a floating point image with range not including zero gets
        # nans scaled correctly
        img_class = self.image_class
        arr = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
        arr[0, 0, 0] = np.nan
        arr[1, 0, 0] = 256  # to push outside uint8 range
        img = img_class(arr, np.eye(4))
        rt_img = bytesio_round_trip(img)
        assert_array_equal(rt_img.get_fdata(), arr)
        # Uncontroversial so far, but now check that nan2zero works correctly
        # for int type
        img.set_data_dtype(np.uint8)
        with np.errstate(invalid='ignore'):
            rt_img = bytesio_round_trip(img)
        assert rt_img.get_fdata()[0, 0, 0] == 0


class TestSpm99AnalyzeImage(test_analyze.TestAnalyzeImage, ImageScalingMixin):
    # class for testing images
    image_class = Spm99AnalyzeImage

    # Decorating the old way, before the team invented @
    test_data_hdr_cache = needs_scipy(test_analyze.TestAnalyzeImage.test_data_hdr_cache)
    test_header_updating = needs_scipy(test_analyze.TestAnalyzeImage.test_header_updating)
    test_offset_to_zero = needs_scipy(test_analyze.TestAnalyzeImage.test_offset_to_zero)
    test_big_offset_exts = needs_scipy(test_analyze.TestAnalyzeImage.test_big_offset_exts)
    test_dtype_to_filename_arg = needs_scipy(
        test_analyze.TestAnalyzeImage.test_dtype_to_filename_arg
    )
    test_header_scaling = needs_scipy(ImageScalingMixin.test_header_scaling)
    test_int_int_scaling = needs_scipy(ImageScalingMixin.test_int_int_scaling)
    test_write_scaling = needs_scipy(ImageScalingMixin.test_write_scaling)
    test_no_scaling = needs_scipy(ImageScalingMixin.test_no_scaling)
    test_nan2zero_range_ok = needs_scipy(ImageScalingMixin.test_nan2zero_range_ok)

    @needs_scipy
    def test_mat_read(self):
        # Test mat file reading and writing for the SPM analyze types
        img_klass = self.image_class
        arr = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
        aff = np.diag([2, 3, 4, 1])  # no LR flip in affine
        img = img_klass(arr, aff)
        fm = img.file_map
        for value in fm.values():
            value.fileobj = BytesIO()
        # Test round trip
        img.to_file_map()
        r_img = img_klass.from_file_map(fm)
        assert_array_equal(r_img.get_fdata(), arr)
        assert_array_equal(r_img.affine, aff)
        # mat files are for matlab and have 111 voxel origins.  We need to
        # adjust for that, when loading and saving.  Check for signs of that in
        # the saved mat file
        mat_fileobj = img.file_map['mat'].fileobj
        from scipy.io import loadmat, savemat

        mat_fileobj.seek(0)
        mats = loadmat(mat_fileobj)
        assert 'M' in mats and 'mat' in mats
        from_111 = np.eye(4)
        from_111[:3, 3] = -1
        to_111 = np.eye(4)
        to_111[:3, 3] = 1
        assert_array_equal(mats['mat'], np.dot(aff, from_111))
        # The M matrix does not include flips, so if we only have the M matrix
        # in the mat file, and we have default flipping, the mat resulting
        # should have a flip.  The 'mat' matrix does include flips and so
        # should be unaffected by the flipping.  If both are present we prefer
        # the the 'mat' matrix.
        assert img.header.default_x_flip  # check the default
        flipper = np.diag([-1, 1, 1, 1])
        assert_array_equal(mats['M'], np.dot(aff, np.dot(flipper, from_111)))
        mat_fileobj.seek(0)
        savemat(mat_fileobj, dict(M=np.diag([3, 4, 5, 1]), mat=np.diag([6, 7, 8, 1])))
        # Check we are preferring the 'mat' matrix
        r_img = img_klass.from_file_map(fm)
        assert_array_equal(r_img.get_fdata(), arr)
        assert_array_equal(r_img.affine, np.dot(np.diag([6, 7, 8, 1]), to_111))
        # But will use M if present
        mat_fileobj.seek(0)
        mat_fileobj.truncate(0)
        savemat(mat_fileobj, dict(M=np.diag([3, 4, 5, 1])))
        r_img = img_klass.from_file_map(fm)
        assert_array_equal(r_img.get_fdata(), arr)
        assert_array_equal(r_img.affine, np.dot(np.diag([3, 4, 5, 1]), np.dot(flipper, to_111)))

    def test_none_affine(self):
        # Allow for possibility of no affine resulting in nothing written into
        # mat file.  If the mat file is a filename, we just get no file, but if
        # it's a fileobj, we get an empty fileobj
        img_klass = self.image_class
        # With a None affine - no matfile written
        img = img_klass(np.zeros((2, 3, 4)), None)
        aff = img.header.get_best_affine()
        # Save / reload using bytes IO objects
        for value in img.file_map.values():
            value.fileobj = BytesIO()
        img.to_file_map()
        img_back = img.from_file_map(img.file_map)
        assert_array_equal(img_back.affine, aff)


def test_origin_affine():
    hdr = Spm99AnalyzeHeader()
    aff = hdr.get_origin_affine()
    assert_array_equal(aff, hdr.get_base_affine())
    hdr.set_data_shape((3, 5, 7))
    hdr.set_zooms((3, 2, 1))
    assert hdr.default_x_flip
    assert_array_almost_equal(
        hdr.get_origin_affine(),  # from center of image
        [
            [-3.0, 0.0, 0.0, 3.0],
            [0.0, 2.0, 0.0, -4.0],
            [0.0, 0.0, 1.0, -3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    hdr['origin'][:3] = [3, 4, 5]
    assert_array_almost_equal(
        hdr.get_origin_affine(),  # using origin
        [
            [-3.0, 0.0, 0.0, 6.0],
            [0.0, 2.0, 0.0, -6.0],
            [0.0, 0.0, 1.0, -4.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    hdr['origin'] = 0  # unset origin
    hdr.set_data_shape((3, 5))
    assert_array_almost_equal(
        hdr.get_origin_affine(),
        [
            [-3.0, 0.0, 0.0, 3.0],
            [0.0, 2.0, 0.0, -4.0],
            [0.0, 0.0, 1.0, -0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    hdr.set_data_shape((3, 5, 7))
    assert_array_almost_equal(
        hdr.get_origin_affine(),  # from center of image
        [
            [-3.0, 0.0, 0.0, 3.0],
            [0.0, 2.0, 0.0, -4.0],
            [0.0, 0.0, 1.0, -3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
