# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for nifti reading package"""

import os
import struct
import unittest
import warnings
from io import BytesIO

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal

from nibabel import nifti1 as nifti1
from nibabel.affines import from_matvec
from nibabel.casting import have_binary128, type_info
from nibabel.eulerangles import euler2mat
from nibabel.nifti1 import (
    Nifti1DicomExtension,
    Nifti1Extension,
    Nifti1Extensions,
    Nifti1Header,
    Nifti1Image,
    Nifti1Pair,
    Nifti1PairHeader,
    data_type_codes,
    extension_codes,
    load,
    slice_order_codes,
)
from nibabel.optpkg import optional_package
from nibabel.pkg_info import cmp_pkg_version
from nibabel.spatialimages import HeaderDataError
from nibabel.tmpdirs import InTemporaryDirectory

from ..freesurfer import load as mghload
from ..orientations import aff2axcodes
from ..testing import (
    bytesio_filemap,
    bytesio_round_trip,
    clear_and_catch_warnings,
    data_path,
    runif_extra_has,
    suppress_warnings,
)
from . import test_analyze as tana
from . import test_spm99analyze as tspm
from .nibabel_data import get_nibabel_data, needs_nibabel_data
from .test_arraywriters import IUINT_TYPES, rt_err_estimate
from .test_orientations import ALL_ORNTS

header_file = os.path.join(data_path, 'nifti1.hdr')
image_file = os.path.join(data_path, 'example4d.nii.gz')

pydicom, have_dicom, _ = optional_package('pydicom')
dicom_test = unittest.skipUnless(have_dicom, 'Could not import pydicom')


# Example transformation matrix
R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # rotation matrix
Z = [2.0, 3.0, 4.0]  # zooms
T = [20, 30, 40]  # translations
A = np.eye(4)
A[:3, :3] = np.array(R) * Z  # broadcasting does the job
A[:3, 3] = T


class TestNifti1PairHeader(tana.TestAnalyzeHeader, tspm.HeaderScalingMixin):
    header_class = Nifti1PairHeader
    example_file = header_file
    quat_dtype = np.float32
    supported_np_types = tana.TestAnalyzeHeader.supported_np_types.union(
        (np.int8, np.uint16, np.uint32, np.int64, np.uint64, np.complex128)
    )
    if have_binary128():
        supported_np_types = supported_np_types.union((np.longdouble, np.clongdouble))
    tana.add_duplicate_types(supported_np_types)

    def test_empty(self):
        tana.TestAnalyzeHeader.test_empty(self)
        hdr = self.header_class()
        assert hdr['magic'] == hdr.pair_magic
        assert hdr['scl_slope'] == 1
        assert hdr['vox_offset'] == 0

    def test_from_eg_file(self):
        hdr = self.header_class.from_fileobj(open(self.example_file, 'rb'))
        assert hdr.endianness == '<'
        assert hdr['magic'] == hdr.pair_magic
        assert hdr['sizeof_hdr'] == self.sizeof_hdr

    def test_data_scaling(self):
        # Test scaling in header
        super().test_data_scaling()
        hdr = self.header_class()
        data = np.arange(0, 3, 0.5).reshape((1, 2, 3))
        hdr.set_data_shape(data.shape)
        hdr.set_data_dtype(np.float32)
        S = BytesIO()
        # Writing to float dtype with scaling gives slope, intercept as (1, 0)
        hdr.data_to_fileobj(data, S, rescale=True)
        assert_array_equal(hdr.get_slope_inter(), (1, 0))
        rdata = hdr.data_from_fileobj(S)
        assert_array_almost_equal(data, rdata)
        # Writing to integer datatype with scaling gives non-identity scaling
        hdr.set_data_dtype(np.int8)
        hdr.set_slope_inter(1, 0)
        hdr.data_to_fileobj(data, S, rescale=True)
        assert not np.allclose(hdr.get_slope_inter(), (1, 0))
        rdata = hdr.data_from_fileobj(S)
        assert_array_almost_equal(data, rdata)
        # Without scaling does rounding, doesn't alter scaling
        hdr.set_slope_inter(1, 0)
        with np.errstate(invalid='ignore'):
            hdr.data_to_fileobj(data, S, rescale=False)
        assert_array_equal(hdr.get_slope_inter(), (1, 0))
        rdata = hdr.data_from_fileobj(S)
        assert_array_almost_equal(np.round(data), rdata)

    def test_big_scaling(self):
        # Test that upcasting works for huge scalefactors
        # See tests for apply_read_scaling in test_volumeutils
        hdr = self.header_class()
        hdr.set_data_shape((2, 1, 1))
        hdr.set_data_dtype(np.int16)
        sio = BytesIO()
        dtt = np.float32
        # This will generate a huge scalefactor
        finf = type_info(dtt)
        data = np.array([finf['min'], finf['max']], dtype=dtt)[:, None, None]
        hdr.data_to_fileobj(data, sio)
        data_back = hdr.data_from_fileobj(sio)
        assert np.allclose(data, data_back)

    def test_slope_inter(self):
        hdr = self.header_class()
        nan, inf, minf = np.nan, np.inf, -np.inf
        HDE = HeaderDataError
        assert hdr.get_slope_inter() == (1.0, 0.0)
        for in_tup, exp_err, out_tup, raw_values in (
            # Null scalings
            ((None, None), None, (None, None), (nan, nan)),
            ((nan, None), None, (None, None), (nan, nan)),
            ((None, nan), None, (None, None), (nan, nan)),
            ((nan, nan), None, (None, None), (nan, nan)),
            # Can only be one null
            ((None, 0), HDE, (None, None), (nan, 0)),
            ((nan, 0), HDE, (None, None), (nan, 0)),
            ((1, None), HDE, (None, None), (1, nan)),
            ((1, nan), HDE, (None, None), (1, nan)),
            # Bad slope plus anything generates an error
            ((0, 0), HDE, (None, None), (0, 0)),
            ((0, None), HDE, (None, None), (0, nan)),
            ((0, nan), HDE, (None, None), (0, nan)),
            ((0, inf), HDE, (None, None), (0, inf)),
            ((0, minf), HDE, (None, None), (0, minf)),
            ((inf, 0), HDE, (None, None), (inf, 0)),
            ((inf, None), HDE, (None, None), (inf, nan)),
            ((inf, nan), HDE, (None, None), (inf, nan)),
            ((inf, inf), HDE, (None, None), (inf, inf)),
            ((inf, minf), HDE, (None, None), (inf, minf)),
            ((minf, 0), HDE, (None, None), (minf, 0)),
            ((minf, None), HDE, (None, None), (minf, nan)),
            ((minf, nan), HDE, (None, None), (minf, nan)),
            ((minf, inf), HDE, (None, None), (minf, inf)),
            ((minf, minf), HDE, (None, None), (minf, minf)),
            # Good slope and bad inter generates error for get_slope_inter
            ((2, None), HDE, HDE, (2, nan)),
            ((2, nan), HDE, HDE, (2, nan)),
            ((2, inf), HDE, HDE, (2, inf)),
            ((2, minf), HDE, HDE, (2, minf)),
            # Good slope and inter - you guessed it
            ((2, 0), None, (2, 0), (2, 0)),
            ((2, 1), None, (2, 1), (2, 1)),
        ):
            hdr = self.header_class()
            if not exp_err is None:
                with pytest.raises(exp_err):
                    hdr.set_slope_inter(*in_tup)
                in_list = [v if not v is None else np.nan for v in in_tup]
                hdr['scl_slope'], hdr['scl_inter'] = in_list
            else:
                hdr.set_slope_inter(*in_tup)
                if isinstance(out_tup, Exception):
                    with pytest.raises(out_tup):
                        hdr.get_slope_inter()
                else:
                    assert hdr.get_slope_inter() == out_tup
                    # Check set survives through checking
                    hdr = self.header_class.from_header(hdr, check=True)
                    assert hdr.get_slope_inter() == out_tup
            assert_array_equal([hdr['scl_slope'], hdr['scl_inter']], raw_values)

    def test_nifti_qfac_checks(self):
        # Test qfac is 1 or -1
        hdr = self.header_class()
        # 1, -1 OK
        hdr['pixdim'][0] = 1
        self.log_chk(hdr, 0)
        hdr['pixdim'][0] = -1
        self.log_chk(hdr, 0)
        # 0 is not
        hdr['pixdim'][0] = 0
        fhdr, message, raiser = self.log_chk(hdr, 20)
        assert fhdr['pixdim'][0] == 1
        assert message == 'pixdim[0] (qfac) should be 1 (default) or -1; setting qfac to 1'

    def test_nifti_qsform_checks(self):
        # qform, sform checks
        HC = self.header_class
        # qform, sform
        hdr = HC()
        hdr['qform_code'] = -1
        fhdr, message, raiser = self.log_chk(hdr, 30)
        assert fhdr['qform_code'] == 0
        assert message == 'qform_code -1 not valid; setting to 0'
        hdr = HC()
        hdr['sform_code'] = -1
        fhdr, message, raiser = self.log_chk(hdr, 30)
        assert fhdr['sform_code'] == 0
        assert message == 'sform_code -1 not valid; setting to 0'

    def test_nifti_xform_codes(self):
        # Verify that all xform codes can be set in both qform and sform
        hdr = self.header_class()
        affine = np.eye(4)
        for code in nifti1.xform_codes.keys():
            hdr.set_qform(affine, code)
            assert hdr['qform_code'] == nifti1.xform_codes[code]
            hdr.set_sform(affine, code)
            assert hdr['sform_code'] == nifti1.xform_codes[code]

        # Raise KeyError on unknown code
        for bad_code in (-1, 6, 10):
            with pytest.raises(KeyError):
                hdr.set_qform(affine, bad_code)
            with pytest.raises(KeyError):
                hdr.set_sform(affine, bad_code)

    def test_magic_offset_checks(self):
        # magic and offset
        HC = self.header_class
        hdr = HC()
        hdr['magic'] = 'ooh'
        fhdr, message, raiser = self.log_chk(hdr, 45)
        assert fhdr['magic'] == b'ooh'
        assert (
            message == "magic string 'ooh' is not valid; "
            'leaving as is, but future errors are likely'
        )
        # For pairs, any offset is OK, but should be divisible by 16
        # Singles need offset of at least 352 (nifti1) or 540 (nifti2) bytes,
        # with the divide by 16 rule
        svo = hdr.single_vox_offset
        for magic, ok, bad_spm in (
            (hdr.pair_magic, 32, 40),
            (hdr.single_magic, svo + 32, svo + 40),
        ):
            hdr['magic'] = magic
            hdr['vox_offset'] = 0
            self.assert_no_log_err(hdr)
            hdr['vox_offset'] = ok
            self.assert_no_log_err(hdr)
            hdr['vox_offset'] = bad_spm
            fhdr, message, raiser = self.log_chk(hdr, 30)
            assert fhdr['vox_offset'] == bad_spm
            assert (
                message == f'vox offset (={bad_spm:g}) not divisible by 16, '
                'not SPM compatible; leaving at current value'
            )
        # Check minimum offset (if offset set)
        hdr['magic'] = hdr.single_magic
        hdr['vox_offset'] = 10
        fhdr, message, raiser = self.log_chk(hdr, 40)
        assert fhdr['vox_offset'] == hdr.single_vox_offset
        assert (
            message == 'vox offset 10 too low for single '
            'file nifti1; setting to minimum value '
            'of ' + str(hdr.single_vox_offset)
        )

    def test_freesurfer_large_vector_hack(self):
        # For large vector images, Freesurfer appears to set dim[1] to -1 and
        # then use glmin for the vector length (an i4)
        HC = self.header_class
        # The standard case
        hdr = HC()
        hdr.set_data_shape((2, 3, 4))
        assert hdr.get_data_shape() == (2, 3, 4)
        assert hdr['glmin'] == 0
        # Just left of the freesurfer case
        dim_type = hdr.template_dtype['dim'].base
        glmin = hdr.template_dtype['glmin'].base
        too_big = int(np.iinfo(dim_type).max) + 1
        hdr.set_data_shape((too_big - 1, 1, 1))
        assert hdr.get_data_shape() == (too_big - 1, 1, 1)
        # The freesurfer case
        full_shape = (too_big, 1, 1, 1, 1, 1, 1)
        for dim in range(3, 8):
            # First element in 'dim' field is number of dimensions
            expected_dim = np.array([dim, -1, 1, 1, 1, 1, 1, 1])
            with suppress_warnings():
                hdr.set_data_shape(full_shape[:dim])
            assert hdr.get_data_shape() == full_shape[:dim]
            assert_array_equal(hdr['dim'], expected_dim)
            assert hdr['glmin'] == too_big
        # Allow the fourth dimension to vary
        with suppress_warnings():
            hdr.set_data_shape((too_big, 1, 1, 4))
        assert hdr.get_data_shape() == (too_big, 1, 1, 4)
        assert_array_equal(hdr['dim'][:5], np.array([4, -1, 1, 1, 4]))
        # This only works when the first 3 dimensions are -1, 1, 1
        pytest.raises(HeaderDataError, hdr.set_data_shape, (too_big,))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (too_big, 1))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (too_big, 1, 2))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (too_big, 2, 1))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (1, too_big))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (1, too_big, 1))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (1, 1, too_big))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (1, 1, 1, too_big))
        # Outside range of glmin raises error
        far_too_big = int(np.iinfo(glmin).max) + 1
        with suppress_warnings():
            hdr.set_data_shape((far_too_big - 1, 1, 1))
        assert hdr.get_data_shape() == (far_too_big - 1, 1, 1)
        with pytest.raises(HeaderDataError):
            hdr.set_data_shape((far_too_big, 1, 1))
        # glmin of zero raises error (implausible vector length)
        hdr.set_data_shape((-1, 1, 1))
        hdr['glmin'] = 0
        with pytest.raises(HeaderDataError):
            hdr.get_data_shape()
        # Lists or tuples or arrays will work for setting shape
        for shape in ((too_big - 1, 1, 1), (too_big, 1, 1)):
            for constructor in (list, tuple, np.array):
                with suppress_warnings():
                    hdr.set_data_shape(constructor(shape))
                assert hdr.get_data_shape() == shape

    @needs_nibabel_data('nitest-freesurfer')
    def test_freesurfer_ico7_hack(self):
        HC = self.header_class
        hdr = HC()
        full_shape = (163842, 1, 1, 1, 1, 1, 1)
        # Test that using ico7 shape automatically uses factored dimensions
        for dim in range(3, 8):
            expected_dim = np.array([dim, 27307, 1, 6, 1, 1, 1, 1])
            hdr.set_data_shape(full_shape[:dim])
            assert hdr.get_data_shape() == full_shape[:dim]
            assert_array_equal(hdr._structarr['dim'], expected_dim)
        # Only works on dimensions >= 3
        pytest.raises(HeaderDataError, hdr.set_data_shape, full_shape[:1])
        pytest.raises(HeaderDataError, hdr.set_data_shape, full_shape[:2])
        # Bad shapes
        pytest.raises(HeaderDataError, hdr.set_data_shape, (163842, 2, 1))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (163842, 1, 2))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (1, 163842, 1))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (1, 1, 163842))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (1, 1, 1, 163842))
        # Test consistency of data in .mgh and mri_convert produced .nii
        nitest_path = os.path.join(get_nibabel_data(), 'nitest-freesurfer')
        mgh = mghload(os.path.join(nitest_path, 'fsaverage', 'surf', 'lh.orig.avg.area.mgh'))
        nii = load(
            os.path.join(nitest_path, 'derivative', 'fsaverage', 'surf', 'lh.orig.avg.area.nii')
        )
        assert mgh.shape == nii.shape
        assert_array_equal(mgh.get_fdata(), nii.get_fdata())
        assert_array_equal(nii.header._structarr['dim'][1:4], np.array([27307, 1, 6]))
        # Test writing produces consistent nii files
        with InTemporaryDirectory():
            nii.to_filename('test.nii')
            nii2 = load('test.nii')
            assert nii.shape == nii2.shape
            assert_array_equal(nii.get_fdata(), nii2.get_fdata())
            assert_array_equal(nii.affine, nii2.affine)

    def test_qform_sform(self):
        HC = self.header_class
        hdr = HC()
        assert_array_equal(hdr.get_qform(), np.eye(4))
        empty_sform = np.zeros((4, 4))
        empty_sform[-1, -1] = 1
        assert_array_equal(hdr.get_sform(), empty_sform)
        assert hdr.get_qform(coded=True) == (None, 0)
        assert hdr.get_sform(coded=True) == (None, 0)
        # Affines with no shears
        nice_aff = np.diag([2, 3, 4, 1])
        another_aff = np.diag([3, 4, 5, 1])
        # Affine with shears
        nasty_aff = from_matvec(np.arange(9).reshape((3, 3)), [9, 10, 11])
        nasty_aff[0, 0] = 1  # Make full rank
        fixed_aff = unshear_44(nasty_aff)
        assert not np.allclose(fixed_aff, nasty_aff)
        for in_meth, out_meth in ((hdr.set_qform, hdr.get_qform), (hdr.set_sform, hdr.get_sform)):
            in_meth(nice_aff, 2)
            aff, code = out_meth(coded=True)
            assert_array_equal(aff, nice_aff)
            assert code == 2
            assert_array_equal(out_meth(), nice_aff)  # non coded
            # Affine may be passed if code == 0, and will get set into header,
            # but the returned affine with 'coded=True' will be None.
            in_meth(another_aff, 0)
            assert out_meth(coded=True) == (None, 0)  # coded -> None
            assert_array_almost_equal(out_meth(), another_aff)  # else -> input
            # Default qform code when previous == 0 is 2
            in_meth(nice_aff)
            aff, code = out_meth(coded=True)
            assert code == 2
            # Unless code was non-zero before
            in_meth(nice_aff, 1)
            in_meth(nice_aff)
            aff, code = out_meth(coded=True)
            assert code == 1
            # Can set code without modifying affine, by passing affine=None
            assert_array_equal(aff, nice_aff)  # affine same as before
            in_meth(None, 3)
            aff, code = out_meth(coded=True)
            assert_array_equal(aff, nice_aff)  # affine same as before
            assert code == 3
            # affine is None on its own, or with code==0, resets code to 0
            in_meth(None, 0)
            assert out_meth(coded=True) == (None, 0)
            in_meth(None)
            assert out_meth(coded=True) == (None, 0)
            # List works as input
            in_meth(nice_aff.tolist())
            assert_array_equal(out_meth(), nice_aff)
        # Qform specifics
        # inexact set (with shears) is OK
        hdr.set_qform(nasty_aff, 1)
        assert_array_almost_equal(hdr.get_qform(), fixed_aff)
        # Unless allow_shears is False
        with pytest.raises(HeaderDataError):
            hdr.set_qform(nasty_aff, 1, False)
        # Reset sform, give qform a code, to test sform
        hdr.set_sform(None)
        hdr.set_qform(nice_aff, 1)
        # Check sform unchanged by setting qform
        assert hdr.get_sform(coded=True) == (None, 0)
        # Setting does change the sform output
        hdr.set_sform(nasty_aff, 1)
        aff, code = hdr.get_sform(coded=True)
        assert_array_equal(aff, nasty_aff)
        assert code == 1

    def test_datatypes(self):
        hdr = self.header_class()
        for code in data_type_codes.value_set():
            dt = data_type_codes.type[code]
            if dt == np.void:
                continue
            hdr.set_data_dtype(code)
            assert hdr.get_data_dtype() == data_type_codes.dtype[code]
        # Check that checks also see new datatypes
        hdr.set_data_dtype(np.complex128)
        hdr.check_fix()

    def test_quaternion(self):
        hdr = self.header_class()
        hdr['quatern_b'] = 0
        hdr['quatern_c'] = 0
        hdr['quatern_d'] = 0
        assert np.allclose(hdr.get_qform_quaternion(), [1.0, 0, 0, 0])
        hdr['quatern_b'] = 1
        hdr['quatern_c'] = 0
        hdr['quatern_d'] = 0
        assert np.allclose(hdr.get_qform_quaternion(), [0, 1, 0, 0])
        # Check threshold set correctly for float32
        hdr['quatern_b'] = 1 + np.finfo(self.quat_dtype).eps
        assert_array_almost_equal(hdr.get_qform_quaternion(), [0, 1, 0, 0])

    def test_qform(self):
        # Test roundtrip case
        ehdr = self.header_class()
        ehdr.set_qform(A)
        qA = ehdr.get_qform()
        assert np.allclose(A, qA, atol=1e-5)
        assert np.allclose(Z, ehdr['pixdim'][1:4])
        xfas = nifti1.xform_codes
        assert ehdr['qform_code'] == xfas['aligned']
        ehdr.set_qform(A, 'scanner')
        assert ehdr['qform_code'] == xfas['scanner']
        ehdr.set_qform(A, xfas['aligned'])
        assert ehdr['qform_code'] == xfas['aligned']
        # Test pixdims[1,2,3] are checked for negatives
        for dims in ((-1, 1, 1), (1, -1, 1), (1, 1, -1)):
            ehdr['pixdim'][1:4] = dims
            with pytest.raises(HeaderDataError):
                ehdr.get_qform()

    def test_sform(self):
        # Test roundtrip case
        ehdr = self.header_class()
        ehdr.set_sform(A)
        sA = ehdr.get_sform()
        assert np.allclose(A, sA, atol=1e-5)
        xfas = nifti1.xform_codes
        assert ehdr['sform_code'] == xfas['aligned']
        ehdr.set_sform(A, 'scanner')
        assert ehdr['sform_code'] == xfas['scanner']
        ehdr.set_sform(A, xfas['aligned'])
        assert ehdr['sform_code'] == xfas['aligned']

    def test_dim_info(self):
        ehdr = self.header_class()
        assert ehdr.get_dim_info() == (None, None, None)
        for info in (
            (0, 2, 1),
            (None, None, None),
            (0, 2, None),
            (0, None, None),
            (None, 2, 1),
            (None, None, 1),
        ):
            ehdr.set_dim_info(*info)
            assert ehdr.get_dim_info() == info

    def test_slice_times(self):
        hdr = self.header_class()
        # error if slice dimension not specified
        with pytest.raises(HeaderDataError):
            hdr.get_slice_times()
        hdr.set_dim_info(slice=2)
        # error if slice dimension outside shape
        with pytest.raises(HeaderDataError):
            hdr.get_slice_times()
        hdr.set_data_shape((1, 1, 7))
        # error if slice duration not set
        with pytest.raises(HeaderDataError):
            hdr.get_slice_times()
        hdr.set_slice_duration(0.1)
        # We need a function to print out the Nones and floating point
        # values in a predictable way, for the tests below.
        _stringer = lambda val: val is not None and f'{val:2.1f}' or None
        _print_me = lambda s: list(map(_stringer, s))
        # The following examples are from the nifti1.h documentation.
        hdr['slice_code'] = slice_order_codes['sequential increasing']
        assert _print_me(hdr.get_slice_times()) == [
            '0.0',
            '0.1',
            '0.2',
            '0.3',
            '0.4',
            '0.5',
            '0.6',
        ]
        hdr['slice_start'] = 1
        hdr['slice_end'] = 5
        assert _print_me(hdr.get_slice_times()) == [None, '0.0', '0.1', '0.2', '0.3', '0.4', None]
        hdr['slice_code'] = slice_order_codes['sequential decreasing']
        assert _print_me(hdr.get_slice_times()) == [None, '0.4', '0.3', '0.2', '0.1', '0.0', None]
        hdr['slice_code'] = slice_order_codes['alternating increasing']
        assert _print_me(hdr.get_slice_times()) == [None, '0.0', '0.3', '0.1', '0.4', '0.2', None]
        hdr['slice_code'] = slice_order_codes['alternating decreasing']
        assert _print_me(hdr.get_slice_times()) == [None, '0.2', '0.4', '0.1', '0.3', '0.0', None]
        hdr['slice_code'] = slice_order_codes['alternating increasing 2']
        assert _print_me(hdr.get_slice_times()) == [None, '0.2', '0.0', '0.3', '0.1', '0.4', None]
        hdr['slice_code'] = slice_order_codes['alternating decreasing 2']
        assert _print_me(hdr.get_slice_times()) == [None, '0.4', '0.1', '0.3', '0.0', '0.2', None]
        # test set
        hdr = self.header_class()
        hdr.set_dim_info(slice=2)
        # need slice dim to correspond with shape
        times = [None, 0.2, 0.4, 0.1, 0.3, 0.0, None]
        with pytest.raises(HeaderDataError):
            hdr.set_slice_times(times)
        hdr.set_data_shape([1, 1, 7])
        with pytest.raises(HeaderDataError):
            # wrong length
            hdr.set_slice_times(times[:-1])
        with pytest.raises(HeaderDataError):
            # all None
            hdr.set_slice_times((None,) * len(times))
        n_mid_times = times.copy()
        n_mid_times[3] = None
        with pytest.raises(HeaderDataError):
            # None in middle
            hdr.set_slice_times(n_mid_times)
        funny_times = times.copy()
        funny_times[3] = 0.05
        with pytest.raises(HeaderDataError):
            # can't get single slice duration
            hdr.set_slice_times(funny_times)
        hdr.set_slice_times(times)
        assert hdr.get_value_label('slice_code') == 'alternating decreasing'
        assert hdr['slice_start'] == 1
        assert hdr['slice_end'] == 5
        assert_array_almost_equal(hdr['slice_duration'], 0.1)

        # Ambiguous case
        hdr2 = self.header_class()
        hdr2.set_dim_info(slice=2)
        hdr2.set_slice_duration(0.1)
        hdr2.set_data_shape((1, 1, 2))
        with pytest.warns(UserWarning) as w:
            hdr2.set_slice_times([0.1, 0])
            assert len(w) == 1
        # but always must be choosing sequential one first
        assert hdr2.get_value_label('slice_code') == 'sequential decreasing'
        # and the other direction
        with pytest.warns(UserWarning) as w:
            hdr2.set_slice_times([0, 0.1])
            assert len(w) == 1
        assert hdr2.get_value_label('slice_code') == 'sequential increasing'

    def test_intents(self):
        ehdr = self.header_class()
        ehdr.set_intent('t test', (10,), name='some score')
        assert ehdr.get_intent() == ('t test', (10.0,), 'some score')
        # unknown intent name or code - unknown name will fail even when
        # allow_unknown=True
        with pytest.raises(KeyError):
            ehdr.set_intent('no intention')
        with pytest.raises(KeyError):
            ehdr.set_intent('no intention', allow_unknown=True)
        with pytest.raises(KeyError):
            ehdr.set_intent(32767)
        # too many parameters
        with pytest.raises(HeaderDataError):
            ehdr.set_intent('t test', (10, 10))
        # too few parameters
        with pytest.raises(HeaderDataError):
            ehdr.set_intent('f test', (10,))
        # check unset parameters are set to 0, and name to ''
        ehdr.set_intent('t test')
        assert (ehdr['intent_p1'], ehdr['intent_p2'], ehdr['intent_p3']) == (0, 0, 0)
        assert ehdr['intent_name'] == b''
        ehdr.set_intent('t test', (10,))
        assert (ehdr['intent_p2'], ehdr['intent_p3']) == (0, 0)
        # store intent that is not in nifti1.intent_codes recoder
        ehdr.set_intent(9999, allow_unknown=True)
        assert ehdr.get_intent() == ('unknown code 9999', (), '')
        assert ehdr.get_intent('code') == (9999, (), '')
        ehdr.set_intent(9999, name='custom intent', allow_unknown=True)
        assert ehdr.get_intent() == ('unknown code 9999', (), 'custom intent')
        assert ehdr.get_intent('code') == (9999, (), 'custom intent')
        # store unknown intent with parameters. set_intent will set the
        # parameters, but get_intent won't return them
        ehdr.set_intent(code=9999, params=(1, 2, 3), allow_unknown=True)
        assert ehdr.get_intent() == ('unknown code 9999', (), '')
        assert ehdr.get_intent('code') == (9999, (), '')
        # unknown intent requires either zero, or three, parameters
        with pytest.raises(HeaderDataError):
            ehdr.set_intent(999, (1,), allow_unknown=True)
        with pytest.raises(HeaderDataError):
            ehdr.set_intent(999, (1, 2), allow_unknown=True)

    def test_set_slice_times(self):
        hdr = self.header_class()
        hdr.set_dim_info(slice=2)
        hdr.set_data_shape([1, 1, 7])
        hdr.set_slice_duration(0.1)
        times = [0] * 6
        pytest.raises(HeaderDataError, hdr.set_slice_times, times)
        times = [None] * 7
        pytest.raises(HeaderDataError, hdr.set_slice_times, times)
        times = [None, 0, 1, None, 3, 4, None]
        pytest.raises(HeaderDataError, hdr.set_slice_times, times)
        times = [None, 0, 1, 2.1, 3, 4, None]
        pytest.raises(HeaderDataError, hdr.set_slice_times, times)
        times = [None, 0, 4, 3, 2, 1, None]
        pytest.raises(HeaderDataError, hdr.set_slice_times, times)
        times = [0, 1, 2, 3, 4, 5, 6]
        hdr.set_slice_times(times)
        assert hdr['slice_code'] == 1
        assert hdr['slice_start'] == 0
        assert hdr['slice_end'] == 6
        assert hdr['slice_duration'] == 1.0
        times = [None, 0, 1, 2, 3, 4, None]
        hdr.set_slice_times(times)
        assert hdr['slice_code'] == 1
        assert hdr['slice_start'] == 1
        assert hdr['slice_end'] == 5
        assert hdr['slice_duration'] == 1.0
        times = [None, 0.4, 0.3, 0.2, 0.1, 0, None]
        hdr.set_slice_times(times)
        assert np.allclose(hdr['slice_duration'], 0.1)
        times = [None, 4, 3, 2, 1, 0, None]
        hdr.set_slice_times(times)
        assert hdr['slice_code'] == 2
        times = [None, 0, 3, 1, 4, 2, None]
        hdr.set_slice_times(times)
        assert hdr['slice_code'] == 3
        times = [None, 2, 4, 1, 3, 0, None]
        hdr.set_slice_times(times)
        assert hdr['slice_code'] == 4
        times = [None, 2, 0, 3, 1, 4, None]
        hdr.set_slice_times(times)
        assert hdr['slice_code'] == 5
        times = [None, 4, 1, 3, 0, 2, None]
        hdr.set_slice_times(times)
        assert hdr['slice_code'] == 6

    def test_xyzt_units(self):
        hdr = self.header_class()
        assert hdr.get_xyzt_units() == ('unknown', 'unknown')
        hdr.set_xyzt_units('mm', 'sec')
        assert hdr.get_xyzt_units() == ('mm', 'sec')
        hdr.set_xyzt_units()
        assert hdr.get_xyzt_units() == ('unknown', 'unknown')

    def test_recoded_fields(self):
        hdr = self.header_class()
        assert hdr.get_value_label('qform_code') == 'unknown'
        hdr['qform_code'] = 3
        assert hdr.get_value_label('qform_code') == 'talairach'
        assert hdr.get_value_label('sform_code') == 'unknown'
        hdr['sform_code'] = 3
        assert hdr.get_value_label('sform_code') == 'talairach'
        assert hdr.get_value_label('intent_code') == 'none'
        hdr.set_intent('t test', (10,), name='some score')
        assert hdr.get_value_label('intent_code') == 't test'
        assert hdr.get_value_label('slice_code') == 'unknown'
        hdr['slice_code'] = 4  # alternating decreasing
        assert hdr.get_value_label('slice_code') == 'alternating decreasing'


def unshear_44(affine):
    RZS = affine[:3, :3]
    zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
    R = RZS / zooms
    P, S, Qs = np.linalg.svd(R)
    PR = np.dot(P, Qs)
    return from_matvec(PR * zooms, affine[:3, 3])


class TestNifti1SingleHeader(TestNifti1PairHeader):
    header_class = Nifti1Header

    def test_empty(self):
        tana.TestAnalyzeHeader.test_empty(self)
        hdr = self.header_class()
        assert hdr['magic'] == hdr.single_magic
        assert hdr['scl_slope'] == 1
        assert hdr['vox_offset'] == 0

    def test_binblock_is_file(self):
        # Override test that binary string is the same as the file on disk; in
        # the case of the single file version of the header, we need to append
        # the extension string (4 0s)
        hdr = self.header_class()
        str_io = BytesIO()
        hdr.write_to(str_io)
        assert str_io.getvalue() == hdr.binaryblock + b'\x00' * 4

    def test_float128(self):
        hdr = self.header_class()
        # Allow for Windows visual studio where longdouble is float64
        ld_dt = np.dtype(np.longdouble)
        if have_binary128() or ld_dt == np.dtype(np.float64):
            hdr.set_data_dtype(np.longdouble)
            assert hdr.get_data_dtype() == ld_dt
        else:
            with pytest.raises(HeaderDataError):
                hdr.set_data_dtype(np.longdouble)


class TestNifti1Pair(tana.TestAnalyzeImage, tspm.ImageScalingMixin):
    # Run analyze-flavor spatialimage tests
    image_class = Nifti1Pair
    supported_np_types = TestNifti1PairHeader.supported_np_types

    def test_int64_warning_or_error(self):
        # Verify that initializing with (u)int64 data and no
        # header/dtype info produces a warning/error
        img_klass = self.image_class
        hdr_klass = img_klass.header_class
        for dtype in (np.int64, np.uint64):
            data = np.arange(24, dtype=dtype).reshape((2, 3, 4))
            # Starts as a warning, transitions to error at 5.0
            if cmp_pkg_version('5.0') <= 0:
                cm = pytest.raises(ValueError)
            else:
                cm = pytest.warns(FutureWarning)
            with cm:
                img_klass(data, np.eye(4))
            # No problems if we're explicit, though
            with clear_and_catch_warnings():
                warnings.simplefilter('error')
                img_klass(data, np.eye(4), dtype=dtype)
                hdr = hdr_klass()
                hdr.set_data_dtype(dtype)
                img_klass(data, np.eye(4), hdr)

    def test_none_qsform(self):
        # Check that affine gets set to q/sform if header is None
        img_klass = self.image_class
        hdr_klass = img_klass.header_class
        shape = (2, 3, 4)
        data = np.arange(24, dtype='f4').reshape((2, 3, 4))
        # With specified affine
        aff = from_matvec(euler2mat(0.1, 0.2, 0.3), [11, 12, 13])
        for hdr in (None, hdr_klass()):
            img = img_klass(data, aff, hdr)
            assert_almost_equal(img.affine, aff)
            assert_almost_equal(img.header.get_sform(), aff)
            assert_almost_equal(img.header.get_qform(), aff)
        # Even if affine is default for empty header
        hdr = hdr_klass()
        hdr.set_data_shape(shape)
        default_aff = hdr.get_best_affine()
        img = img_klass(data, default_aff, None)
        assert_almost_equal(img.header.get_sform(), default_aff)
        assert_almost_equal(img.header.get_qform(), default_aff)
        # If affine is None, s/qform not set
        img = img_klass(data, None, None)
        assert_almost_equal(img.header.get_sform(), np.diag([0, 0, 0, 1]))
        assert_almost_equal(img.header.get_qform(), np.eye(4))

    def _qform_rt(self, img):
        # Round trip image after setting qform, sform codes
        hdr = img.header
        hdr['qform_code'] = 3
        hdr['sform_code'] = 4
        # Save / reload using bytes IO objects
        for value in img.file_map.values():
            value.fileobj = BytesIO()
        img.to_file_map()
        return img.from_file_map(img.file_map)

    def test_qform_cycle(self):
        # Qform load save cycle
        img_klass = self.image_class
        # None affine
        img = img_klass(np.zeros((2, 3, 4)), None)
        hdr_back = self._qform_rt(img).header
        assert hdr_back['qform_code'] == 3
        assert hdr_back['sform_code'] == 4
        # Try non-None affine
        img = img_klass(np.zeros((2, 3, 4)), np.eye(4))
        hdr_back = self._qform_rt(img).header
        assert hdr_back['qform_code'] == 3
        assert hdr_back['sform_code'] == 4
        # Modify affine in-place - does it hold?
        img.affine[0, 0] = 9
        img.to_file_map()
        img_back = img.from_file_map(img.file_map)
        exp_aff = np.diag([9, 1, 1, 1])
        assert_array_equal(img_back.affine, exp_aff)
        hdr_back = img.header
        assert_array_equal(hdr_back.get_sform(), exp_aff)
        assert_array_equal(hdr_back.get_qform(), exp_aff)

    def test_header_update_affine(self):
        # Test that updating occurs only if affine is not allclose
        img = self.image_class(np.zeros((2, 3, 4)), np.eye(4))
        hdr = img.header
        aff = img.affine
        aff[:] = np.diag([1.1, 1.1, 1.1, 1])  # inexact floats
        hdr.set_qform(aff, 2)
        hdr.set_sform(aff, 2)
        img.update_header()
        assert hdr['sform_code'] == 2
        assert hdr['qform_code'] == 2

    def test_set_qform(self):
        img = self.image_class(np.zeros((2, 3, 4)), np.diag([2.2, 3.3, 4.3, 1]))
        hdr = img.header
        new_affine = np.diag([1.1, 1.1, 1.1, 1])
        # Affine is same as sform (best affine)
        assert_array_almost_equal(img.affine, hdr.get_best_affine())
        # Reset affine to something different again
        aff_affine = np.diag([3.3, 4.5, 6.6, 1])
        img.affine[:] = aff_affine
        assert_array_almost_equal(img.affine, aff_affine)
        # Set qform using new_affine
        img.set_qform(new_affine, 1)
        assert_array_almost_equal(img.get_qform(), new_affine)
        assert hdr['qform_code'] == 1
        # Image get is same as header get
        assert_array_almost_equal(img.get_qform(), new_affine)
        # Coded version of get gets same information
        qaff, code = img.get_qform(coded=True)
        assert code == 1
        assert_array_almost_equal(qaff, new_affine)
        # Image affine now reset to best affine (which is sform)
        assert_array_almost_equal(img.affine, hdr.get_best_affine())
        # Reset image affine and try update_affine == False
        img.affine[:] = aff_affine
        img.set_qform(new_affine, 1, update_affine=False)
        assert_array_almost_equal(img.affine, aff_affine)
        # Clear qform using None, zooms unchanged
        assert_array_almost_equal(hdr.get_zooms(), [1.1, 1.1, 1.1])
        img.set_qform(None)
        qaff, code = img.get_qform(coded=True)
        assert (qaff, code) == (None, 0)
        assert_array_almost_equal(hdr.get_zooms(), [1.1, 1.1, 1.1])
        # Best affine similarly
        assert_array_almost_equal(img.affine, hdr.get_best_affine())
        # If sform is not set, qform should update affine
        img.set_sform(None)
        img.set_qform(new_affine, 1)
        qaff, code = img.get_qform(coded=True)
        assert code == 1
        assert_array_almost_equal(img.affine, new_affine)
        new_affine[0, 1] = 2
        # If affine has has shear, should raise Error if strip_shears=False
        img.set_qform(new_affine, 2)
        with pytest.raises(HeaderDataError):
            img.set_qform(new_affine, 2, False)
        # Unexpected keyword raises error
        with pytest.raises(TypeError):
            img.get_qform(strange=True)
        # updating None affine, None header does not work, because None header
        # results in setting the sform to default
        img = self.image_class(np.zeros((2, 3, 4)), None)
        new_affine = np.eye(4)
        img.set_qform(new_affine, 2)
        assert_array_almost_equal(img.affine, img.header.get_best_affine())
        # Unless we unset the sform
        img.set_sform(None, update_affine=True)
        assert_array_almost_equal(img.affine, new_affine)

    def test_set_sform(self):
        orig_aff = np.diag([2.2, 3.3, 4.3, 1])
        img = self.image_class(np.zeros((2, 3, 4)), orig_aff)
        hdr = img.header
        new_affine = np.diag([1.1, 1.1, 1.1, 1])
        qform_affine = np.diag([1.2, 1.2, 1.2, 1])
        # Reset image affine to something different again
        aff_affine = np.diag([3.3, 4.5, 6.6, 1])
        img.affine[:] = aff_affine
        assert_array_almost_equal(img.affine, aff_affine)
        # Sform, Qform codes are 'aligned',  'unknown' by default
        assert (hdr['sform_code'], hdr['qform_code']) == (2, 0)
        # Set sform using new_affine when qform is 0
        img.set_sform(new_affine, 1)
        assert hdr['sform_code'] == 1
        assert_array_almost_equal(hdr.get_sform(), new_affine)
        # Image get is same as header get
        assert_array_almost_equal(img.get_sform(), new_affine)
        # Coded version gives same result
        saff, code = img.get_sform(coded=True)
        assert code == 1
        assert_array_almost_equal(saff, new_affine)
        # Because we've reset the sform with update_affine, the affine changes
        assert_array_almost_equal(img.affine, hdr.get_best_affine())
        # Reset image affine and try update_affine == False
        img.affine[:] = aff_affine
        img.set_sform(new_affine, 1, update_affine=False)
        assert_array_almost_equal(img.affine, aff_affine)
        # zooms do not get updated when qform is 0
        assert_array_almost_equal(img.get_qform(), orig_aff)
        assert_array_almost_equal(hdr.get_zooms(), [2.2, 3.3, 4.3])
        img.set_qform(None)
        assert_array_almost_equal(hdr.get_zooms(), [2.2, 3.3, 4.3])
        # Set sform using new_affine when qform is set
        img.set_qform(qform_affine, 1)
        img.set_sform(new_affine, 1)
        saff, code = img.get_sform(coded=True)
        assert code == 1
        assert_array_almost_equal(saff, new_affine)
        assert_array_almost_equal(img.affine, new_affine)
        # zooms follow qform
        assert_array_almost_equal(hdr.get_zooms(), [1.2, 1.2, 1.2])
        # Clear sform using None, best_affine should fall back on qform
        img.set_sform(None)
        assert hdr['sform_code'] == 0
        assert hdr['qform_code'] == 1
        # Sform holds previous affine from last set
        assert_array_almost_equal(hdr.get_sform(), saff)
        # Image affine follows qform
        assert_array_almost_equal(img.affine, qform_affine)
        assert_array_almost_equal(hdr.get_best_affine(), img.affine)
        # Unexpected keyword raises error
        with pytest.raises(TypeError):
            img.get_sform(strange=True)
        # updating None affine should also work
        img = self.image_class(np.zeros((2, 3, 4)), None)
        new_affine = np.eye(4)
        img.set_sform(new_affine, 2)
        assert_array_almost_equal(img.affine, new_affine)

    def test_sqform_code_type(self):
        # make sure get_s/qform returns codes as integers
        img = self.image_class(np.zeros((2, 3, 4)), None)
        assert isinstance(img.get_sform(coded=True)[1], int)
        assert isinstance(img.get_qform(coded=True)[1], int)
        img.set_sform(None, 3)
        img.set_qform(None, 3)
        assert isinstance(img.get_sform(coded=True)[1], int)
        assert isinstance(img.get_qform(coded=True)[1], int)
        img.set_sform(None, 2.0)
        img.set_qform(None, 4.0)
        assert isinstance(img.get_sform(coded=True)[1], int)
        assert isinstance(img.get_qform(coded=True)[1], int)
        img.set_sform(None, img.get_sform(coded=True)[1])
        img.set_qform(None, img.get_qform(coded=True)[1])

    def test_hdr_diff(self):
        # Check an offset beyond data does not raise an error
        img = self.image_class(np.zeros((2, 3, 4)), np.eye(4))
        ext = dict(img.files_types)['image']
        hdr_len = len(img.header.binaryblock)
        img.header['vox_offset'] = hdr_len + 400
        with InTemporaryDirectory():
            img.to_filename('another_file' + ext)

    def test_load_save(self):
        IC = self.image_class
        img_ext = IC.files_types[0][1]
        shape = (2, 4, 6)
        npt = np.float32
        data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
        affine = np.diag([1, 2, 3, 1])
        img = IC(data, affine)
        assert img.header.get_data_offset() == 0
        assert img.shape == shape
        img.set_data_dtype(npt)
        img2 = bytesio_round_trip(img)
        assert_array_equal(img2.get_fdata(), data)
        with InTemporaryDirectory() as tmpdir:
            for ext in ('', '.gz', '.bz2'):
                fname = os.path.join(tmpdir, 'test' + img_ext + ext)
                img.to_filename(fname)
                img3 = IC.load(fname)
                assert isinstance(img3, img.__class__)
                assert_array_equal(img3.get_fdata(), data)
                assert img3.header == img.header
                assert isinstance(
                    np.asanyarray(img3.dataobj), np.memmap if ext == '' else np.ndarray
                )
                # del to avoid windows errors of form 'The process cannot
                # access the file because it is being used'
                del img3

    def test_load_pixdims(self):
        # Make sure load preserves separate qform, pixdims, sform
        IC = self.image_class
        HC = IC.header_class
        arr = np.arange(24).reshape((2, 3, 4))
        qaff = np.diag([2, 3, 4, 1])
        saff = np.diag([5, 6, 7, 1])
        hdr = HC()
        hdr.set_qform(qaff)
        assert_array_equal(hdr.get_qform(), qaff)
        hdr.set_sform(saff)
        assert_array_equal(hdr.get_sform(), saff)
        simg = IC(arr, None, hdr)
        img_hdr = simg.header
        # Check qform, sform, pixdims are the same
        assert_array_equal(img_hdr.get_qform(), qaff)
        assert_array_equal(img_hdr.get_sform(), saff)
        assert_array_equal(img_hdr.get_zooms(), [2, 3, 4])
        # Save to stringio
        re_simg = bytesio_round_trip(simg)
        assert_array_equal(re_simg.get_fdata(), arr)
        # Check qform, sform, pixdims are the same
        rimg_hdr = re_simg.header
        assert_array_equal(rimg_hdr.get_qform(), qaff)
        assert_array_equal(rimg_hdr.get_sform(), saff)
        assert_array_equal(rimg_hdr.get_zooms(), [2, 3, 4])

    def test_affines_init(self):
        # Test we are doing vaguely spec-related qform things.  The 'spec' here
        # is some thoughts by Mark Jenkinson:
        # http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/qsform_brief_usage
        IC = self.image_class
        arr = np.arange(24, dtype='f4').reshape((2, 3, 4))
        aff = np.diag([2, 3, 4, 1])
        # Default is sform set, qform not set
        img = IC(arr, aff)
        hdr = img.header
        assert hdr['qform_code'] == 0
        assert hdr['sform_code'] == 2
        assert_array_equal(hdr.get_zooms(), [2, 3, 4])
        # This is also true for affines with header passed
        qaff = np.diag([3, 4, 5, 1])
        saff = np.diag([6, 7, 8, 1])
        hdr.set_qform(qaff, code='scanner')
        hdr.set_sform(saff, code='talairach')
        assert_array_equal(hdr.get_zooms(), [3, 4, 5])
        img = IC(arr, aff, hdr)
        new_hdr = img.header
        # Again affine is sort of anonymous space
        assert new_hdr['qform_code'] == 0
        assert new_hdr['sform_code'] == 2
        assert_array_equal(new_hdr.get_sform(), aff)
        assert_array_equal(new_hdr.get_zooms(), [2, 3, 4])
        # But if no affine passed, codes and matrices stay the same
        img = IC(arr, None, hdr)
        new_hdr = img.header
        assert new_hdr['qform_code'] == 1  # scanner
        assert_array_equal(new_hdr.get_qform(), qaff)
        assert new_hdr['sform_code'] == 3  # Still talairach
        assert_array_equal(new_hdr.get_sform(), saff)
        # Pixdims as in the original header
        assert_array_equal(new_hdr.get_zooms(), [3, 4, 5])

    def test_read_no_extensions(self):
        IC = self.image_class
        arr = np.arange(24, dtype='f4').reshape((2, 3, 4))
        img = IC(arr, np.eye(4))
        assert len(img.header.extensions) == 0
        img_rt = bytesio_round_trip(img)
        assert len(img_rt.header.extensions) == 0
        # Check simple round trip with large offset
        img.header.set_data_offset(1024)
        img_rt = bytesio_round_trip(img)
        assert len(img_rt.header.extensions) == 0

    def _get_raw_scaling(self, hdr):
        return hdr['scl_slope'], hdr['scl_inter']

    def _set_raw_scaling(self, hdr, slope, inter):
        # Brutal set of slope and inter
        hdr['scl_slope'] = slope
        hdr['scl_inter'] = inter

    def test_write_scaling(self):
        # Check we can set slope, inter on write
        for slope, inter, e_slope, e_inter in (
            (1, 0, 1, 0),
            (2, 0, 2, 0),
            (2, 1, 2, 1),
            (0, 0, 1, 0),
            (np.inf, 0, 1, 0),
        ):
            with np.errstate(invalid='ignore'):
                self._check_write_scaling(slope, inter, e_slope, e_inter)

    def test_dynamic_dtype_aliases(self):
        for in_dt, mn, mx, alias, effective_dt in [
            (np.uint8, 0, 255, 'compat', np.uint8),
            (np.int8, 0, 127, 'compat', np.uint8),
            (np.int8, -128, 127, 'compat', np.int16),
            (np.int16, -32768, 32767, 'compat', np.int16),
            (np.uint16, 0, 32767, 'compat', np.int16),
            (np.uint16, 0, 65535, 'compat', np.int32),
            (np.int32, -(2**31), 2**31 - 1, 'compat', np.int32),
            (np.uint32, 0, 2**31 - 1, 'compat', np.int32),
            (np.uint32, 0, 2**32 - 1, 'compat', None),
            (np.int64, -(2**31), 2**31 - 1, 'compat', np.int32),
            (np.uint64, 0, 2**31 - 1, 'compat', np.int32),
            (np.int64, 0, 2**32 - 1, 'compat', None),
            (np.uint64, 0, 2**32 - 1, 'compat', None),
            (np.float32, 0, 1e30, 'compat', np.float32),
            (np.float64, 0, 1e30, 'compat', np.float32),
            (np.float64, 0, 1e40, 'compat', None),
            (np.int64, 0, 255, 'smallest', np.uint8),
            (np.int64, 0, 256, 'smallest', np.int16),
            (np.int64, -1, 255, 'smallest', np.int16),
            (np.int64, 0, 32768, 'smallest', np.int32),
            (np.int64, 0, 4294967296, 'smallest', None),
            (np.float32, 0, 1, 'smallest', None),
            (np.float64, 0, 1, 'smallest', None),
        ]:
            arr = np.arange(24, dtype=in_dt).reshape((2, 3, 4))
            arr[0, 0, :2] = [mn, mx]
            img = self.image_class(arr, np.eye(4), dtype=alias)
            # Stored as alias
            assert img.get_data_dtype() == alias
            if effective_dt is None:
                with pytest.raises(ValueError):
                    img.get_data_dtype(finalize=True)
                continue
            # Finalizing sets and clears the alias
            assert img.get_data_dtype(finalize=True) == effective_dt
            assert img.get_data_dtype() == effective_dt
            # Re-set to alias
            img.set_data_dtype(alias)
            assert img.get_data_dtype() == alias
            img_rt = bytesio_round_trip(img)
            assert img_rt.get_data_dtype() == effective_dt
            # Serializing does not finalize the source image
            assert img.get_data_dtype() == alias

    def test_static_dtype_aliases(self):
        for alias, effective_dt in [
            ('mask', np.uint8),
        ]:
            for orig_dt in ('u1', 'i8', 'f4'):
                arr = np.arange(24, dtype=orig_dt).reshape((2, 3, 4))
                img = self.image_class(arr, np.eye(4), dtype=alias)
                assert img.get_data_dtype() == effective_dt
                img_rt = bytesio_round_trip(img)
                assert img_rt.get_data_dtype() == effective_dt


class TestNifti1Image(TestNifti1Pair):
    # Run analyze-flavor spatialimage tests
    image_class = Nifti1Image

    def test_offset_errors(self):
        # Test that explicit offset too low raises error
        IC = self.image_class
        arr = np.arange(24, dtype='f4').reshape((2, 3, 4))
        img = IC(arr, np.eye(4))
        assert img.header.get_data_offset() == 0
        # Saving with zero offset is OK
        img_rt = bytesio_round_trip(img)
        assert img_rt.header.get_data_offset() == 0
        # Saving with too low offset explicitly set gives error
        fm = bytesio_filemap(IC)
        img.header.set_data_offset(16)
        with pytest.raises(HeaderDataError):
            img.to_file_map(fm)


def test_extension_basics():
    raw = '123'
    ext = Nifti1Extension('comment', raw)
    assert ext.get_sizeondisk() == 16
    assert ext.get_content() == raw
    assert ext.get_code() == 6
    # Test that extensions already aligned to 16 bytes are not padded
    ext = Nifti1Extension('comment', b'x' * 24)
    assert ext.get_sizeondisk() == 32


def test_ext_eq():
    ext = Nifti1Extension('comment', '123')
    assert ext == ext
    assert not ext != ext
    ext2 = Nifti1Extension('comment', '124')
    assert ext != ext2
    assert not ext == ext2


def test_extension_content_access():
    ext = Nifti1Extension('comment', b'123')
    # Unmangled content access
    assert ext.get_content() == b'123'

    # Raw, text and JSON access
    assert ext.content == b'123'
    assert ext.text == '123'
    assert ext.json() == 123

    # Encoding can be set
    ext.encoding = 'ascii'
    assert ext.text == '123'

    # Test that encoding errors are caught
    ascii_ext = Nifti1Extension('comment', 'hôpital'.encode())
    ascii_ext.encoding = 'ascii'
    with pytest.raises(UnicodeDecodeError):
        ascii_ext.text

    json_ext = Nifti1Extension('unknown', b'{"a": 1}')
    assert json_ext.content == b'{"a": 1}'
    assert json_ext.text == '{"a": 1}'
    assert json_ext.json() == {'a': 1}


def test_legacy_underscore_content():
    """Verify that subclasses that depended on access to ._content continue to work."""
    import io
    import json

    class MyLegacyExtension(Nifti1Extension):
        def _mangle(self, value):
            return json.dumps(value).encode()

        def _unmangle(self, value):
            if isinstance(value, bytes):
                value = value.decode()
            return json.loads(value)

    ext = MyLegacyExtension(0, '{}')

    assert isinstance(ext._content, dict)
    # Object identity is not broken by multiple accesses
    assert ext._content is ext._content

    ext._content['val'] = 1

    fobj = io.BytesIO()
    ext.write_to(fobj)
    assert fobj.getvalue() == b'\x20\x00\x00\x00\x00\x00\x00\x00{"val": 1}' + bytes(14)


def test_extension_codes():
    for k in extension_codes.keys():
        Nifti1Extension(k, 'somevalue')


def test_extension_list():
    ext_c0 = Nifti1Extensions()
    ext_c1 = Nifti1Extensions()
    assert ext_c0 == ext_c1
    ext = Nifti1Extension('comment', '123')
    ext_c1.append(ext)
    assert not ext_c0 == ext_c1
    ext_c0.append(ext)
    assert ext_c0 == ext_c1


def test_extension_io():
    bio = BytesIO()
    ext1 = Nifti1Extension(6, b'Extra comment')
    ext1.write_to(bio, False)
    bio.seek(0)
    ebacks = Nifti1Extensions.from_fileobj(bio, -1, False)
    assert len(ebacks) == 1
    assert ext1 == ebacks[0]
    # Check the start is what we expect
    exp_dtype = np.dtype([('esize', 'i4'), ('ecode', 'i4')])
    bio.seek(0)
    buff = np.ndarray(shape=(), dtype=exp_dtype, buffer=bio.read(16))
    assert buff['esize'] == 32
    assert buff['ecode'] == 6
    # Try another extension on top
    bio.seek(32)
    ext2 = Nifti1Extension(6, b'Comment')
    ext2.write_to(bio, False)
    bio.seek(0)
    ebacks = Nifti1Extensions.from_fileobj(bio, -1, False)
    assert len(ebacks) == 2
    assert ext1 == ebacks[0]
    assert ext2 == ebacks[1]
    # Rewrite but deliberately setting esize wrongly
    bio.truncate(0)
    bio.seek(0)
    ext1.write_to(bio, False)
    bio.seek(0)
    start = np.zeros((1,), dtype=exp_dtype)
    start['esize'] = 24
    start['ecode'] = 6
    bio.write(start.tobytes())
    bio.seek(24)
    ext2.write_to(bio, False)
    # Result should still be OK, but with a warning
    bio.seek(0)
    with warnings.catch_warnings(record=True) as warns:
        ebacks = Nifti1Extensions.from_fileobj(bio, -1, False)
        assert len(warns) == 1
        assert warns[0].category == UserWarning
        assert len(ebacks) == 2
        assert ext1 == ebacks[0]
        assert ext2 == ebacks[1]


def test_nifti_extensions():
    nim = load(image_file)
    # basic checks of the available extensions
    hdr = nim.header
    exts_container = hdr.extensions
    assert len(exts_container) == 2
    assert exts_container.count('comment') == 2
    assert exts_container.count('afni') == 0
    assert exts_container.get_codes() == [6, 6]
    assert (exts_container.get_sizeondisk()) % 16 == 0
    # first extension should be short one
    assert exts_container[0].get_content() == b'extcomment1'
    # add one
    afniext = Nifti1Extension('afni', '<xml></xml>')
    exts_container.append(afniext)
    assert exts_container.get_codes() == [6, 6, 4]
    assert exts_container.count('comment') == 2
    assert exts_container.count('afni') == 1
    assert (exts_container.get_sizeondisk()) % 16 == 0
    # delete one
    del exts_container[1]
    assert exts_container.get_codes() == [6, 4]
    assert exts_container.count('comment') == 1
    assert exts_container.count('afni') == 1


@dicom_test
def test_nifti_dicom_extension():
    nim = load(image_file)
    hdr = nim.header
    exts_container = hdr.extensions

    # create an empty dataset if no content provided (to write a new header)
    dcmext = Nifti1DicomExtension(2, b'')
    assert dcmext.get_content().__class__ == pydicom.dataset.Dataset
    assert len(dcmext.get_content().values()) == 0

    # create an empty dataset if no content provided (to write a new header)
    dcmext = Nifti1DicomExtension(2, None)
    assert dcmext.get_content().__class__ == pydicom.dataset.Dataset
    assert len(dcmext.get_content().values()) == 0

    # use a dataset if provided
    ds = pydicom.dataset.Dataset()
    ds.add_new((0x10, 0x20), 'LO', 'NiPy')
    dcmext = Nifti1DicomExtension(2, ds)
    assert dcmext.get_content().__class__ == pydicom.dataset.Dataset
    assert len(dcmext.get_content().values()) == 1
    assert dcmext.get_content().PatientID == 'NiPy'

    # create a single dicom tag (Patient ID, [0010,0020]) with Explicit VR / LE
    dcmbytes_explicit = struct.pack('<HH2sH4s', 0x10, 0x20, b'LO', 4, b'NiPy')
    dcmext = Nifti1DicomExtension(2, dcmbytes_explicit)
    assert dcmext.__class__ == Nifti1DicomExtension
    assert dcmext._is_implicit_VR is False
    assert dcmext._is_little_endian is True
    assert dcmext.get_code() == 2
    assert dcmext.get_content().PatientID == 'NiPy'
    assert len(dcmext.get_content().values()) == 1
    assert dcmext._mangle(dcmext.get_content()) == dcmbytes_explicit
    assert dcmext.get_sizeondisk() % 16 == 0

    # create a single dicom tag (Patient ID, [0010,0020]) with Implicit VR
    dcmbytes_implicit = struct.pack('<HHL4s', 0x10, 0x20, 4, b'NiPy')
    dcmext = Nifti1DicomExtension(2, dcmbytes_implicit)
    assert dcmext._is_implicit_VR is True
    assert dcmext.get_code() == 2
    assert dcmext.get_content().PatientID == 'NiPy'
    assert len(dcmext.get_content().values()) == 1
    assert dcmext._mangle(dcmext.get_content()) == dcmbytes_implicit
    assert dcmext.get_sizeondisk() % 16 == 0

    # create a single dicom tag (Patient ID, [0010,0020]) with Explicit VR / BE
    dcmbytes_explicit_be = struct.pack('>2H2sH4s', 0x10, 0x20, b'LO', 4, b'NiPy')
    hdr_be = Nifti1Header(endianness='>')  # Big Endian Nifti1Header
    dcmext = Nifti1DicomExtension(2, dcmbytes_explicit_be, parent_hdr=hdr_be)
    assert dcmext.__class__ == Nifti1DicomExtension
    assert dcmext._is_implicit_VR is False
    assert dcmext.get_code() == 2
    assert dcmext.get_content().PatientID == 'NiPy'
    assert dcmext.get_content()[0x10, 0x20].value == 'NiPy'
    assert len(dcmext.get_content().values()) == 1
    assert dcmext._mangle(dcmext.get_content()) == dcmbytes_explicit_be
    assert dcmext.get_sizeondisk() % 16 == 0

    # Check that a dicom dataset is written w/ BE encoding when not created
    # using BE bytestring when given a BE nifti header
    dcmext = Nifti1DicomExtension(2, ds, parent_hdr=hdr_be)
    assert dcmext._mangle(dcmext.get_content()) == dcmbytes_explicit_be

    # dicom extension access from nifti extensions
    assert exts_container.count('dicom') == 0
    exts_container.append(dcmext)
    assert exts_container.count('dicom') == 1
    assert exts_container.get_codes() == [6, 6, 2]
    assert dcmext._mangle(dcmext.get_content()) == dcmbytes_explicit_be
    assert dcmext.get_sizeondisk() % 16 == 0

    # creating an extension with bad content should raise
    with pytest.raises(TypeError):
        Nifti1DicomExtension(2, 0)


class TestNifti1General:
    """Test class to test nifti1 in general

    Tests here which mix the pair and the single type, and that should only be
    run once (not for each type) because they are slow
    """

    single_class = Nifti1Image
    pair_class = Nifti1Pair
    module = nifti1
    example_file = image_file

    def test_loadsave_cycle(self):
        nim = self.module.load(self.example_file)
        # ensure we have extensions
        hdr = nim.header
        exts_container = hdr.extensions
        assert len(exts_container) > 0
        # write into the air ;-)
        lnim = bytesio_round_trip(nim)
        hdr = lnim.header
        lexts_container = hdr.extensions
        assert exts_container == lexts_container
        # build int16 image
        data = np.ones((2, 3, 4, 5), dtype='int16')
        img = self.single_class(data, np.eye(4))
        hdr = img.header
        assert hdr.get_data_dtype() == np.int16
        # default should have no scaling
        assert_array_equal(hdr.get_slope_inter(), (None, None))
        # set scaling
        hdr.set_slope_inter(2, 8)
        assert hdr.get_slope_inter() == (2, 8)
        # now build new image with updated header
        wnim = self.single_class(data, np.eye(4), header=hdr)
        assert wnim.get_data_dtype() == np.int16
        # Header scaling reset to default by image creation
        assert wnim.header.get_slope_inter() == (None, None)
        # But we can reset it again after image creation
        wnim.header.set_slope_inter(2, 8)
        assert wnim.header.get_slope_inter() == (2, 8)
        # write into the air again ;-)
        lnim = bytesio_round_trip(wnim)
        assert lnim.get_data_dtype() == np.int16
        # Scaling applied
        assert_array_equal(lnim.get_fdata(), data * 2.0 + 8.0)
        # slope, inter reset by image creation, but saved in proxy
        assert lnim.header.get_slope_inter() == (None, None)
        assert (lnim.dataobj.slope, lnim.dataobj.inter) == (2, 8)

    def test_load(self):
        # test module level load.  We try to load a nii and an .img and a .hdr
        # and expect to get a nifti back of single or pair type
        arr = np.arange(24, dtype='f4').reshape((2, 3, 4))
        aff = np.diag([2, 3, 4, 1])
        simg = self.single_class(arr, aff)
        pimg = self.pair_class(arr, aff)
        save = self.module.save
        load = self.module.load
        with InTemporaryDirectory():
            for img in (simg, pimg):
                save(img, 'test.nii')
                assert_array_equal(arr, load('test.nii').get_fdata())
                save(simg, 'test.img')
                assert_array_equal(arr, load('test.img').get_fdata())
                save(simg, 'test.hdr')
                assert_array_equal(arr, load('test.hdr').get_fdata())

    def test_float_int_min_max(self):
        # Conversion between float and int
        # Parallel test to arraywriters
        aff = np.eye(4)
        for in_dt in (np.float32, np.float64):
            finf = type_info(in_dt)
            arr = np.array([finf['min'], finf['max']], dtype=in_dt)
            for out_dt in IUINT_TYPES:
                img = self.single_class(arr, aff)
                img_back = bytesio_round_trip(img)
                arr_back_sc = img_back.get_fdata()
                assert np.allclose(arr, arr_back_sc)

    def test_float_int_spread(self):
        # Test rounding error for spread of values
        # Parallel test to arraywriters
        powers = np.arange(-10, 10, 0.5)
        arr = np.concatenate((-(10**powers), 10**powers))
        aff = np.eye(4)
        for in_dt in (np.float32, np.float64):
            arr_t = arr.astype(in_dt)
            for out_dt in IUINT_TYPES:
                img = self.single_class(arr_t, aff)
                img_back = bytesio_round_trip(img)
                arr_back_sc = img_back.get_fdata()
                slope, inter = img_back.header.get_slope_inter()
                # Get estimate for error
                max_miss = rt_err_estimate(arr_t, arr_back_sc.dtype, slope, inter)
                # Simulate allclose test with large atol
                diff = np.abs(arr_t - arr_back_sc)
                rdiff = diff / np.abs(arr_t)
                assert np.all((diff <= max_miss) | (rdiff <= 1e-5))

    def test_rt_bias(self):
        # Check for bias in round trip
        # Parallel test to arraywriters
        rng = np.random.RandomState(20111214)
        mu, std, count = 100, 10, 100
        arr = rng.normal(mu, std, size=(count,))
        eps = np.finfo(np.float32).eps
        aff = np.eye(4)
        for in_dt in (np.float32, np.float64):
            arr_t = arr.astype(in_dt)
            for out_dt in IUINT_TYPES:
                img = self.single_class(arr_t, aff)
                img_back = bytesio_round_trip(img)
                arr_back_sc = img_back.get_fdata()
                slope, inter = img_back.header.get_slope_inter()
                bias = np.mean(arr_t - arr_back_sc)
                # Get estimate for error
                max_miss = rt_err_estimate(arr_t, arr_back_sc.dtype, slope, inter)
                # Hokey use of max_miss as a std estimate
                bias_thresh = np.max([max_miss / np.sqrt(count), eps])
                assert np.abs(bias) < bias_thresh

    def test_reoriented_dim_info(self):
        # Check that dim_info is reoriented correctly
        arr = np.arange(24, dtype='f4').reshape((2, 3, 4))
        # Start as RAS
        aff = np.diag([2, 3, 4, 1])
        simg = self.single_class(arr, aff)
        for freq, phas, slic in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (2, 0, 1),
            (None, None, None),
            (0, 2, None),
            (0, None, None),
            (None, 2, 1),
            (None, None, 1),
        ):
            simg.header.set_dim_info(freq, phas, slic)
            fdir = 'RAS'[freq] if freq is not None else None
            pdir = 'RAS'[phas] if phas is not None else None
            sdir = 'RAS'[slic] if slic is not None else None
            for ornt in ALL_ORNTS:
                rimg = simg.as_reoriented(np.array(ornt))
                axcode = aff2axcodes(rimg.affine)
                dirs = ''.join(axcode).replace('P', 'A').replace('I', 'S').replace('L', 'R')
                new_freq, new_phas, new_slic = rimg.header.get_dim_info()
                new_fdir = dirs[new_freq] if new_freq is not None else None
                new_pdir = dirs[new_phas] if new_phas is not None else None
                new_sdir = dirs[new_slic] if new_slic is not None else None
                assert (new_fdir, new_pdir, new_sdir) == (fdir, pdir, sdir)


@runif_extra_has('slow')
def test_large_nifti1():
    image_shape = (91, 109, 91, 1200)
    img = Nifti1Image(np.ones(image_shape, dtype=np.float32), affine=np.eye(4))
    # Dump and load the large image.
    with InTemporaryDirectory():
        img.to_filename('test.nii.gz')
        del img
        data = load('test.nii.gz').get_fdata()
    # Check that the data are all ones
    assert image_shape == data.shape
    n_ones = np.sum(data == 1.0)
    assert np.prod(image_shape) == n_ones
