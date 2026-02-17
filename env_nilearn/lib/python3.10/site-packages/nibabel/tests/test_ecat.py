# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import os
from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ..ecat import (
    EcatHeader,
    EcatImage,
    EcatSubHeader,
    get_frame_order,
    get_series_framenumbers,
    read_mlist,
)
from ..openers import Opener
from ..testing import data_path, suppress_warnings
from ..tmpdirs import InTemporaryDirectory
from . import test_wrapstruct as tws
from .test_fileslice import slicer_samples

ecat_file = os.path.join(data_path, 'tinypet.v')


class TestEcatHeader(tws._TestWrapStructBase):
    header_class = EcatHeader
    example_file = ecat_file

    def test_header_size(self):
        assert self.header_class.template_dtype.itemsize == 512

    def test_empty(self):
        hdr = self.header_class()
        assert len(hdr.binaryblock) == 512
        assert hdr['magic_number'] == b'MATRIX72'
        assert hdr['sw_version'] == 74
        assert hdr['num_frames'] == 0
        assert hdr['file_type'] == 0
        assert hdr['ecat_calibration_factor'] == 1.0

    def _set_something_into_hdr(self, hdr):
        # Called from test_bytes test method.  Specific to the header data type
        hdr['scan_start_time'] = 42

    def test_dtype(self):
        # dtype not specified in header, only in subheaders
        hdr = self.header_class()
        with pytest.raises(NotImplementedError):
            hdr.get_data_dtype()

    def test_header_codes(self):
        fid = open(self.example_file, 'rb')
        hdr = self.header_class()
        newhdr = hdr.from_fileobj(fid)
        fid.close()
        assert newhdr.get_filetype() == 'ECAT7_VOLUME16'
        assert newhdr.get_patient_orient() == 'ECAT7_Unknown_Orientation'

    def test_update(self):
        hdr = self.header_class()
        assert hdr['num_frames'] == 0
        hdr['num_frames'] = 2
        assert hdr['num_frames'] == 2

    def test_from_eg_file(self):
        # Example header is big-endian
        with Opener(self.example_file) as fileobj:
            hdr = self.header_class.from_fileobj(fileobj, check=False)
        assert hdr.endianness == '>'


class TestEcatMlist(TestCase):
    header_class = EcatHeader
    example_file = ecat_file

    def test_mlist(self):
        fid = open(self.example_file, 'rb')
        hdr = self.header_class.from_fileobj(fid)
        mlist = read_mlist(fid, hdr.endianness)
        fid.seek(0)
        fid.seek(512)
        dat = fid.read(128 * 32)
        dt = np.dtype([('matlist', np.int32)])
        dt = dt.newbyteorder('>')
        mats = np.recarray(shape=(32, 4), dtype=dt, buf=dat)
        fid.close()
        # tests
        assert mats['matlist'][0, 0] + mats['matlist'][0, 3] == 31
        assert get_frame_order(mlist)[0][0] == 0
        assert get_frame_order(mlist)[0][1] == 16842758.0
        # test badly ordered mlist
        badordermlist = np.array(
            [
                [1.68427540e07, 3.00000000e00, 1.20350000e04, 1.00000000e00],
                [1.68427530e07, 1.20360000e04, 2.40680000e04, 1.00000000e00],
                [1.68427550e07, 2.40690000e04, 3.61010000e04, 1.00000000e00],
                [1.68427560e07, 3.61020000e04, 4.81340000e04, 1.00000000e00],
                [1.68427570e07, 4.81350000e04, 6.01670000e04, 1.00000000e00],
                [1.68427580e07, 6.01680000e04, 7.22000000e04, 1.00000000e00],
            ]
        )
        with suppress_warnings():  # STORED order
            assert get_frame_order(badordermlist)[0][0] == 1

    def test_mlist_errors(self):
        fid = open(self.example_file, 'rb')
        hdr = self.header_class.from_fileobj(fid)
        hdr['num_frames'] = 6
        mlist = read_mlist(fid, hdr.endianness)
        fid.close()
        mlist = np.array(
            [
                [1.68427540e07, 3.00000000e00, 1.20350000e04, 1.00000000e00],
                [1.68427530e07, 1.20360000e04, 2.40680000e04, 1.00000000e00],
                [1.68427550e07, 2.40690000e04, 3.61010000e04, 1.00000000e00],
                [1.68427560e07, 3.61020000e04, 4.81340000e04, 1.00000000e00],
                [1.68427570e07, 4.81350000e04, 6.01670000e04, 1.00000000e00],
                [1.68427580e07, 6.01680000e04, 7.22000000e04, 1.00000000e00],
            ]
        )
        with suppress_warnings():  # STORED order
            series_framenumbers = get_series_framenumbers(mlist)
        # first frame stored was actually 2nd frame acquired
        assert series_framenumbers[0] == 2
        order = [series_framenumbers[x] for x in sorted(series_framenumbers)]
        # true series order is [2,1,3,4,5,6], note counting starts at 1
        assert order == [2, 1, 3, 4, 5, 6]
        mlist[0, 0] = 0
        with suppress_warnings():
            frames_order = get_frame_order(mlist)
        neworder = [frames_order[x][0] for x in sorted(frames_order)]
        assert neworder == [1, 2, 3, 4, 5]
        with suppress_warnings():
            with pytest.raises(OSError):
                get_series_framenumbers(mlist)


class TestEcatSubHeader(TestCase):
    header_class = EcatHeader
    subhdr_class = EcatSubHeader
    example_file = ecat_file
    fid = open(example_file, 'rb')
    hdr = header_class.from_fileobj(fid)
    mlist = read_mlist(fid, hdr.endianness)
    subhdr = subhdr_class(hdr, mlist, fid)

    def test_subheader_size(self):
        assert self.subhdr_class._subhdrdtype.itemsize == 510

    def test_subheader(self):
        assert self.subhdr.get_shape() == (10, 10, 3)
        assert self.subhdr.get_nframes() == 1
        assert self.subhdr.get_nframes() == len(self.subhdr.subheaders)
        assert self.subhdr._check_affines() is True
        assert_array_almost_equal(
            np.diag(self.subhdr.get_frame_affine()), np.array([2.20241979, 2.20241979, 3.125, 1.0])
        )
        assert self.subhdr.get_zooms()[0] == 2.20241978764534
        assert self.subhdr.get_zooms()[2] == 3.125
        assert self.subhdr._get_data_dtype(0) == np.int16
        # assert_equal(self.subhdr._get_frame_offset(), 1024)
        assert self.subhdr._get_frame_offset() == 1536
        dat = self.subhdr.raw_data_from_fileobj()
        assert dat.shape == self.subhdr.get_shape()
        assert self.subhdr.subheaders[0]['scale_factor'].item() == 1.0
        ecat_calib_factor = self.hdr['ecat_calibration_factor']
        assert ecat_calib_factor == 25007614.0


class TestEcatImage(TestCase):
    image_class = EcatImage
    example_file = ecat_file
    img = image_class.load(example_file)

    def test_file(self):
        assert Path(self.img.file_map['header'].filename) == Path(self.example_file)
        assert Path(self.img.file_map['image'].filename) == Path(self.example_file)

    def test_save(self):
        tmp_file = 'tinypet_tmp.v'
        with InTemporaryDirectory():
            self.img.to_filename(tmp_file)
            other = self.image_class.load(tmp_file)
            assert_array_equal(self.img.get_fdata(), other.get_fdata())
            # Delete object holding reference to temporary file to make Windows
            # happier.
            del other

    def test_data(self):
        dat = self.img.get_fdata()
        assert dat.shape == self.img.shape
        frame = self.img.get_frame(0)
        assert_array_equal(frame, dat[:, :, :, 0])

    def test_array_proxy(self):
        # Get the cached data copy
        dat = self.img.get_fdata()
        # Make a new one to test arrayproxy
        img = self.image_class.load(self.example_file)
        data_prox = img.dataobj
        data2 = np.array(data_prox)
        assert_array_equal(data2, dat)
        # Check it rereads
        data3 = np.array(data_prox)
        assert_array_equal(data3, dat)

    def test_array_proxy_slicing(self):
        # Test slicing of array proxy
        arr = self.img.get_fdata()
        prox = self.img.dataobj
        assert prox.is_proxy
        for sliceobj in slicer_samples(self.img.shape):
            assert_array_equal(arr[sliceobj], prox[sliceobj])

    def test_isolation(self):
        # Test image isolated from external changes to affine
        img_klass = self.image_class
        arr, aff, hdr, sub_hdr, mlist = (
            self.img.get_fdata(),
            self.img.affine,
            self.img.header,
            self.img.get_subheaders(),
            self.img.get_mlist(),
        )
        img = img_klass(arr, aff, hdr, sub_hdr, mlist)
        assert_array_equal(img.affine, aff)
        aff[0, 0] = 99
        assert not np.all(img.affine == aff)

    def test_float_affine(self):
        # Check affines get converted to float
        img_klass = self.image_class
        arr, aff, hdr, sub_hdr, mlist = (
            self.img.get_fdata(),
            self.img.affine,
            self.img.header,
            self.img.get_subheaders(),
            self.img.get_mlist(),
        )
        img = img_klass(arr, aff.astype(np.float32), hdr, sub_hdr, mlist)
        assert img.affine.dtype == np.dtype(np.float64)
        img = img_klass(arr, aff.astype(np.int16), hdr, sub_hdr, mlist)
        assert img.affine.dtype == np.dtype(np.float64)

    def test_data_regression(self):
        # Test whether data read has changed since 1.3.0
        # These values came from reading the example image using nibabel 1.3.0
        vals = dict(max=248750736458.0, min=1125342630.0, mean=117907565661.46666)
        data = self.img.get_fdata()
        assert data.max() == vals['max']
        assert data.min() == vals['min']
        assert_array_almost_equal(data.mean(), vals['mean'])

    def test_mlist_regression(self):
        # Test mlist is as same as for nibabel 1.3.0
        assert_array_equal(self.img.get_mlist(), [[16842758, 3, 3011, 1]])
