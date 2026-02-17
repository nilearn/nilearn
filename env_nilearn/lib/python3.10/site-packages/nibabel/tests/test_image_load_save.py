# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for loader function"""

import logging
import pathlib
import shutil
from io import BytesIO
from os.path import dirname
from os.path import join as pjoin
from tempfile import mkdtemp

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from .. import (
    AnalyzeImage,
    MGHImage,
    Minc1Image,
    Minc2Image,
    Nifti1Image,
    Nifti1Pair,
    Nifti2Image,
    Nifti2Pair,
    Spm2AnalyzeImage,
    Spm99AnalyzeImage,
    all_image_classes,
)
from .. import analyze as ana
from .. import loadsave as nils
from .. import nifti1 as ni1
from .. import spm2analyze as spm2
from .. import spm99analyze as spm99
from ..optpkg import optional_package
from ..spatialimages import SpatialImage
from ..testing import deprecated_to, expires
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import native_code, swapped_code

_, have_scipy, _ = optional_package('scipy')  # No scipy=>no SPM-format writing
DATA_PATH = pjoin(dirname(__file__), 'data')
MGH_DATA_PATH = pjoin(dirname(__file__), '..', 'freesurfer', 'tests', 'data')


def round_trip(img):
    # round trip a nifti single
    return Nifti1Image.from_bytes(img.to_bytes())


def test_conversion_spatialimages(caplog):
    shape = (2, 4, 6)
    affine = np.diag([1, 2, 3, 1])
    klasses = [
        klass for klass in all_image_classes if klass.rw and issubclass(klass, SpatialImage)
    ]
    for npt in np.float32, np.int16:
        data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
        for r_class in klasses:
            if not r_class.makeable:
                continue
            img = r_class(data, affine)
            img.set_data_dtype(npt)
            for w_class in klasses:
                if not w_class.makeable:
                    continue
                # Suppress header field mismatch reports
                with caplog.at_level(logging.CRITICAL):
                    img2 = w_class.from_image(img)
                assert_array_equal(img2.get_fdata(), data)
                assert_array_equal(img2.affine, affine)


def test_save_load_endian():
    shape = (2, 4, 6)
    affine = np.diag([1, 2, 3, 1])
    data = np.arange(np.prod(shape), dtype='f4').reshape(shape)
    # Native endian image
    img = Nifti1Image(data, affine)
    assert img.header.endianness == native_code
    img2 = round_trip(img)
    assert img2.header.endianness == native_code
    assert_array_equal(img2.get_fdata(), data)
    assert_array_equal(np.asanyarray(img2.dataobj), data)
    # byte swapped endian image
    bs_hdr = img.header.as_byteswapped()
    bs_img = Nifti1Image(data, affine, bs_hdr)
    assert bs_img.header.endianness == swapped_code
    # of course the data is the same because it's not written to disk
    assert_array_equal(bs_img.get_fdata(), data)
    assert_array_equal(np.asanyarray(bs_img.dataobj), data)
    # Check converting to another image
    cbs_img = AnalyzeImage.from_image(bs_img)
    # this will make the header native by doing the header conversion
    cbs_hdr = cbs_img.header
    assert cbs_hdr.endianness == native_code
    # and the byte order follows it back into another image
    cbs_img2 = Nifti1Image.from_image(cbs_img)
    cbs_hdr2 = cbs_img2.header
    assert cbs_hdr2.endianness == native_code
    # Try byteswapped round trip
    bs_img2 = round_trip(bs_img)
    bs_data2 = np.asanyarray(bs_img2.dataobj)
    bs_fdata2 = bs_img2.get_fdata()
    # now the data dtype was swapped endian, so the read data is too
    assert bs_data2.dtype.byteorder == swapped_code
    assert bs_img2.header.endianness == swapped_code
    assert_array_equal(bs_data2, data)
    # but get_fdata uses native endian
    assert bs_fdata2.dtype.byteorder != swapped_code
    assert_array_equal(bs_fdata2, data)
    # Now mix up byteswapped data and non-byteswapped header
    mixed_img = Nifti1Image(bs_data2, affine)
    assert mixed_img.header.endianness == native_code
    m_img2 = round_trip(mixed_img)
    assert m_img2.header.endianness == native_code
    assert_array_equal(m_img2.get_fdata(), data)


def test_save_load():
    shape = (2, 4, 6)
    npt = np.float32
    data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
    affine = np.diag([1, 2, 3, 1])
    affine[:3, 3] = [3, 2, 1]
    img = ni1.Nifti1Image(data, affine)
    img.set_data_dtype(npt)
    with InTemporaryDirectory():
        nifn = 'an_image.nii'
        sifn = 'another_image.img'
        ni1.save(img, nifn)
        re_img = nils.load(nifn)
        assert isinstance(re_img, ni1.Nifti1Image)
        assert_array_equal(re_img.get_fdata(), data)
        assert_array_equal(re_img.affine, affine)
        # These and subsequent del statements are to prevent confusing
        # windows errors when trying to open files or delete the
        # temporary directory.
        del re_img
        if have_scipy:  # skip we we cannot read .mat files
            spm2.save(img, sifn)
            re_img2 = nils.load(sifn)
            assert isinstance(re_img2, spm2.Spm2AnalyzeImage)
            assert_array_equal(re_img2.get_fdata(), data)
            assert_array_equal(re_img2.affine, affine)
            del re_img2
            spm99.save(img, sifn)
            re_img3 = nils.load(sifn)
            assert isinstance(re_img3, spm99.Spm99AnalyzeImage)
            assert_array_equal(re_img3.get_fdata(), data)
            assert_array_equal(re_img3.affine, affine)
            ni1.save(re_img3, nifn)
            del re_img3
        re_img = nils.load(nifn)
        assert isinstance(re_img, ni1.Nifti1Image)
        assert_array_equal(re_img.get_fdata(), data)
        assert_array_equal(re_img.affine, affine)
        del re_img


def test_two_to_one():
    # test going from two to one file in save
    shape = (2, 4, 6)
    npt = np.float32
    data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
    affine = np.diag([1, 2, 3, 1])
    affine[:3, 3] = [3, 2, 1]
    # single file format
    img = ni1.Nifti1Image(data, affine)
    assert img.header['magic'] == b'n+1'
    str_io = BytesIO()
    img.file_map['image'].fileobj = str_io
    # check that the single format vox offset stays at zero
    img.to_file_map()
    assert img.header['magic'] == b'n+1'
    assert img.header['vox_offset'] == 0
    # make a new pair image, with the single image header
    pimg = ni1.Nifti1Pair(data, affine, img.header)
    isio = BytesIO()
    hsio = BytesIO()
    pimg.file_map['image'].fileobj = isio
    pimg.file_map['header'].fileobj = hsio
    pimg.to_file_map()
    # the offset stays at zero (but is 352 on disk)
    assert pimg.header['magic'] == b'ni1'
    assert pimg.header['vox_offset'] == 0
    assert_array_equal(pimg.get_fdata(), data)
    # same for from_image, going from single image to pair format
    ana_img = ana.AnalyzeImage.from_image(img)
    assert ana_img.header['vox_offset'] == 0
    # back to the single image, save it again to a stringio
    str_io = BytesIO()
    img.file_map['image'].fileobj = str_io
    img.to_file_map()
    assert img.header['vox_offset'] == 0
    aimg = ana.AnalyzeImage.from_image(img)
    assert aimg.header['vox_offset'] == 0
    aimg = spm99.Spm99AnalyzeImage.from_image(img)
    assert aimg.header['vox_offset'] == 0
    aimg = spm2.Spm2AnalyzeImage.from_image(img)
    assert aimg.header['vox_offset'] == 0
    nfimg = ni1.Nifti1Pair.from_image(img)
    assert nfimg.header['vox_offset'] == 0
    # now set the vox offset directly
    hdr = nfimg.header
    hdr['vox_offset'] = 16
    assert nfimg.header['vox_offset'] == 16
    # check it gets properly set by the nifti single image
    nfimg = ni1.Nifti1Image.from_image(img)
    assert nfimg.header['vox_offset'] == 0


def test_negative_load_save():
    shape = (1, 2, 5)
    data = np.arange(10).reshape(shape) - 10.0
    affine = np.eye(4)
    hdr = ni1.Nifti1Header()
    hdr.set_data_dtype(np.int16)
    img = Nifti1Image(data, affine, hdr)
    str_io = BytesIO()
    img.file_map['image'].fileobj = str_io
    img.to_file_map()
    str_io.seek(0)
    re_img = Nifti1Image.from_file_map(img.file_map)
    assert_array_almost_equal(re_img.get_fdata(), data, 4)


def test_filename_save():
    # This is to test the logic in the load and save routines, relating
    # extensions to filetypes
    # Tuples of class, ext, loadedclass
    inklass_ext_loadklasses = (
        (Nifti1Image, '.nii', Nifti1Image),
        (Nifti2Image, '.nii', Nifti2Image),
        (Nifti1Pair, '.nii', Nifti1Image),
        (Nifti2Pair, '.nii', Nifti2Image),
        (Nifti1Image, '.img', Nifti1Pair),
        (Nifti2Image, '.img', Nifti2Pair),
        (Nifti1Pair, '.img', Nifti1Pair),
        (Nifti2Pair, '.img', Nifti2Pair),
        (Nifti1Image, '.hdr', Nifti1Pair),
        (Nifti2Image, '.hdr', Nifti2Pair),
        (Nifti1Pair, '.hdr', Nifti1Pair),
        (Nifti2Pair, '.hdr', Nifti2Pair),
        (Minc1Image, '.nii', Nifti1Image),
        (Minc1Image, '.img', Nifti1Pair),
        (Spm2AnalyzeImage, '.nii', Nifti1Image),
        (Spm2AnalyzeImage, '.img', Spm2AnalyzeImage),
        (Spm99AnalyzeImage, '.nii', Nifti1Image),
        (Spm99AnalyzeImage, '.img', Spm2AnalyzeImage),
        (AnalyzeImage, '.nii', Nifti1Image),
        (AnalyzeImage, '.img', Spm2AnalyzeImage),
    )
    shape = (2, 4, 6)
    affine = np.diag([1, 2, 3, 1])
    data = np.arange(np.prod(shape), dtype='f4').reshape(shape)
    for inklass, out_ext, loadklass in inklass_ext_loadklasses:
        if not have_scipy:
            # We can't load a SPM analyze type without scipy.  These types have
            # a 'mat' file (the type we can't load)
            if ('mat', '.mat') in loadklass.files_types:
                continue
        img = inklass(data, affine)
        try:
            pth = mkdtemp()
            fname = pjoin(pth, 'image' + out_ext)
            for path in (fname, pathlib.Path(fname)):
                nils.save(img, path)
                rt_img = nils.load(path)
                assert_array_almost_equal(rt_img.get_fdata(), data)
                assert type(rt_img) is loadklass
                # delete image to allow file close.  Otherwise windows
                # raises an error when trying to delete the directory
                del rt_img
        finally:
            shutil.rmtree(pth)


@expires('5.0.0')
def test_guessed_image_type():
    # Test whether we can guess the image type from example files
    with deprecated_to('5.0.0'):
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'example4d.nii.gz')) == Nifti1Image
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'nifti1.hdr')) == Nifti1Pair
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'example_nifti2.nii.gz')) == Nifti2Image
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'nifti2.hdr')) == Nifti2Pair
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'tiny.mnc')) == Minc1Image
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'small.mnc')) == Minc2Image
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'test.mgz')) == MGHImage
        assert nils.guessed_image_type(pjoin(DATA_PATH, 'analyze.hdr')) == Spm2AnalyzeImage


def test_fail_save():
    with InTemporaryDirectory():
        dataobj = np.ones((10, 10, 10), dtype=np.float16)
        affine = np.eye(4, dtype=np.float32)
        img = SpatialImage(dataobj, affine)
        # Fails because float16 is not supported.
        with pytest.raises(AttributeError):
            nils.save(img, 'foo.nii.gz')
        del img
