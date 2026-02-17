"""Testing loadsave module"""

import pathlib
import shutil
from os.path import dirname
from os.path import join as pjoin
from tempfile import TemporaryDirectory

import numpy as np

from .. import (
    Nifti1Image,
    Nifti1Pair,
    Nifti2Image,
    Nifti2Pair,
    Spm2AnalyzeImage,
    Spm99AnalyzeImage,
)
from ..filebasedimages import ImageFileError
from ..loadsave import _signature_matches_extension, load, read_img_data
from ..openers import Opener
from ..optpkg import optional_package
from ..testing import deprecated_to, expires
from ..tmpdirs import InTemporaryDirectory

_, have_scipy, _ = optional_package('scipy')
_, have_pyzstd, _ = optional_package('pyzstd')

import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

data_path = pjoin(dirname(__file__), 'data')


@expires('5.0.0')
def test_read_img_data():
    fnames_test = [
        'example4d.nii.gz',
        'example_nifti2.nii.gz',
        'minc1_1_scale.mnc',
        'minc1_4d.mnc',
        'test.mgz',
        'tiny.mnc',
    ]
    fnames_test += [pathlib.Path(p) for p in fnames_test]
    for fname in fnames_test:
        fpath = pjoin(data_path, fname)
        if isinstance(fname, pathlib.Path):
            fpath = pathlib.Path(fpath)
        img = load(fpath)
        data = img.get_fdata()
        with deprecated_to('5.0.0'):
            data2 = read_img_data(img)
        assert_array_equal(data, data2)
        # These examples have null scaling - assert prefer=unscaled is the same
        dao = img.dataobj
        if hasattr(dao, 'slope') and hasattr(img.header, 'raw_data_from_fileobj'):
            assert (dao.slope, dao.inter) == (1, 0)
            with deprecated_to('5.0.0'):
                assert_array_equal(read_img_data(img, prefer='unscaled'), data)
        # Assert all caps filename works as well
        with TemporaryDirectory() as tmpdir:
            up_fpath = pjoin(tmpdir, str(fname).upper())
            if isinstance(fname, pathlib.Path):
                up_fpath = pathlib.Path(up_fpath)
            shutil.copyfile(fpath, up_fpath)
            img = load(up_fpath)
            assert_array_equal(img.dataobj, data)
            del img


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load('does_not_exist.nii.gz')


def test_load_empty_image():
    with InTemporaryDirectory():
        open('empty.nii', 'w').close()
        with pytest.raises(ImageFileError) as err:
            load('empty.nii')
    assert str(err.value).startswith('Empty file: ')


@pytest.mark.parametrize('extension', ['.gz', '.bz2', '.zst'])
def test_load_bad_compressed_extension(tmp_path, extension):
    if extension == '.zst' and not have_pyzstd:
        pytest.skip()
    file_path = tmp_path / f'img.nii{extension}'
    file_path.write_bytes(b'bad')
    with pytest.raises(ImageFileError, match='.*is not a .* file'):
        load(file_path)


@pytest.mark.parametrize('extension', ['.gz', '.bz2', '.zst'])
def test_load_good_extension_with_bad_data(tmp_path, extension):
    if extension == '.zst' and not have_pyzstd:
        pytest.skip()
    file_path = tmp_path / f'img.nii{extension}'
    with Opener(file_path, 'wb') as fobj:
        fobj.write(b'bad')
    with pytest.raises(ImageFileError, match='Cannot work out file type of .*'):
        load(file_path)


def test_signature_matches_extension(tmp_path):
    gz_signature = b'\x1f\x8b'
    good_file = tmp_path / 'good.gz'
    good_file.write_bytes(gz_signature)
    bad_file = tmp_path / 'bad.gz'
    bad_file.write_bytes(b'bad')
    matches, msg = _signature_matches_extension(tmp_path / 'uncompressed.nii')
    assert matches
    assert msg == ''
    matches, msg = _signature_matches_extension(tmp_path / 'missing.gz')
    assert not matches
    assert msg.startswith('Could not read')
    matches, msg = _signature_matches_extension(bad_file)
    assert not matches
    assert 'is not a' in msg
    matches, msg = _signature_matches_extension(good_file)
    assert matches
    assert msg == ''
    matches, msg = _signature_matches_extension(tmp_path / 'missing.nii')
    assert matches
    assert msg == ''


@expires('5.0.0')
def test_read_img_data_nifti():
    shape = (2, 3, 4)
    data = np.random.normal(size=shape)
    out_dtype = np.dtype(np.int16)
    classes = (Nifti1Pair, Nifti1Image, Nifti2Pair, Nifti2Image)
    if have_scipy:
        classes += (Spm99AnalyzeImage, Spm2AnalyzeImage)
    with InTemporaryDirectory():
        for i, img_class in enumerate(classes):
            img = img_class(data, np.eye(4))
            img.set_data_dtype(out_dtype)
            # No filemap => error
            with deprecated_to('5.0.0'), pytest.raises(ImageFileError):
                read_img_data(img)
            # Make a filemap
            froot = f'an_image_{i}'
            img.file_map = img.filespec_to_file_map(froot)
            # Trying to read from this filemap will generate an error because
            # we are going to read from files that do not exist
            with deprecated_to('5.0.0'), pytest.raises(OSError):
                read_img_data(img)
            img.to_file_map()
            # Load - now the scaling and offset correctly applied
            img_fname = img.file_map['image'].filename
            img_back = load(img_fname)
            data_back = img_back.get_fdata()
            with deprecated_to('5.0.0'):
                assert_array_equal(data_back, read_img_data(img_back))
            # This is the same as if we loaded the image and header separately
            hdr_fname = img.file_map['header'].filename if 'header' in img.file_map else img_fname
            with open(hdr_fname, 'rb') as fobj:
                hdr_back = img_back.header_class.from_fileobj(fobj)
            with open(img_fname, 'rb') as fobj:
                scaled_back = hdr_back.data_from_fileobj(fobj)
            assert_array_equal(data_back, scaled_back)
            # Unscaled is the same as returned from raw_data_from_fileobj
            with open(img_fname, 'rb') as fobj:
                unscaled_back = hdr_back.raw_data_from_fileobj(fobj)
            with deprecated_to('5.0.0'):
                assert_array_equal(unscaled_back, read_img_data(img_back, prefer='unscaled'))
            # If we futz with the scaling in the header, the result changes
            with deprecated_to('5.0.0'):
                assert_array_equal(data_back, read_img_data(img_back))
            has_inter = hdr_back.has_data_intercept
            old_slope = hdr_back['scl_slope']
            old_inter = hdr_back['scl_inter'] if has_inter else 0
            est_unscaled = (data_back - old_inter) / old_slope
            with deprecated_to('5.0.0'):
                actual_unscaled = read_img_data(img_back, prefer='unscaled')
            assert_almost_equal(est_unscaled, actual_unscaled)
            img_back.header['scl_slope'] = 2.1
            if has_inter:
                new_inter = 3.14
                img_back.header['scl_inter'] = 3.14
            else:
                new_inter = 0
            # scaled scaling comes from new parameters in header
            with deprecated_to('5.0.0'):
                assert np.allclose(actual_unscaled * 2.1 + new_inter, read_img_data(img_back))
            # Unscaled array didn't change
            with deprecated_to('5.0.0'):
                assert_array_equal(actual_unscaled, read_img_data(img_back, prefer='unscaled'))
            # Check the offset too
            img.header.set_data_offset(1024)
            # Delete arrays still pointing to file, so Windows can reuse
            del actual_unscaled, unscaled_back
            img.to_file_map()
            # Write an integer of zeros after
            with open(img_fname, 'ab') as fobj:
                fobj.write(b'\x00\x00')
            img_back = load(img_fname)
            data_back = img_back.get_fdata()
            with deprecated_to('5.0.0'):
                assert_array_equal(data_back, read_img_data(img_back))
            img_back.header.set_data_offset(1026)
            # Check we pick up new offset
            exp_offset = np.zeros((data.size,), data.dtype) + old_inter
            exp_offset[:-1] = np.ravel(data_back, order='F')[1:]
            exp_offset = np.reshape(exp_offset, shape, order='F')
            with deprecated_to('5.0.0'):
                assert_array_equal(exp_offset, read_img_data(img_back))
            # Delete stuff that might hold onto file references
            del img, img_back, data_back
