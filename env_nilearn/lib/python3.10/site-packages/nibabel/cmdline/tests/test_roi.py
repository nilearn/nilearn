import os
from unittest import mock

import numpy as np
import pytest

import nibabel as nb
from nibabel.cmdline.roi import lossless_slice, main, parse_slice
from nibabel.testing import data_path


def test_parse_slice():
    assert parse_slice(None) == slice(None)
    assert parse_slice('1:5') == slice(1, 5)
    assert parse_slice('1:') == slice(1, None)
    assert parse_slice(':5') == slice(None, 5)
    assert parse_slice(':-1') == slice(None, -1)
    assert parse_slice('-5:-1') == slice(-5, -1)
    assert parse_slice('1:5:') == slice(1, 5, None)
    assert parse_slice('1::') == slice(1, None, None)
    assert parse_slice(':5:') == slice(None, 5, None)
    assert parse_slice(':-1:') == slice(None, -1, None)
    assert parse_slice('-5:-1:') == slice(-5, -1, None)
    assert parse_slice('1:5:1') == slice(1, 5, 1)
    assert parse_slice('1::1') == slice(1, None, 1)
    assert parse_slice(':5:1') == slice(None, 5, 1)
    assert parse_slice(':-1:1') == slice(None, -1, 1)
    assert parse_slice('-5:-1:1') == slice(-5, -1, 1)
    assert parse_slice('5:1:-1') == slice(5, 1, -1)
    assert parse_slice(':1:-1') == slice(None, 1, -1)
    assert parse_slice('5::-1') == slice(5, None, -1)
    assert parse_slice('-1::-1') == slice(-1, None, -1)
    assert parse_slice('-1:-5:-1') == slice(-1, -5, -1)

    # Max of start:stop:step
    with pytest.raises(ValueError):
        parse_slice('1:2:3:4')
    # Integers only
    with pytest.raises(ValueError):
        parse_slice('abc:2:3')
    with pytest.raises(ValueError):
        parse_slice('1.2:2:3')
    # Unit steps only
    with pytest.raises(ValueError):
        parse_slice('1:5:2')


def test_parse_slice_disallow_step():
    # Permit steps of 1
    assert parse_slice('1:5', False) == slice(1, 5)
    assert parse_slice('1:5:', False) == slice(1, 5)
    assert parse_slice('1:5:1', False) == slice(1, 5, 1)
    # Disable other steps
    with pytest.raises(ValueError):
        parse_slice('1:5:-1', False)
    with pytest.raises(ValueError):
        parse_slice('1:5:-2', False)


def test_lossless_slice_unknown_axes():
    img = nb.load(os.path.join(data_path, 'minc1_4d.mnc'))
    with pytest.raises(ValueError):
        lossless_slice(img, (slice(None), slice(None), slice(None)))


def test_lossless_slice_scaling(tmp_path):
    fname = tmp_path / 'image.nii'
    img = nb.Nifti1Image(np.random.uniform(-20000, 20000, (5, 5, 5, 5)), affine=np.eye(4))
    img.header.set_data_dtype('int16')
    img.to_filename(fname)
    img1 = nb.load(fname)
    sliced_fname = tmp_path / 'sliced.nii'
    lossless_slice(img1, (slice(None), slice(None), slice(2, 4))).to_filename(sliced_fname)
    img2 = nb.load(sliced_fname)

    assert np.array_equal(img1.get_fdata()[:, :, 2:4], img2.get_fdata())
    assert np.array_equal(img1.dataobj.get_unscaled()[:, :, 2:4], img2.dataobj.get_unscaled())
    assert img1.dataobj.slope == img2.dataobj.slope
    assert img1.dataobj.inter == img2.dataobj.inter


def test_lossless_slice_noscaling(tmp_path):
    fname = tmp_path / 'image.mgh'
    img = nb.MGHImage(
        np.random.uniform(-20000, 20000, (5, 5, 5, 5)).astype('float32'), affine=np.eye(4)
    )
    img.to_filename(fname)
    img1 = nb.load(fname)
    sliced_fname = tmp_path / 'sliced.mgh'
    lossless_slice(img1, (slice(None), slice(None), slice(2, 4))).to_filename(sliced_fname)
    img2 = nb.load(sliced_fname)

    assert np.array_equal(img1.get_fdata()[:, :, 2:4], img2.get_fdata())
    assert np.array_equal(img1.dataobj.get_unscaled()[:, :, 2:4], img2.dataobj.get_unscaled())
    assert img1.dataobj.slope == img2.dataobj.slope
    assert img1.dataobj.inter == img2.dataobj.inter


@pytest.mark.parametrize('inplace', (True, False))
def test_nib_roi(tmp_path, inplace):
    in_file = os.path.join(data_path, 'functional.nii')
    out_file = str(tmp_path / 'sliced.nii')
    in_img = nb.load(in_file)

    if inplace:
        in_img.to_filename(out_file)
        in_file = out_file

    retval = main([in_file, out_file, '-i', '1:-1', '-j', '-1:1:-1', '-k', '::', '-t', ':5'])
    assert retval == 0

    out_img = nb.load(out_file)
    in_data = in_img.dataobj[:]
    in_sliced = in_img.slicer[1:-1, -1:1:-1, :, :5]
    assert out_img.shape == in_sliced.shape
    assert np.array_equal(in_data[1:-1, -1:1:-1, :, :5], out_img.dataobj)
    assert np.allclose(in_sliced.dataobj, out_img.dataobj)
    assert np.allclose(in_sliced.affine, out_img.affine)


@pytest.mark.parametrize(
    ('args', 'errmsg'),
    (
        (('-i', '1:1'), 'Cannot take zero-length slice'),
        (('-j', '1::2'), 'Downsampling is not supported'),
        (('-t', '5::-1'), 'Step entry not permitted'),
    ),
)
def test_nib_roi_bad_slices(capsys, args, errmsg):
    in_file = os.path.join(data_path, 'functional.nii')

    retval = main([in_file, os.devnull, *args])
    assert retval != 0
    captured = capsys.readouterr()
    assert errmsg in captured.out


def test_entrypoint(capsys):
    # Check that we handle missing args as expected
    with mock.patch('sys.argv', ['nib-roi', '--help']):
        with pytest.raises(SystemExit):
            main()
    captured = capsys.readouterr()
    assert captured.out.startswith('usage: nib-roi')


def test_nib_roi_unknown_axes(capsys):
    in_file = os.path.join(data_path, 'minc1_4d.mnc')
    with pytest.raises(ValueError):
        main([in_file, os.devnull, '-i', ':'])
    captured = capsys.readouterr()
    assert 'Could not slice image.' in captured.out
