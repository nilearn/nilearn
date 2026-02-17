#!python
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
import pytest

import nibabel as nib
from nibabel.cmdline import convert
from nibabel.testing import get_test_data


def test_convert_noop(tmp_path):
    infile = get_test_data(fname='anatomical.nii')
    outfile = tmp_path / 'output.nii.gz'

    orig = nib.load(infile)
    assert not outfile.exists()

    convert.main([str(infile), str(outfile)])
    assert outfile.is_file()

    converted = nib.load(outfile)
    assert np.allclose(converted.affine, orig.affine)
    assert converted.shape == orig.shape
    assert converted.get_data_dtype() == orig.get_data_dtype()

    infile = get_test_data(fname='resampled_anat_moved.nii')

    with pytest.raises(FileExistsError):
        convert.main([str(infile), str(outfile)])

    convert.main([str(infile), str(outfile), '--force'])
    assert outfile.is_file()

    # Verify that we did overwrite
    converted2 = nib.load(outfile)
    assert not (
        converted2.shape == converted.shape
        and np.allclose(converted2.affine, converted.affine)
        and np.allclose(converted2.get_fdata(), converted.get_fdata())
    )


@pytest.mark.parametrize('data_dtype', ('u1', 'i2', 'float32', 'float', 'int64'))
def test_convert_dtype(tmp_path, data_dtype):
    infile = get_test_data(fname='anatomical.nii')
    outfile = tmp_path / 'output.nii.gz'

    orig = nib.load(infile)
    assert not outfile.exists()

    # np.dtype() will give us the dtype for the system endianness if that
    # mismatches the data file, we will fail equality, so get the dtype that
    # matches the requested precision but in the endianness of the file
    expected_dtype = np.dtype(data_dtype).newbyteorder(orig.header.endianness)

    convert.main([str(infile), str(outfile), '--out-dtype', data_dtype])
    assert outfile.is_file()

    converted = nib.load(outfile)
    assert np.allclose(converted.affine, orig.affine)
    assert converted.shape == orig.shape
    assert converted.get_data_dtype() == expected_dtype


@pytest.mark.parametrize(
    ('ext', 'img_class'),
    [
        ('mgh', nib.MGHImage),
        ('img', nib.Nifti1Pair),
    ],
)
def test_convert_by_extension(tmp_path, ext, img_class):
    infile = get_test_data(fname='anatomical.nii')
    outfile = tmp_path / f'output.{ext}'

    orig = nib.load(infile)
    assert not outfile.exists()

    convert.main([str(infile), str(outfile)])
    assert outfile.is_file()

    converted = nib.load(outfile)
    assert np.allclose(converted.affine, orig.affine)
    assert converted.shape == orig.shape
    assert converted.__class__ == img_class


@pytest.mark.parametrize(
    ('ext', 'img_class'),
    [
        ('mgh', nib.MGHImage),
        ('img', nib.Nifti1Pair),
        ('nii', nib.Nifti2Image),
    ],
)
def test_convert_imgtype(tmp_path, ext, img_class):
    infile = get_test_data(fname='anatomical.nii')
    outfile = tmp_path / f'output.{ext}'

    orig = nib.load(infile)
    assert not outfile.exists()

    convert.main([str(infile), str(outfile), '--image-type', img_class.__name__])
    assert outfile.is_file()

    converted = nib.load(outfile)
    assert np.allclose(converted.affine, orig.affine)
    assert converted.shape == orig.shape
    assert converted.__class__ == img_class


def test_convert_nifti_int_fail(tmp_path):
    infile = get_test_data(fname='anatomical.nii')
    outfile = tmp_path / 'output.nii'

    orig = nib.load(infile)
    assert not outfile.exists()

    with pytest.raises(ValueError):
        convert.main([str(infile), str(outfile), '--out-dtype', 'int'])
    assert not outfile.exists()

    with pytest.warns(UserWarning):
        convert.main([str(infile), str(outfile), '--out-dtype', 'int', '--force'])
    assert outfile.is_file()

    converted = nib.load(outfile)
    assert np.allclose(converted.affine, orig.affine)
    assert converted.shape == orig.shape
    # Note: '--force' ignores the error, but can't interpret it enough to do
    # the cast anyway
    assert converted.get_data_dtype() == orig.get_data_dtype()


@pytest.mark.parametrize(
    ('orig_dtype', 'alias', 'expected_dtype'),
    [
        ('int64', 'mask', 'uint8'),
        ('int64', 'compat', 'int32'),
        ('int64', 'smallest', 'uint8'),
        ('float64', 'mask', 'uint8'),
        ('float64', 'compat', 'float32'),
    ],
)
def test_convert_aliases(tmp_path, orig_dtype, alias, expected_dtype):
    orig_fname = tmp_path / 'orig.nii'
    out_fname = tmp_path / 'out.nii'

    arr = np.arange(24).reshape((2, 3, 4))
    img = nib.Nifti1Image(arr, np.eye(4), dtype=orig_dtype)
    img.to_filename(orig_fname)

    assert orig_fname.exists()
    assert not out_fname.exists()

    convert.main([str(orig_fname), str(out_fname), '--out-dtype', alias])
    assert out_fname.is_file()

    expected_dtype = np.dtype(expected_dtype).newbyteorder(img.header.endianness)

    converted = nib.load(out_fname)
    assert converted.get_data_dtype() == expected_dtype
