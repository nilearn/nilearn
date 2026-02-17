# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Test scripts

Test running scripts
"""

import csv
import os
import shutil
import sys
import unittest
from glob import glob
from os.path import abspath, basename, dirname, exists, splitext
from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

import nibabel as nib

from ..loadsave import load
from ..orientations import aff2axcodes, inv_ornt_aff
from ..testing import assert_data_similar, assert_dt_equal, assert_re_in
from ..tmpdirs import InTemporaryDirectory
from .nibabel_data import needs_nibabel_data
from .scriptrunner import ScriptRunner
from .test_parrec import DTI_PAR_BVALS, DTI_PAR_BVECS
from .test_parrec import EXAMPLE_IMAGES as PARREC_EXAMPLES
from .test_parrec_data import AFF_OFF, BALLS


def _proc_stdout(stdout):
    stdout_str = stdout.decode('latin1').strip()
    return stdout_str.replace(os.linesep, '\n')


runner = ScriptRunner(
    script_sdir='bin', debug_print_var='NIPY_DEBUG_PRINT', output_processor=_proc_stdout
)
run_command = runner.run_command


def script_test(func):
    # Decorator to label test as a script_test
    func.script_test = True
    return func


script_test.__test__ = False  # It's not a test

DATA_PATH = abspath(pjoin(dirname(__file__), 'data'))


def load_small_file():
    try:
        load(pjoin(DATA_PATH, 'small.mnc'))
        return True
    except:
        return False


def check_nib_ls_example4d(opts=[], hdrs_str='', other_str=''):
    # test nib-ls script
    fname = pjoin(DATA_PATH, 'example4d.nii.gz')
    expected_re = (
        ' (int16|[<>]i2) \\[128,  96,  24,   2\\] 2.00x2.00x2.20x2000.00  '
        f'#exts: 2{hdrs_str} sform{other_str}$'
    )
    cmd = ['nib-ls'] + opts + [fname]
    code, stdout, stderr = run_command(cmd)
    assert fname == stdout[: len(fname)]
    assert_re_in(expected_re, stdout[len(fname) :])


def check_nib_diff_examples():
    fnames = [pjoin(DATA_PATH, f) for f in ('standard.nii.gz', 'example4d.nii.gz')]
    code, stdout, stderr = run_command(['nib-diff'] + fnames, check_code=False)
    checked_fields = [
        'Field/File',
        'regular',
        'dim_info',
        'dim',
        'datatype',
        'bitpix',
        'pixdim',
        'slice_end',
        'xyzt_units',
        'cal_max',
        'descrip',
        'qform_code',
        'sform_code',
        'quatern_b',
        'quatern_c',
        'quatern_d',
        'qoffset_x',
        'qoffset_y',
        'qoffset_z',
        'srow_x',
        'srow_y',
        'srow_z',
        'DATA(md5)',
        'DATA(diff 1:)',
    ]
    for item in checked_fields:
        assert item in stdout

    fnames2 = [pjoin(DATA_PATH, f) for f in ('example4d.nii.gz', 'example4d.nii.gz')]
    code, stdout, stderr = run_command(['nib-diff'] + fnames2, check_code=False)
    assert stdout == 'These files are identical.'

    fnames3 = [
        pjoin(DATA_PATH, f)
        for f in ('standard.nii.gz', 'example4d.nii.gz', 'example_nifti2.nii.gz')
    ]
    code, stdout, stderr = run_command(['nib-diff'] + fnames3, check_code=False)
    for item in checked_fields:
        assert item in stdout

    fnames4 = [
        pjoin(DATA_PATH, f) for f in ('standard.nii.gz', 'standard.nii.gz', 'standard.nii.gz')
    ]
    code, stdout, stderr = run_command(['nib-diff'] + fnames4, check_code=False)
    assert stdout == 'These files are identical.'

    code, stdout, stderr = run_command(['nib-diff', '--dt', 'float64'] + fnames, check_code=False)
    for item in checked_fields:
        assert item in stdout


@pytest.mark.parametrize(
    'args',
    [
        [],
        [['-H', 'dim,bitpix'], r' \[  4 128  96  24   2   1   1   1\] 16'],
        [['-c'], '', ' !1030 uniques. Use --all-counts'],
        [['-c', '--all-counts'], '', ' 2:3 3:2 4:1 5:1.*'],
        # both stats and counts
        [['-c', '-s', '--all-counts'], '', r' \[229725\] \[2, 1.2e\+03\] 2:3 3:2 4:1 5:1.*'],
        # and must not error out if we allow for zeros
        [
            ['-c', '-s', '-z', '--all-counts'],
            '',
            r' \[589824\] \[0, 1.2e\+03\] 0:360099 2:3 3:2 4:1 5:1.*',
        ],
    ],
)
@script_test
def test_nib_ls(args):
    check_nib_ls_example4d(*args)


@unittest.skipUnless(load_small_file(), "Can't load the small.mnc file")
@script_test
def test_nib_ls_multiple():
    # verify that correctly lists/formats for multiple files
    fnames = [
        pjoin(DATA_PATH, f)
        for f in ('example4d.nii.gz', 'example_nifti2.nii.gz', 'small.mnc', 'nifti2.hdr')
    ]
    code, stdout, stderr = run_command(['nib-ls'] + fnames)
    stdout_lines = stdout.split('\n')
    assert len(stdout_lines) == 4

    # they should be indented correctly.  Since all files are int type -
    ln = max(len(f) for f in fnames)
    i_str = ' i' if sys.byteorder == 'little' else ' <i'
    assert [l[ln : ln + len(i_str)] for l in stdout_lines] == [
        i_str
    ] * 4, f"Type sub-string didn't start with '{i_str}'. Full output was: {stdout_lines}"
    # and if disregard type indicator which might vary
    assert [l[l.index('[') :] for l in stdout_lines] == [
        '[128,  96,  24,   2] 2.00x2.00x2.20x2000.00  #exts: 2 sform',
        '[ 32,  20,  12,   2] 2.00x2.00x2.20x2000.00  #exts: 2 sform',
        '[ 18,  28,  29]      9.00x8.00x7.00',
        '[ 91, 109,  91]      2.00x2.00x2.00',
    ]

    # Now run with -s for stats
    code, stdout, stderr = run_command(['nib-ls', '-s'] + fnames)
    stdout_lines = stdout.split('\n')
    assert len(stdout_lines) == 4
    assert [l[l.index('[') :] for l in stdout_lines] == [
        '[128,  96,  24,   2] 2.00x2.00x2.20x2000.00  #exts: 2 sform [229725] [2, 1.2e+03]',
        '[ 32,  20,  12,   2] 2.00x2.00x2.20x2000.00  #exts: 2 sform [15360]  [46, 7.6e+02]',
        '[ 18,  28,  29]      9.00x8.00x7.00                         [14616]  [0.12, 93]',
        '[ 91, 109,  91]      2.00x2.00x2.00                          !error',
    ]


@script_test
def test_help():
    for cmd in ['parrec2nii', 'nib-dicomfs', 'nib-ls', 'nib-nifti-dx']:
        if cmd == 'nib-dicomfs':
            # needs special treatment since depends on fuse module which
            # might not be available.
            try:
                import fuse  # noqa: F401
            except Exception:
                continue  # do not test this one
        code, stdout, stderr = run_command([cmd, '--help'])
        assert code == 0
        assert_re_in(f'.*{cmd}', stdout)
        assert_re_in('.*[uU]sage', stdout)
        # Some third party modules might like to announce some Deprecation
        # etc warnings, see e.g. https://travis-ci.org/nipy/nibabel/jobs/370353602
        if 'warning' not in stderr.lower():
            assert stderr == ''


@script_test
def test_nib_diff():
    check_nib_diff_examples()


@script_test
def test_nib_nifti_dx():
    # Test nib-nifti-dx script
    clean_hdr = pjoin(DATA_PATH, 'nifti1.hdr')
    cmd = ['nib-nifti-dx', clean_hdr]
    code, stdout, stderr = run_command(cmd)
    assert stdout.strip() == f'Header for "{clean_hdr}" is clean'
    dirty_hdr = pjoin(DATA_PATH, 'analyze.hdr')
    cmd = ['nib-nifti-dx', dirty_hdr]
    code, stdout, stderr = run_command(cmd)
    expected = f"""Picky header check output for "{dirty_hdr}"

pixdim[0] (qfac) should be 1 (default) or -1
magic string '' is not valid
sform_code 11776 not valid"""
    # Split strings to remove line endings
    assert stdout == expected


def vox_size(affine):
    return np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))


def check_conversion(cmd, pr_data, out_fname):
    run_command(cmd)
    img = load(out_fname)
    # Check orientations always LAS
    assert aff2axcodes(img.affine) == tuple('LAS')
    data = img.get_fdata()
    assert np.allclose(data, pr_data)
    assert np.allclose(img.header['cal_min'], data.min())
    assert np.allclose(img.header['cal_max'], data.max())
    del img, data  # for windows to be able to later delete the file
    # Check minmax options
    run_command(cmd + ['--minmax', '1', '2'])
    img = load(out_fname)
    data = img.get_fdata()
    assert np.allclose(data, pr_data)
    assert np.allclose(img.header['cal_min'], 1)
    assert np.allclose(img.header['cal_max'], 2)
    del img, data  # for windows
    run_command(cmd + ['--minmax', 'parse', '2'])
    img = load(out_fname)
    data = img.get_fdata()
    assert np.allclose(data, pr_data)
    assert np.allclose(img.header['cal_min'], data.min())
    assert np.allclose(img.header['cal_max'], 2)
    del img, data  # for windows
    run_command(cmd + ['--minmax', '1', 'parse'])
    img = load(out_fname)
    data = img.get_fdata()
    assert np.allclose(data, pr_data)
    assert np.allclose(img.header['cal_min'], 1)
    assert np.allclose(img.header['cal_max'], data.max())
    del img, data


@script_test
def test_parrec2nii():
    # Test parrec2nii script
    cmd = ['parrec2nii', '--help']
    code, stdout, stderr = run_command(cmd)
    assert stdout.startswith('Usage')
    with InTemporaryDirectory():
        for eg_dict in PARREC_EXAMPLES:
            fname = eg_dict['fname']
            run_command(['parrec2nii', fname])
            out_froot = splitext(basename(fname))[0] + '.nii'
            img = load(out_froot)
            assert img.shape == eg_dict['shape']
            assert_dt_equal(img.get_data_dtype(), eg_dict['dtype'])
            # Check against values from Philips converted nifti image
            data = img.get_fdata()
            assert_data_similar(data, eg_dict)
            assert_almost_equal(img.header.get_zooms(), eg_dict['zooms'])
            # Standard save does not save extensions
            assert len(img.header.extensions) == 0
            # Delete previous img, data to make Windows happier
            del img, data
            # Does not overwrite unless option given
            code, stdout, stderr = run_command(['parrec2nii', fname], check_code=False)
            assert code == 1
            # Default scaling is dv
            pr_img = load(fname)
            flipped_data = np.flip(pr_img.get_fdata(), 1)
            base_cmd = ['parrec2nii', '--overwrite', fname]
            check_conversion(base_cmd, flipped_data, out_froot)
            check_conversion(base_cmd + ['--scaling=dv'], flipped_data, out_froot)
            # fp
            pr_img = load(fname, scaling='fp')
            flipped_data = np.flip(pr_img.get_fdata(), 1)
            check_conversion(base_cmd + ['--scaling=fp'], flipped_data, out_froot)
            # no scaling
            unscaled_flipped = np.flip(pr_img.dataobj.get_unscaled(), 1)
            check_conversion(base_cmd + ['--scaling=off'], unscaled_flipped, out_froot)
            # Save extensions
            run_command(base_cmd + ['--store-header'])
            img = load(out_froot)
            assert len(img.header.extensions) == 1
            del img  # To help windows delete the file


@script_test
@needs_nibabel_data('nitest-balls1')
def test_parrec2nii_with_data():
    # Use nibabel-data to test conversion
    # Premultiplier to relate our affines to Philips conversion
    LAS2LPS = inv_ornt_aff([[0, 1], [1, -1], [2, 1]], (80, 80, 10))
    with InTemporaryDirectory():
        for par in glob(pjoin(BALLS, 'PARREC', '*.PAR')):
            par_root, ext = splitext(basename(par))
            # NA.PAR appears to be a localizer, with three slices in each of
            # the three orientations: sagittal; coronal, transverse
            if par_root == 'NA':
                continue
            # Do conversion
            run_command(['parrec2nii', par])
            conved_img = load(par_root + '.nii')
            # Confirm parrec2nii conversions are LAS
            assert aff2axcodes(conved_img.affine) == tuple('LAS')
            # Shape same whether LPS or LAS
            assert conved_img.shape[:3] == (80, 80, 10)
            # Test against original converted NIfTI
            nifti_fname = pjoin(BALLS, 'NIFTI', par_root + '.nii.gz')
            if exists(nifti_fname):
                philips_img = load(nifti_fname)
                # Confirm Philips converted image always LPS
                assert aff2axcodes(philips_img.affine) == tuple('LPS')
                # Equivalent to Philips LPS affine
                equiv_affine = conved_img.affine.dot(LAS2LPS)
                assert_almost_equal(philips_img.affine[:3, :3], equiv_affine[:3, :3], 3)
                # The translation part is always off by the same ammout
                aff_off = equiv_affine[:3, 3] - philips_img.affine[:3, 3]
                assert_almost_equal(aff_off, AFF_OFF, 3)
                # The difference is max in the order of 0.5 voxel
                vox_sizes = vox_size(philips_img.affine)
                assert np.all(np.abs(aff_off / vox_sizes) <= 0.501)
                # The data is very close, unless it's the fieldmap
                if par_root != 'fieldmap':
                    conved_data_lps = np.flip(conved_img.dataobj, 1)
                    assert np.allclose(conved_data_lps, philips_img.dataobj)
    with InTemporaryDirectory():
        # Test some options
        dti_par = pjoin(BALLS, 'PARREC', 'DTI.PAR')
        run_command(['parrec2nii', dti_par])
        assert exists('DTI.nii')
        assert not exists('DTI.bvals')
        assert not exists('DTI.bvecs')
        # Does not overwrite unless option given
        code, stdout, stderr = run_command(['parrec2nii', dti_par], check_code=False)
        assert code == 1
        # Writes bvals, bvecs files if asked
        run_command(['parrec2nii', '--overwrite', '--keep-trace', '--bvs', dti_par])
        bvecs_trace = np.loadtxt('DTI.bvecs').T
        bvals_trace = np.loadtxt('DTI.bvals')
        assert_almost_equal(bvals_trace, DTI_PAR_BVALS)
        img = load('DTI.nii')
        data = img.get_fdata()
        del img
        # Bvecs in header, transposed from PSL to LPS
        bvecs_LPS = DTI_PAR_BVECS[:, [2, 0, 1]]
        # Adjust for output flip of Y axis in data and bvecs
        bvecs_LAS = bvecs_LPS * [1, -1, 1]
        assert_almost_equal(np.loadtxt('DTI.bvecs'), bvecs_LAS.T)
        # Dwell time
        assert not exists('DTI.dwell_time')
        # Need field strength if requesting dwell time
        (
            code,
            _,
            _,
        ) = run_command(['parrec2nii', '--overwrite', '--dwell-time', dti_par], check_code=False)
        assert code == 1
        run_command(
            ['parrec2nii', '--overwrite', '--dwell-time', '--field-strength', '3', dti_par]
        )
        exp_dwell = (26 * 9.087) / (42.576 * 3.4 * 3 * 28)
        with open('DTI.dwell_time') as fobj:
            contents = fobj.read().strip()
        assert_almost_equal(float(contents), exp_dwell)
        # ensure trace is removed by default
        run_command(['parrec2nii', '--overwrite', '--bvs', dti_par])
        assert exists('DTI.bvals')
        assert exists('DTI.bvecs')
        img = load('DTI.nii')
        bvecs_notrace = np.loadtxt('DTI.bvecs').T
        bvals_notrace = np.loadtxt('DTI.bvals')
        data_notrace = img.get_fdata()
        assert data_notrace.shape[-1] == len(bvecs_notrace)
        del img
        # ensure correct volume was removed
        good_mask = np.logical_or((bvecs_trace != 0).any(axis=1), bvals_trace == 0)
        assert_almost_equal(data_notrace, data[..., good_mask])
        assert_almost_equal(bvals_notrace, np.array(DTI_PAR_BVALS)[good_mask])
        assert_almost_equal(bvecs_notrace, bvecs_LAS[good_mask])
        # test --strict-sort
        run_command(
            ['parrec2nii', '--overwrite', '--keep-trace', '--bvs', '--strict-sort', dti_par]
        )
        # strict-sort: bvals should be in ascending order
        assert_almost_equal(np.loadtxt('DTI.bvals'), np.sort(DTI_PAR_BVALS))
        img = load('DTI.nii')
        data_sorted = img.get_fdata()
        assert_almost_equal(data[..., np.argsort(DTI_PAR_BVALS, kind='stable')], data_sorted)
        del img

        # Writes .ordering.csv if requested
        run_command(['parrec2nii', '--overwrite', '--volume-info', dti_par])
        assert exists('DTI.ordering.csv')
        with open('DTI.ordering.csv') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            csv_keys = next(csvreader)  # header row
            nlines = 0  # count number of non-header rows
            for line in csvreader:
                nlines += 1

        assert sorted(csv_keys) == ['diffusion b value number', 'gradient orientation number']
        assert nlines == 8  # 8 volumes present in DTI.PAR


@script_test
def test_nib_trk2tck():
    simple_trk = pjoin(DATA_PATH, 'simple.trk')
    standard_trk = pjoin(DATA_PATH, 'standard.trk')

    with InTemporaryDirectory() as tmpdir:
        # Copy input files to convert.
        shutil.copy(simple_trk, tmpdir)
        shutil.copy(standard_trk, tmpdir)
        simple_trk = pjoin(tmpdir, 'simple.trk')
        standard_trk = pjoin(tmpdir, 'standard.trk')
        simple_tck = pjoin(tmpdir, 'simple.tck')
        standard_tck = pjoin(tmpdir, 'standard.tck')

        # Convert one file.
        cmd = ['nib-trk2tck', simple_trk]
        code, stdout, stderr = run_command(cmd)
        assert len(stdout) == 0
        assert os.path.isfile(simple_tck)
        trk = nib.streamlines.load(simple_trk)
        tck = nib.streamlines.load(simple_tck)
        assert (tck.streamlines.get_data() == trk.streamlines.get_data()).all()
        assert isinstance(tck, nib.streamlines.TckFile)

        # Skip non TRK files.
        cmd = ['nib-trk2tck', simple_tck]
        code, stdout, stderr = run_command(cmd)
        assert 'Skipping non TRK file' in stdout

        # By default, refuse to overwrite existing output files.
        cmd = ['nib-trk2tck', simple_trk]
        code, stdout, stderr = run_command(cmd)
        assert 'Skipping existing file' in stdout

        # Convert multiple files and with --force.
        cmd = ['nib-trk2tck', '--force', simple_trk, standard_trk]
        code, stdout, stderr = run_command(cmd)
        assert len(stdout) == 0
        trk = nib.streamlines.load(standard_trk)
        tck = nib.streamlines.load(standard_tck)
        assert (tck.streamlines.get_data() == trk.streamlines.get_data()).all()


@script_test
def test_nib_tck2trk():
    anat = pjoin(DATA_PATH, 'standard.nii.gz')
    standard_tck = pjoin(DATA_PATH, 'standard.tck')

    with InTemporaryDirectory() as tmpdir:
        # Copy input file to convert.
        shutil.copy(standard_tck, tmpdir)
        standard_trk = pjoin(tmpdir, 'standard.trk')
        standard_tck = pjoin(tmpdir, 'standard.tck')

        # Anatomical image not found as first argument.
        cmd = ['nib-tck2trk', standard_tck, anat]
        code, stdout, stderr = run_command(cmd, check_code=False)
        assert code == 2  # Parser error.
        assert 'Expecting anatomical image as first argument' in stderr

        # Convert one file.
        cmd = ['nib-tck2trk', anat, standard_tck]
        code, stdout, stderr = run_command(cmd)
        assert len(stdout) == 0
        assert os.path.isfile(standard_trk)
        tck = nib.streamlines.load(standard_tck)
        trk = nib.streamlines.load(standard_trk)
        assert (trk.streamlines.get_data() == tck.streamlines.get_data()).all()
        assert isinstance(trk, nib.streamlines.TrkFile)

        # Skip non TCK files.
        cmd = ['nib-tck2trk', anat, standard_trk]
        code, stdout, stderr = run_command(cmd)
        assert 'Skipping non TCK file' in stdout

        # By default, refuse to overwrite existing output files.
        cmd = ['nib-tck2trk', anat, standard_tck]
        code, stdout, stderr = run_command(cmd)
        assert 'Skipping existing file' in stdout

        # Convert multiple files and with --force.
        cmd = ['nib-tck2trk', '--force', anat, standard_tck, standard_tck]
        code, stdout, stderr = run_command(cmd)
        assert len(stdout) == 0
        tck = nib.streamlines.load(standard_tck)
        trk = nib.streamlines.load(standard_trk)
        assert (tck.streamlines.get_data() == trk.streamlines.get_data()).all()
