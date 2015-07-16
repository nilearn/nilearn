"""
Test the datasets module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import os
import shutil
import numpy as np
from tempfile import mkdtemp

import nibabel

from nose import with_setup
from nose.tools import assert_true, assert_equal

from nilearn._utils.testing import (mock_request, wrap_chunk_read_,
                                    FetchFilesMock, assert_raises_regex)

from nilearn._utils.compat import _basestring

from nilearn.datasets import utils, atlas, struct


original_fetch_files = None
original_url_request = None
original_chunk_read = None
mock_fetch_files = None
mock_url_request = None
mock_chunk_read = None


def setup_mock():
    global original_url_request
    global mock_url_request
    mock_url_request = mock_request()
    original_url_request = utils._urllib.request
    utils._urllib.request = mock_url_request

    global original_chunk_read
    global mock_chunk_read
    mock_chunk_read = wrap_chunk_read_(utils._chunk_read_)
    original_chunk_read = utils._chunk_read_
    utils._chunk_read_ = mock_chunk_read

    global original_fetch_files
    global mock_fetch_files
    mock_fetch_files = FetchFilesMock()
    original_fetch_files = atlas._fetch_files
    atlas._fetch_files = mock_fetch_files


def teardown_mock():
    global original_url_request
    utils._urllib.request = original_url_request

    global original_chunk_read
    utils._chunk_read_ = original_chunk_read

    global original_fetch_files
    atlas._fetch_files = original_fetch_files


tmpdir = None
currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')


def setup_tmpdata():
    # create temporary dir
    global tmpdir
    tmpdir = mkdtemp()


def teardown_tmpdata():
    # remove temporary dir
    global tmpdir
    if tmpdir is not None:
        shutil.rmtree(tmpdir)


@with_setup(setup_tmpdata, teardown_tmpdata)
def test_get_dataset_dir():
    # testing folder creation under different environments, enforcing
    # a custom clean install
    os.environ.pop('NILEARN_DATA', None)
    os.environ.pop('NILEARN_SHARED_DATA', None)

    expected_base_dir = os.path.expanduser('~/nilearn_data')
    data_dir = utils._get_dataset_dir('test', verbose=0)
    assert_equal(data_dir, os.path.join(expected_base_dir, 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = os.path.join(tmpdir, 'test_nilearn_data')
    os.environ['NILEARN_DATA'] = expected_base_dir
    data_dir = utils._get_dataset_dir('test', verbose=0)
    assert_equal(data_dir, os.path.join(expected_base_dir, 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = os.path.join(tmpdir, 'nilearn_shared_data')
    os.environ['NILEARN_SHARED_DATA'] = expected_base_dir
    data_dir = utils._get_dataset_dir('test', verbose=0)
    assert_equal(data_dir, os.path.join(expected_base_dir, 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = os.path.join(tmpdir, 'env_data')
    expected_dataset_dir = os.path.join(expected_base_dir, 'test')
    data_dir = utils._get_dataset_dir(
        'test', default_paths=[expected_dataset_dir], verbose=0)
    assert_equal(data_dir, os.path.join(expected_base_dir, 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    no_write = os.path.join(tmpdir, 'no_write')
    os.makedirs(no_write)
    os.chmod(no_write, 0o400)

    expected_base_dir = os.path.join(tmpdir, 'nilearn_shared_data')
    os.environ['NILEARN_SHARED_DATA'] = expected_base_dir
    data_dir = utils._get_dataset_dir('test',
                                               default_paths=[no_write],
                                               verbose=0)
    # Non writeable dir is returned because dataset may be in there.
    assert_equal(data_dir, no_write)
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    # Verify exception for a path which exists and is a file
    test_file = os.path.join(tmpdir, 'some_file')
    with open(test_file, 'w') as out:
        out.write('abcfeg')
    assert_raises_regex(OSError, 'Not a directory',
                        utils._get_dataset_dir, 'test', test_file,
                        verbose=0)


@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fail_fetch_atlas_harvard_oxford():
    # specify non-existing atlas item
    assert_raises_regex(ValueError, 'Invalid atlas name',
                        atlas.fetch_atlas_harvard_oxford,
                        'not_inside')

    # specify existing atlas item
    target_atlas = 'cort-maxprob-thr0-1mm'
    target_atlas_fname = 'HarvardOxford-' + target_atlas + '.nii.gz'

    HO_dir = os.path.join(tmpdir, 'fsl', 'data', 'atlases')
    os.makedirs(HO_dir)
    nifti_dir = os.path.join(HO_dir, 'HarvardOxford')
    os.makedirs(nifti_dir)

    target_atlas_nii = os.path.join(nifti_dir, target_atlas_fname)
    struct.load_mni152_template().to_filename(target_atlas_nii)

    dummy = open(os.path.join(HO_dir, 'HarvardOxford-Cortical.xml'), 'w')
    dummy.write("<?xml version='1.0' encoding='us-ascii'?> "
                "<metadata>"
                "</metadata>")
    dummy.close()

    ho = atlas.fetch_atlas_harvard_oxford(target_atlas,
                                                   data_dir=tmpdir)

    assert_true(isinstance(nibabel.load(ho.maps), nibabel.Nifti1Image))
    assert_true(isinstance(ho.labels, np.ndarray))
    assert_true(len(ho.labels) > 0)


@with_setup(setup_mock, teardown_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_atlas_craddock_2012():
    print '*' * 80
    print tmpdir
    print '*' * 80
    bunch = atlas.fetch_atlas_craddock_2012(data_dir=tmpdir,
                                                     verbose=0)

    keys = ("scorr_mean", "tcorr_mean",
            "scorr_2level", "tcorr_2level",
            "random")
    filenames = [
            "scorr05_mean_all.nii.gz",
            "tcorr05_mean_all.nii.gz",
            "scorr05_2level_all.nii.gz",
            "tcorr05_2level_all.nii.gz",
            "random_all.nii.gz",
    ]
    assert_equal(len(mock_url_request.urls), 1)
    for key, fn in zip(keys, filenames):
        assert_equal(bunch[key], os.path.join(tmpdir, 'craddock_2012', fn))


@with_setup(setup_mock, teardown_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_atlas_smith_2009():
    bunch = atlas.fetch_atlas_smith_2009(data_dir=tmpdir, verbose=0)

    keys = ("rsn20", "rsn10", "rsn70",
            "bm20", "bm10", "bm70")
    filenames = [
            "rsn20.nii.gz",
            "PNAS_Smith09_rsn10.nii.gz",
            "rsn70.nii.gz",
            "bm20.nii.gz",
            "PNAS_Smith09_bm10.nii.gz",
            "bm70.nii.gz",
    ]

    assert_equal(len(mock_url_request.urls), 6)
    for key, fn in zip(keys, filenames):
        assert_equal(bunch[key], os.path.join(tmpdir, 'smith_2009', fn))


def test_fetch_atlas_power_2011_atlas():
    bunch = atlas.fetch_atlas_power_2011()
    assert_equal(len(bunch.rois), 264)


@with_setup(setup_mock, teardown_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_atlas_destrieux_2009_atlas():
    datadir = os.path.join(tmpdir, 'destrieux_2009')
    os.mkdir(datadir)
    dummy = open(os.path.join(
        datadir, 'destrieux2009_rois_labels_lateralized.csv'), 'w')
    dummy.write("name,index")
    dummy.close()
    bunch = atlas.fetch_atlas_destrieux_2009(data_dir=tmpdir,
                                                      verbose=0)

    assert_equal(len(mock_url_request.urls), 1)
    assert_equal(bunch['maps'], os.path.join(
        tmpdir, 'destrieux_2009', 'destrieux2009_rois_lateralized.nii.gz'))

    dummy = open(os.path.join(
        datadir, 'destrieux2009_rois_labels.csv'), 'w')
    dummy.write("name,index")
    dummy.close()
    bunch = atlas.fetch_atlas_destrieux_2009(
        lateralized=False, data_dir=tmpdir, verbose=0)

    assert_equal(len(mock_url_request.urls), 1)
    assert_equal(bunch['maps'], os.path.join(
        tmpdir, 'destrieux_2009', 'destrieux2009_rois.nii.gz'))


@with_setup(setup_mock, teardown_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_atlas_msdl():
    dataset = atlas.fetch_atlas_msdl(data_dir=tmpdir, verbose=0)
    assert_true(isinstance(dataset.labels, _basestring))
    assert_true(isinstance(dataset.maps, _basestring))
    assert_equal(len(mock_url_request.urls), 1)


@with_setup(setup_mock, teardown_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_atlas_yeo_2011():
    dataset = atlas.fetch_atlas_yeo_2011(data_dir=tmpdir, verbose=0)
    assert_true(isinstance(dataset.anat, _basestring))
    assert_true(isinstance(dataset.colors_17, _basestring))
    assert_true(isinstance(dataset.colors_7, _basestring))
    assert_true(isinstance(dataset.thick_17, _basestring))
    assert_true(isinstance(dataset.thick_7, _basestring))
    assert_true(isinstance(dataset.thin_17, _basestring))
    assert_true(isinstance(dataset.thin_7, _basestring))
    assert_equal(len(mock_url_request.urls), 1)
