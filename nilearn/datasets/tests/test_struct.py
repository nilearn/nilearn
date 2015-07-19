"""
Test the datasets module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import os
import shutil
import numpy as np
from tempfile import mkdtemp

from nose import with_setup
from nose.tools import assert_true, assert_equal

from nilearn.datasets import utils, struct
from nilearn._utils.testing import (mock_request, wrap_chunk_read_,
                                    FetchFilesMock, assert_raises_regex)

from nilearn._utils.compat import _basestring


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
    original_fetch_files = struct._fetch_files
    struct._fetch_files = mock_fetch_files


def teardown_mock():
    global original_url_request
    utils._urllib.request = original_url_request

    global original_chunk_read
    utils._chunk_read_ = original_chunk_read

    global original_fetch_files
    struct._fetch_files = original_fetch_files


currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')
tmpdir = None


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


@with_setup(setup_mock, teardown_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_icbm152_2009():
    dataset = struct.fetch_icbm152_2009(data_dir=tmpdir, verbose=0)
    assert_true(isinstance(dataset.csf, _basestring))
    assert_true(isinstance(dataset.eye_mask, _basestring))
    assert_true(isinstance(dataset.face_mask, _basestring))
    assert_true(isinstance(dataset.gm, _basestring))
    assert_true(isinstance(dataset.mask, _basestring))
    assert_true(isinstance(dataset.pd, _basestring))
    assert_true(isinstance(dataset.t1, _basestring))
    assert_true(isinstance(dataset.t2, _basestring))
    assert_true(isinstance(dataset.t2_relax, _basestring))
    assert_true(isinstance(dataset.wm, _basestring))
    assert_equal(len(mock_url_request.urls), 1)


@with_setup(setup_mock, teardown_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_oasis_vbm():
    local_url = "file://" + datadir
    ids = np.asarray(['OAS1_%4d' % i for i in range(457)])
    ids = ids.view(dtype=[('ID', 'S9')])
    mock_fetch_files.add_csv('oasis_cross-sectional.csv', ids)

    # Disabled: cannot be tested without actually fetching covariates CSV file
    dataset = struct.fetch_oasis_vbm(data_dir=tmpdir, url=local_url,
                                     verbose=0)
    assert_equal(len(dataset.gray_matter_maps), 403)
    assert_equal(len(dataset.white_matter_maps), 403)
    assert_true(isinstance(dataset.gray_matter_maps[0], _basestring))
    assert_true(isinstance(dataset.white_matter_maps[0], _basestring))
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.data_usage_agreement, _basestring))
    assert_equal(len(mock_url_request.urls), 3)

    dataset = struct.fetch_oasis_vbm(data_dir=tmpdir, url=local_url,
                                     dartel_version=False, verbose=0)
    assert_equal(len(dataset.gray_matter_maps), 415)
    assert_equal(len(dataset.white_matter_maps), 415)
    assert_true(isinstance(dataset.gray_matter_maps[0], _basestring))
    assert_true(isinstance(dataset.white_matter_maps[0], _basestring))
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.data_usage_agreement, _basestring))
    assert_equal(len(mock_url_request.urls), 4)


def test_load_mni152_template():
    # All subjects
    template_nii = struct.load_mni152_template()
    assert_equal(template_nii.shape, (91, 109, 91))
    assert_equal(template_nii.get_header().get_zooms(), (2.0, 2.0, 2.0))
