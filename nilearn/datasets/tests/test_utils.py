"""
Test the datasets module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import contextlib
import gzip
import os
import shutil
import tarfile
import zipfile
import urllib
import json
from unittest.mock import MagicMock
from tempfile import mkdtemp, mkstemp

try:
    import boto3  # noqa: F401

except ImportError:
    BOTO_INSTALLED = False
else:
    BOTO_INSTALLED = True

import numpy as np
import pytest
import requests

from nilearn import datasets
from nilearn.datasets.utils import (_get_dataset_dir,
                                    make_fresh_openneuro_dataset_urls_index)
from nilearn.datasets import utils


currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')


@pytest.mark.parametrize("should_cast_path_to_string", [False, True])
def test_get_dataset_dir(should_cast_path_to_string, tmp_path):
    # testing folder creation under different environments, enforcing
    # a custom clean install
    os.environ.pop('NILEARN_DATA', None)
    os.environ.pop('NILEARN_SHARED_DATA', None)

    expected_base_dir = os.path.expanduser('~/nilearn_data')
    data_dir = datasets.utils._get_dataset_dir('test', verbose=0)
    assert data_dir == os.path.join(expected_base_dir, 'test')
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = str(tmp_path / 'test_nilearn_data')
    os.environ['NILEARN_DATA'] = expected_base_dir
    data_dir = datasets.utils._get_dataset_dir('test', verbose=0)
    assert data_dir == os.path.join(expected_base_dir, 'test')
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = str(tmp_path / 'nilearn_shared_data')
    os.environ['NILEARN_SHARED_DATA'] = expected_base_dir
    data_dir = datasets.utils._get_dataset_dir('test', verbose=0)
    assert data_dir == os.path.join(expected_base_dir, 'test')
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = tmp_path / 'env_data'
    expected_dataset_dir = expected_base_dir / 'test'
    if should_cast_path_to_string:
        expected_dataset_dir = str(expected_dataset_dir)
    data_dir = datasets.utils._get_dataset_dir(
        'test', default_paths=[expected_dataset_dir], verbose=0)
    assert data_dir == str(expected_dataset_dir)
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    no_write = str(tmp_path / 'no_write')
    os.makedirs(no_write)
    os.chmod(no_write, 0o400)

    expected_base_dir = str(tmp_path / 'nilearn_shared_data')
    os.environ['NILEARN_SHARED_DATA'] = expected_base_dir
    data_dir = datasets.utils._get_dataset_dir('test',
                                               default_paths=[no_write],
                                               verbose=0)
    # Non writeable dir is returned because dataset may be in there.
    assert data_dir == no_write
    assert os.path.exists(data_dir)
    os.chmod(no_write, 0o600)
    shutil.rmtree(data_dir)

    # Verify exception for a path which exists and is a file
    test_file = str(tmp_path / 'some_file')
    with open(test_file, 'w') as out:
        out.write('abcfeg')

    with pytest.raises(
            OSError,
            match='Nilearn tried to store the dataset in the following '
                  'directories, but'):
        datasets.utils._get_dataset_dir('test', test_file, verbose=0)


def test_md5_sum_file():
    # Create dummy temporary file
    out, f = mkstemp()
    os.write(out, b'abcfeg')
    os.close(out)
    assert (datasets.utils._md5_sum_file(f) ==
                 '18f32295c556b2a1a3a8e68fe1ad40f7')
    os.remove(f)


def test_read_md5_sum_file():
    # Create dummy temporary file
    out, f = mkstemp()
    os.write(out, b'20861c8c3fe177da19a7e9539a5dbac  /tmp/test\n'
             b'70886dcabe7bf5c5a1c24ca24e4cbd94  test/some_image.nii')
    os.close(out)
    h = datasets.utils._read_md5_sum_file(f)
    assert '/tmp/test' in h
    assert not '/etc/test' in h
    assert h['test/some_image.nii'] == '70886dcabe7bf5c5a1c24ca24e4cbd94'
    assert h['/tmp/test'] == '20861c8c3fe177da19a7e9539a5dbac'
    os.remove(f)


def test_tree():
    # Create a dummy directory tree
    parent = mkdtemp()

    open(os.path.join(parent, 'file1'), 'w').close()
    open(os.path.join(parent, 'file2'), 'w').close()
    dir1 = os.path.join(parent, 'dir1')
    dir11 = os.path.join(dir1, 'dir11')
    dir12 = os.path.join(dir1, 'dir12')
    dir2 = os.path.join(parent, 'dir2')
    os.mkdir(dir1)
    os.mkdir(dir11)
    os.mkdir(dir12)
    os.mkdir(dir2)
    open(os.path.join(dir1, 'file11'), 'w').close()
    open(os.path.join(dir1, 'file12'), 'w').close()
    open(os.path.join(dir11, 'file111'), 'w').close()
    open(os.path.join(dir2, 'file21'), 'w').close()

    tree_ = datasets.utils._tree(parent)

    # Check the tree
    # assert_equal(tree_[0]['dir1'][0]['dir11'][0], 'file111')
    # assert_equal(len(tree_[0]['dir1'][1]['dir12']), 0)
    # assert_equal(tree_[0]['dir1'][2], 'file11')
    # assert_equal(tree_[0]['dir1'][3], 'file12')
    # assert_equal(tree_[1]['dir2'][0], 'file21')
    # assert_equal(tree_[2], 'file1')
    # assert_equal(tree_[3], 'file2')
    assert tree_[0][1][0][1][0] == os.path.join(dir11, 'file111')
    assert len(tree_[0][1][1][1]) == 0
    assert tree_[0][1][2] == os.path.join(dir1, 'file11')
    assert tree_[0][1][3] == os.path.join(dir1, 'file12')
    assert tree_[1][1][0] == os.path.join(dir2, 'file21')
    assert tree_[2] == os.path.join(parent, 'file1')
    assert tree_[3] == os.path.join(parent, 'file2')

    # Clean
    shutil.rmtree(parent)


def test_movetree():
    # Create a dummy directory tree
    parent = mkdtemp()

    dir1 = os.path.join(parent, 'dir1')
    dir11 = os.path.join(dir1, 'dir11')
    dir12 = os.path.join(dir1, 'dir12')
    dir2 = os.path.join(parent, 'dir2')
    os.mkdir(dir1)
    os.mkdir(dir11)
    os.mkdir(dir12)
    os.mkdir(dir2)
    os.mkdir(os.path.join(dir2, 'dir12'))
    open(os.path.join(dir1, 'file11'), 'w').close()
    open(os.path.join(dir1, 'file12'), 'w').close()
    open(os.path.join(dir11, 'file111'), 'w').close()
    open(os.path.join(dir12, 'file121'), 'w').close()
    open(os.path.join(dir2, 'file21'), 'w').close()

    datasets.utils.movetree(dir1, dir2)

    assert not os.path.exists(dir11)
    assert not os.path.exists(dir12)
    assert not os.path.exists(os.path.join(dir1, 'file11'))
    assert not os.path.exists(os.path.join(dir1, 'file12'))
    assert not os.path.exists(os.path.join(dir11, 'file111'))
    assert not os.path.exists(os.path.join(dir12, 'file121'))
    dir11 = os.path.join(dir2, 'dir11')
    dir12 = os.path.join(dir2, 'dir12')

    assert os.path.exists(dir11)
    assert os.path.exists(dir12)
    assert os.path.exists(os.path.join(dir2, 'file11'))
    assert os.path.exists(os.path.join(dir2, 'file12'))
    assert os.path.exists(os.path.join(dir11, 'file111'))
    assert os.path.exists(os.path.join(dir12, 'file121'))


def test_filter_columns():
    # Create fake recarray
    value1 = np.arange(500)
    strings = np.asarray(['a', 'b', 'c'])
    value2 = strings[value1 % 3]

    values = np.asarray(list(zip(value1, value2)),
                        dtype=[('INT', int), ('STR', 'S1')])

    f = datasets.utils._filter_columns(values, {'INT': (23, 46)})
    assert np.sum(f) == 24

    f = datasets.utils._filter_columns(values, {'INT': [0, 9, (12, 24)]})
    assert np.sum(f) == 15

    value1 = value1 % 2
    values = np.asarray(list(zip(value1, value2)),
                        dtype=[('INT', int), ('STR', b'S1')])

    # No filter
    f = datasets.utils._filter_columns(values, [])
    assert np.sum(f) == 500

    f = datasets.utils._filter_columns(values, {'STR': b'b'})
    assert np.sum(f) == 167

    f = datasets.utils._filter_columns(values, {'STR': u'b'})
    assert np.sum(f) == 167

    f = datasets.utils._filter_columns(values, {'INT': 1, 'STR': b'b'})
    assert np.sum(f) == 84

    f = datasets.utils._filter_columns(values, {'INT': 1, 'STR': b'b'},
                                       combination='or')
    assert np.sum(f) == 333


def test_uncompress():
    # for each kind of compression, we create:
    # - a temporary directory (dtemp)
    # - a compressed object (ztemp)
    # - a temporary file-like object to compress into ztemp
    # we then uncompress the ztemp object into dtemp under the name ftemp
    # and check if ftemp exists
    dtemp = mkdtemp()
    ztemp = os.path.join(dtemp, 'test.zip')
    ftemp = 'test'
    try:
        with contextlib.closing(zipfile.ZipFile(ztemp, 'w')) as testzip:
            testzip.writestr(ftemp, ' ')
        datasets.utils._uncompress_file(ztemp, verbose=0)
        assert (os.path.exists(os.path.join(dtemp, ftemp)))
        shutil.rmtree(dtemp)

        dtemp = mkdtemp()
        ztemp = os.path.join(dtemp, 'test.tar')

        # Create dummy file in the dtemp folder, so that the finally statement
        # can easily remove it
        fd, temp = mkstemp(dir=dtemp)
        os.close(fd)
        with contextlib.closing(tarfile.open(ztemp, 'w')) as tar:
            tar.add(temp, arcname=ftemp)
        datasets.utils._uncompress_file(ztemp, verbose=0)
        assert (os.path.exists(os.path.join(dtemp, ftemp)))
        shutil.rmtree(dtemp)

        dtemp = mkdtemp()
        ztemp = os.path.join(dtemp, 'test.gz')
        gzip.open(ztemp, 'wb').close()
        datasets.utils._uncompress_file(ztemp, verbose=0)
        # test.gz gets uncompressed into test
        assert (os.path.exists(os.path.join(dtemp, 'test')))
        shutil.rmtree(dtemp)
    finally:
        # all temp files are created into dtemp except temp
        if os.path.exists(dtemp):
            shutil.rmtree(dtemp)


@pytest.mark.parametrize("should_cast_path_to_string", [False, True])
def test_fetch_file_overwrite(should_cast_path_to_string,
                              tmp_path, request_mocker):
    if should_cast_path_to_string:
        tmp_path = str(tmp_path)

    # overwrite non-exiting file.
    fil = datasets.utils._fetch_file(url='http://foo/', data_dir=str(tmp_path),
                                     verbose=0, overwrite=True)
    assert request_mocker.url_count == 1
    assert os.path.exists(fil)
    with open(fil, 'r') as fp:
        assert fp.read() == ''

    # Modify content
    with open(fil, 'w') as fp:
        fp.write('some content')

    # Don't overwrite existing file.
    fil = datasets.utils._fetch_file(url='http://foo/', data_dir=str(tmp_path),
                                     verbose=0, overwrite=False)

    assert request_mocker.url_count == 1
    assert os.path.exists(fil)
    with open(fil, 'r') as fp:
        assert fp.read() == 'some content'

    # Overwrite existing file.
    # Overwrite existing file.
    fil = datasets.utils._fetch_file(url='http://foo/', data_dir=str(tmp_path),
                                     verbose=0, overwrite=True)
    assert request_mocker.url_count == 2
    assert os.path.exists(fil)
    with open(fil, 'r') as fp:
        assert fp.read() == ''


@pytest.mark.parametrize("should_cast_path_to_string", [False, True])
def test_fetch_files_use_session(should_cast_path_to_string,
                                 tmp_path, request_mocker):
    if should_cast_path_to_string:
        tmp_path = str(tmp_path)

    # regression test for https://github.com/nilearn/nilearn/issues/2863
    session = MagicMock()
    datasets.utils._fetch_files(
        files=[
            ("example1", "https://example.org/example1", {"overwrite": True}),
            ("example2", "https://example.org/example2", {"overwrite": True}),
        ],
        data_dir=str(tmp_path),
        session=session,
    )
    assert session.send.call_count == 2


@pytest.mark.parametrize("should_cast_path_to_string", [False, True])
def test_fetch_files_overwrite(should_cast_path_to_string,
                               tmp_path, request_mocker):
    if should_cast_path_to_string:
        tmp_path = str(tmp_path)

    # overwrite non-exiting file.
    files = ('1.txt', 'http://foo/1.txt')
    fil = datasets.utils._fetch_files(data_dir=str(tmp_path), verbose=0,
                                      files=[files + (dict(overwrite=True),)])
    assert request_mocker.url_count == 1
    assert os.path.exists(fil[0])
    with open(fil[0], 'r') as fp:
        assert fp.read() == ''

    # Modify content
    with open(fil[0], 'w') as fp:
        fp.write('some content')

    # Don't overwrite existing file.
    fil = datasets.utils._fetch_files(data_dir=str(tmp_path), verbose=0,
                                      files=[files + (dict(overwrite=False),)])
    assert request_mocker.url_count == 1
    assert os.path.exists(fil[0])
    with open(fil[0], 'r') as fp:
        assert fp.read() == 'some content'

    # Overwrite existing file.
    fil = datasets.utils._fetch_files(data_dir=str(tmp_path), verbose=0,
                                      files=[files + (dict(overwrite=True),)])
    assert request_mocker.url_count == 2
    assert os.path.exists(fil[0])
    with open(fil[0], 'r') as fp:
        assert fp.read() == ''


@pytest.mark.skipif(not BOTO_INSTALLED,
                    reason='Boto3  missing; necessary for this test')
def test_make_fresh_openneuro_dataset_urls_index(tmp_path, request_mocker):
    dataset_version = 'ds000030_R1.0.4'
    data_prefix = '{}/{}/uncompressed'.format(
        dataset_version.split('_')[0], dataset_version)
    data_dir = _get_dataset_dir(data_prefix, data_dir=str(tmp_path),
                                verbose=1)
    url_file = os.path.join(data_dir,
                            'nistats_fetcher_openneuro_dataset_urls.json',
                            )
    # Prepare url files for subject and filter tests
    file_list = [data_prefix + '/stuff.html',
                 data_prefix + '/sub-xxx.html',
                 data_prefix + '/sub-yyy.html',
                 data_prefix + '/sub-xxx/ses-01_task-rest.txt',
                 data_prefix + '/sub-xxx/ses-01_task-other.txt',
                 data_prefix + '/sub-xxx/ses-02_task-rest.txt',
                 data_prefix + '/sub-xxx/ses-02_task-other.txt',
                 data_prefix + '/sub-yyy/ses-01.txt',
                 data_prefix + '/sub-yyy/ses-02.txt']
    with open(url_file, 'w') as f:
        json.dump(file_list, f)

    # Only 1 subject and not subject specific files get downloaded
    datadir, dl_files = make_fresh_openneuro_dataset_urls_index(
        str(tmp_path), dataset_version)
    assert isinstance(datadir, str)
    assert isinstance(dl_files, list)
    assert len(dl_files) == len(file_list)


def test_naive_ftp_adapter():
    sender = utils._NaiveFTPAdapter()
    resp = sender.send(
        requests.Request("GET", "ftp://example.com").prepare())
    resp.close()
    resp.raw.close.assert_called_with()
    urllib.request.OpenerDirector.open.side_effect = urllib.error.URLError(
        "timeout")
    with pytest.raises(requests.RequestException, match="timeout"):
        resp = sender.send(
            requests.Request("GET", "ftp://example.com").prepare())
