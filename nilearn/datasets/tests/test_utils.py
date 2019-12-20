"""
Test the datasets module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import contextlib
import os
import shutil
import numpy as np
import zipfile
import tarfile
import gzip
from tempfile import mkdtemp, mkstemp

import pytest

from nilearn import datasets
from nilearn._utils.testing import (mock_request, wrap_chunk_read_,
                                    FetchFilesMock)

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')
url_request = None
file_mock = None


@pytest.fixture()
def request_mock():
    setup_mock()
    yield
    teardown_mock()


def setup_mock(utils_mod=datasets.utils, dataset_mod=datasets.utils):
    global original_url_request
    global mock_url_request
    mock_url_request = mock_request()
    original_url_request = utils_mod._urllib.request
    utils_mod._urllib.request = mock_url_request

    global original_chunk_read
    global mock_chunk_read
    mock_chunk_read = wrap_chunk_read_(utils_mod._chunk_read_)
    original_chunk_read = utils_mod._chunk_read_
    utils_mod._chunk_read_ = mock_chunk_read

    global original_fetch_files
    global mock_fetch_files
    mock_fetch_files = FetchFilesMock()
    original_fetch_files = dataset_mod._fetch_files
    dataset_mod._fetch_files = mock_fetch_files


def teardown_mock(utils_mod=datasets.utils, dataset_mod=datasets.utils):
    global original_url_request
    utils_mod._urllib.request = original_url_request

    global original_chunk_read
    utils_mod.chunk_read_ = original_chunk_read

    global original_fetch_files
    dataset_mod._fetch_files = original_fetch_files


def test_get_dataset_dir(tmp_path):
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

    expected_base_dir = str(tmp_path / 'env_data')
    expected_dataset_dir = os.path.join(expected_base_dir, 'test')
    data_dir = datasets.utils._get_dataset_dir(
        'test', default_paths=[expected_dataset_dir], verbose=0)
    assert data_dir == os.path.join(expected_base_dir, 'test')
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


def test_fetch_file_overwrite(tmp_path, request_mock):
    # overwrite non-exiting file.
    fil = datasets.utils._fetch_file(url='http://foo/', data_dir=str(tmp_path),
                                     verbose=0, overwrite=True)
    assert len(mock_url_request.urls) == 1
    assert os.path.exists(fil)
    with open(fil, 'r') as fp:
        assert fp.read() == ''

    # Modify content
    with open(fil, 'w') as fp:
        fp.write('some content')

    # Don't overwrite existing file.
    fil = datasets.utils._fetch_file(url='http://foo/', data_dir=str(tmp_path),
                                     verbose=0, overwrite=False)
    assert len(mock_url_request.urls) == 1
    assert os.path.exists(fil)
    with open(fil, 'r') as fp:
        assert fp.read() == 'some content'

    # Overwrite existing file.
    # Overwrite existing file.
    fil = datasets.utils._fetch_file(url='http://foo/', data_dir=str(tmp_path),
                                     verbose=0, overwrite=True)
    assert len(mock_url_request.urls) == 1
    assert os.path.exists(fil)
    with open(fil, 'r') as fp:
        assert fp.read() == ''


def test_fetch_files_overwrite(tmp_path, request_mock):
    # overwrite non-exiting file.
    files = ('1.txt', 'http://foo/1.txt')
    fil = datasets.utils._fetch_files(data_dir=str(tmp_path), verbose=0,
                                      files=[files + (dict(overwrite=True),)])
    assert len(mock_url_request.urls) == 1
    assert os.path.exists(fil[0])
    with open(fil[0], 'r') as fp:
        assert fp.read() == ''

    # Modify content
    with open(fil[0], 'w') as fp:
        fp.write('some content')

    # Don't overwrite existing file.
    fil = datasets.utils._fetch_files(data_dir=str(tmp_path), verbose=0,
                                      files=[files + (dict(overwrite=False),)])
    assert len(mock_url_request.urls) == 1
    assert os.path.exists(fil[0])
    with open(fil[0], 'r') as fp:
        assert fp.read() == 'some content'

    # Overwrite existing file.
    fil = datasets.utils._fetch_files(data_dir=str(tmp_path), verbose=0,
                                      files=[files + (dict(overwrite=True),)])
    assert len(mock_url_request.urls) == 1
    assert os.path.exists(fil[0])
    with open(fil[0], 'r') as fp:
        assert fp.read() == ''
