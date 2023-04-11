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
from unittest.mock import MagicMock
from tempfile import mkdtemp, mkstemp


import numpy as np
import pytest
import requests

from nilearn import datasets
from nilearn.datasets.utils import _get_dataset_dir, _get_dataset_descr
from nilearn.datasets import utils
from nilearn.image import load_img


currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')

DATASET_NAMES = set([
    "aal_SPM12", "ABIDE_pcp", "adhd", "allen_rsn_2011",
    "basc_multiscale_2015", "brainomics_localizer", "cobre", "craddock_2012",
    "destrieux_surface", "development_fmri", "difumo_atlases",
    "dosenbach_2010", "fsaverage3", "fsaverage4", "fsaverage5",
    "fsaverage6", "fsaverage", "haxby2001",
    "icbm152_2009", "Megatrawls", "miyawaki2008", "msdl_atlas",
    "neurovault", "nki_enhanced_surface", "nyu_rest", "oasis1",
    "pauli_2017", "power_2011", "schaefer_2018", "smith_2009",
    "talairach_atlas", "yeo_2011"
])


def test_get_dataset_descr_warning():
    """Tests that function ``_get_dataset_descr()`` gives a warning
    when no description is available.
    """
    with pytest.warns(UserWarning,
                      match="Could not find dataset description."):
        descr = _get_dataset_descr("")
    assert descr == ""


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_get_dataset_descr(name):
    """Test function ``_get_dataset_descr()``."""
    descr = _get_dataset_descr(name)
    assert isinstance(descr, str)
    assert len(descr) > 0


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


def test_safe_extract(tmp_path):
    # Test vulnerability patch by mimicking path traversal
    ztemp = os.path.join(tmp_path, 'test.tar')
    in_archive_file = tmp_path / "something.txt"
    in_archive_file.write_text("hello")
    with contextlib.closing(tarfile.open(ztemp, 'w')) as tar:
        arcname = os.path.normpath('../test.tar')
        tar.add(in_archive_file, arcname=arcname)
    with pytest.raises(
            Exception, match="Attempted Path Traversal in Tar File"
    ):
        datasets.utils._uncompress_file(ztemp, verbose=0)


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


def test_load_sample_motor_activation_image():
    path_img = utils.load_sample_motor_activation_image()
    assert os.path.exists(path_img)
    assert load_img(path_img)
