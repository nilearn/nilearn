"""Test the datasets module."""

# Author: Alexandre Abraham

import contextlib
import gzip
import os
import shutil
import tarfile
import urllib
import zipfile
from pathlib import Path
from tempfile import mkdtemp, mkstemp
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests

from nilearn.datasets import _utils

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = Path(currdir, "data")

DATASET_NAMES = {
    "aal",
    "ABIDE_pcp",
    "adhd",
    "allen_rsn_2011",
    "basc_multiscale_2015",
    "bids_langloc",
    "brainomics_localizer",
    "craddock_2012",
    "destrieux_surface",
    "development_fmri",
    "difumo_atlases",
    "dosenbach_2010",
    "fiac",
    "fsaverage3",
    "fsaverage4",
    "fsaverage5",
    "fsaverage6",
    "fsaverage",
    "harvard_oxford",
    "haxby2001",
    "icbm152_2009",
    "language_localizer_demo",
    "localizer_first_level",
    "juelich",
    "Megatrawls",
    "mixed_gambles",
    "miyawaki2008",
    "msdl_atlas",
    "neurovault",
    "nki_enhanced_surface",
    "oasis1",
    "pauli_2017",
    "power_2011",
    "spm_auditory",
    "spm_multimodal",
    "schaefer_2018",
    "smith_2009",
    "talairach_atlas",
    "yeo_2011",
}


def test_get_dataset_descr_warning():
    """Tests that function ``get_dataset_descr()`` gives a warning \
       when no description is available.
    """
    with pytest.warns(
        UserWarning, match="Could not find dataset description."
    ):
        descr = _utils.get_dataset_descr("")

    assert descr == ""


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_get_dataset_descr(name):
    """Test function ``get_dataset_descr()``."""
    descr = _utils.get_dataset_descr(name)

    assert isinstance(descr, str)
    assert len(descr) > 0


def test_get_dataset_dir(tmp_path):
    # testing folder creation under different environments, enforcing
    # a custom clean install
    os.environ.pop("NILEARN_DATA", None)
    os.environ.pop("NILEARN_SHARED_DATA", None)

    expected_base_dir = os.path.expanduser("~/nilearn_data")
    data_dir = _utils.get_dataset_dir("test", verbose=0)

    assert data_dir == Path(expected_base_dir, "test")
    assert os.path.exists(data_dir)

    shutil.rmtree(data_dir)

    expected_base_dir = str(tmp_path / "test_nilearn_data")
    os.environ["NILEARN_DATA"] = expected_base_dir
    data_dir = _utils.get_dataset_dir("test", verbose=0)

    assert data_dir == Path(expected_base_dir, "test")
    assert os.path.exists(data_dir)

    shutil.rmtree(data_dir)

    expected_base_dir = str(tmp_path / "nilearn_shared_data")
    os.environ["NILEARN_SHARED_DATA"] = expected_base_dir
    data_dir = _utils.get_dataset_dir("test", verbose=0)

    assert data_dir == Path(expected_base_dir, "test")
    assert os.path.exists(data_dir)

    shutil.rmtree(data_dir)

    # Verify exception for a path which exists and is a file
    test_file = str(tmp_path / "some_file")
    with open(test_file, "w") as out:
        out.write("abcfeg")

    with pytest.raises(
        OSError,
        match="Nilearn tried to store the dataset in the following "
        "directories, but",
    ):
        _utils.get_dataset_dir("test", test_file, verbose=0)


def test_add_readme_to_default_data_locations(tmp_path):
    assert not (tmp_path / "README.md").exists()
    _utils.get_dataset_dir(dataset_name="test", verbose=0, data_dir=tmp_path)
    assert (tmp_path / "README.md").exists()


@pytest.mark.parametrize("should_cast_path_to_string", [False, True])
def test_get_dataset_dir_path_as_str(should_cast_path_to_string, tmp_path):
    expected_base_dir = tmp_path / "env_data"
    expected_dataset_dir = expected_base_dir / "test"
    if should_cast_path_to_string:
        expected_dataset_dir = str(expected_dataset_dir)
    data_dir = _utils.get_dataset_dir(
        "test", default_paths=[expected_dataset_dir], verbose=0
    )

    assert data_dir == str(expected_dataset_dir)
    assert os.path.exists(data_dir)

    shutil.rmtree(data_dir)


def test_get_dataset_dir_write_access(tmp_path):
    os.environ.pop("NILEARN_SHARED_DATA", None)

    no_write = str(tmp_path / "no_write")
    os.makedirs(no_write)
    os.chmod(no_write, 0o400)

    expected_base_dir = str(tmp_path / "nilearn_shared_data")
    os.environ["NILEARN_SHARED_DATA"] = expected_base_dir
    data_dir = _utils.get_dataset_dir(
        "test", default_paths=[no_write], verbose=0
    )

    # Non writeable dir is returned because dataset may be in there.
    assert data_dir == no_write
    assert os.path.exists(data_dir)

    os.chmod(no_write, 0o600)
    shutil.rmtree(data_dir)


def test_md5_sum_file():
    # Create dummy temporary file
    out, f = mkstemp()
    os.write(out, b"abcfeg")
    os.close(out)

    assert _utils._md5_sum_file(f) == "18f32295c556b2a1a3a8e68fe1ad40f7"

    os.remove(f)


def test_read_md5_sum_file():
    # Create dummy temporary file
    out, f = mkstemp()
    os.write(
        out,
        b"20861c8c3fe177da19a7e9539a5dbac  /tmp/test\n"
        b"70886dcabe7bf5c5a1c24ca24e4cbd94  test/some_image.nii",
    )
    os.close(out)
    h = _utils.read_md5_sum_file(f)

    assert "/tmp/test" in h
    assert "/etc/test" not in h
    assert h["test/some_image.nii"] == "70886dcabe7bf5c5a1c24ca24e4cbd94"
    assert h["/tmp/test"] == "20861c8c3fe177da19a7e9539a5dbac"

    os.remove(f)


def test_tree():
    # Create a dummy directory tree
    parent = mkdtemp()

    open(Path(parent, "file1"), "w").close()
    open(Path(parent, "file2"), "w").close()
    dir1 = Path(parent, "dir1")
    dir11 = Path(dir1, "dir11")
    dir12 = Path(dir1, "dir12")
    dir2 = Path(parent, "dir2")
    os.mkdir(dir1)
    os.mkdir(dir11)
    os.mkdir(dir12)
    os.mkdir(dir2)
    open(Path(dir1, "file11"), "w").close()
    open(Path(dir1, "file12"), "w").close()
    open(Path(dir11, "file111"), "w").close()
    open(Path(dir2, "file21"), "w").close()

    tree_ = _utils.tree(parent)

    # Check the tree
    # assert_equal(tree_[0]['dir1'][0]['dir11'][0], 'file111')
    # assert_equal(len(tree_[0]['dir1'][1]['dir12']), 0)
    # assert_equal(tree_[0]['dir1'][2], 'file11')
    # assert_equal(tree_[0]['dir1'][3], 'file12')
    # assert_equal(tree_[1]['dir2'][0], 'file21')
    # assert_equal(tree_[2], 'file1')
    # assert_equal(tree_[3], 'file2')
    assert tree_[0][1][0][1][0] == Path(dir11, "file111")
    assert len(tree_[0][1][1][1]) == 0
    assert tree_[0][1][2] == Path(dir1, "file11")
    assert tree_[0][1][3] == Path(dir1, "file12")
    assert tree_[1][1][0] == Path(dir2, "file21")
    assert tree_[2] == Path(parent, "file1")
    assert tree_[3] == Path(parent, "file2")

    # Clean
    shutil.rmtree(parent)


def test_movetree():
    # Create a dummy directory tree
    parent = mkdtemp()

    dir1 = Path(parent, "dir1")
    dir11 = Path(dir1, "dir11")
    dir12 = Path(dir1, "dir12")
    dir2 = Path(parent, "dir2")
    os.mkdir(dir1)
    os.mkdir(dir11)
    os.mkdir(dir12)
    os.mkdir(dir2)
    os.mkdir(Path(dir2, "dir12"))
    open(Path(dir1, "file11"), "w").close()
    open(Path(dir1, "file12"), "w").close()
    open(Path(dir11, "file111"), "w").close()
    open(Path(dir12, "file121"), "w").close()
    open(Path(dir2, "file21"), "w").close()

    _utils.movetree(dir1, dir2)

    assert not os.path.exists(dir11)
    assert not os.path.exists(dir12)
    assert not os.path.exists(Path(dir1, "file11"))
    assert not os.path.exists(Path(dir1, "file12"))
    assert not os.path.exists(Path(dir11, "file111"))
    assert not os.path.exists(Path(dir12, "file121"))

    dir11 = Path(dir2, "dir11")
    dir12 = Path(dir2, "dir12")

    assert os.path.exists(dir11)
    assert os.path.exists(dir12)
    assert os.path.exists(Path(dir2, "file11"))
    assert os.path.exists(Path(dir2, "file12"))
    assert os.path.exists(Path(dir11, "file111"))
    assert os.path.exists(Path(dir12, "file121"))


def test_filter_columns():
    # Create fake recarray
    value1 = np.arange(500)
    strings = np.asarray(["a", "b", "c"])
    value2 = strings[value1 % 3]

    values = np.asarray(
        list(zip(value1, value2)), dtype=[("INT", int), ("STR", "S1")]
    )

    f = _utils.filter_columns(values, {"INT": (23, 46)})

    assert np.sum(f) == 24

    f = _utils.filter_columns(values, {"INT": [0, 9, (12, 24)]})

    assert np.sum(f) == 15

    value1 = value1 % 2
    values = np.asarray(
        list(zip(value1, value2)), dtype=[("INT", int), ("STR", b"S1")]
    )

    # No filter
    f = _utils.filter_columns(values, [])

    assert np.sum(f) == 500

    f = _utils.filter_columns(values, {"STR": b"b"})

    assert np.sum(f) == 167

    f = _utils.filter_columns(values, {"STR": "b"})

    assert np.sum(f) == 167

    f = _utils.filter_columns(values, {"INT": 1, "STR": b"b"})

    assert np.sum(f) == 84

    f = _utils.filter_columns(
        values, {"INT": 1, "STR": b"b"}, combination="or"
    )

    assert np.sum(f) == 333


def test_uncompress():
    # for each kind of compression, we create:
    # - a temporary directory (dtemp)
    # - a compressed object (ztemp)
    # - a temporary file-like object to compress into ztemp
    # we then uncompress the ztemp object into dtemp under the name ftemp
    # and check if ftemp exists
    dtemp = mkdtemp()
    ztemp = Path(dtemp, "test.zip")
    ftemp = "test"
    try:
        with contextlib.closing(zipfile.ZipFile(ztemp, "w")) as testzip:
            testzip.writestr(ftemp, " ")
        _utils.uncompress_file(ztemp, verbose=0)

        assert os.path.exists(Path(dtemp, ftemp))

        shutil.rmtree(dtemp)

        dtemp = mkdtemp()
        ztemp = Path(dtemp, "test.tar")

        # Create dummy file in the dtemp folder, so that the finally statement
        # can easily remove it
        fd, temp = mkstemp(dir=dtemp)
        os.close(fd)
        with contextlib.closing(tarfile.open(ztemp, "w")) as tar:
            tar.add(temp, arcname=ftemp)
        _utils.uncompress_file(ztemp, verbose=0)

        assert os.path.exists(Path(dtemp, ftemp))

        shutil.rmtree(dtemp)

        dtemp = mkdtemp()
        ztemp = Path(dtemp, "test.gz")
        gzip.open(ztemp, "wb").close()
        _utils.uncompress_file(ztemp, verbose=0)

        # test.gz gets uncompressed into test
        assert os.path.exists(Path(dtemp, "test"))

        shutil.rmtree(dtemp)

    finally:
        # all temp files are created into dtemp except temp
        if os.path.exists(dtemp):
            shutil.rmtree(dtemp)


def test_safe_extract(tmp_path):
    # Test vulnerability patch by mimicking path traversal
    ztemp = Path(tmp_path, "test.tar")
    in_archive_file = tmp_path / "something.txt"
    in_archive_file.write_text("hello")
    with contextlib.closing(tarfile.open(ztemp, "w")) as tar:
        arcname = os.path.normpath("../test.tar")
        tar.add(in_archive_file, arcname=arcname)

    with pytest.raises(
        Exception, match="Attempted Path Traversal in Tar File"
    ):
        _utils.uncompress_file(ztemp, verbose=0)


@pytest.mark.parametrize("should_cast_path_to_string", [False, True])
def test_fetch_file_overwrite(
    should_cast_path_to_string, tmp_path, request_mocker
):
    if should_cast_path_to_string:
        tmp_path = str(tmp_path)

    # overwrite non-exiting file.
    fil = _utils.fetch_single_file(
        url="http://foo/", data_dir=str(tmp_path), verbose=0, overwrite=True
    )

    assert request_mocker.url_count == 1
    assert os.path.exists(fil)
    with open(fil) as fp:
        assert fp.read() == ""

    # Modify content
    with open(fil, "w") as fp:
        fp.write("some content")

    # Don't overwrite existing file.
    fil = _utils.fetch_single_file(
        url="http://foo/", data_dir=str(tmp_path), verbose=0, overwrite=False
    )

    assert request_mocker.url_count == 1
    assert os.path.exists(fil)
    with open(fil) as fp:
        assert fp.read() == "some content"

    # Overwrite existing file.
    fil = _utils.fetch_single_file(
        url="http://foo/", data_dir=str(tmp_path), verbose=0, overwrite=True
    )

    assert request_mocker.url_count == 2
    assert os.path.exists(fil)
    with open(fil) as fp:
        assert fp.read() == ""


@pytest.mark.parametrize("should_cast_path_to_string", [False, True])
def test_fetch_files_use_session(
    should_cast_path_to_string, tmp_path, request_mocker
):
    if should_cast_path_to_string:
        tmp_path = str(tmp_path)

    # regression test for https://github.com/nilearn/nilearn/issues/2863
    session = MagicMock()
    _utils.fetch_files(
        files=[
            ("example1", "https://example.org/example1", {"overwrite": True}),
            ("example2", "https://example.org/example2", {"overwrite": True}),
        ],
        data_dir=str(tmp_path),
        session=session,
    )

    assert session.send.call_count == 2


@pytest.mark.parametrize("should_cast_path_to_string", [False, True])
def test_fetch_files_overwrite(
    should_cast_path_to_string, tmp_path, request_mocker
):
    if should_cast_path_to_string:
        tmp_path = str(tmp_path)

    # overwrite non-exiting file.
    files = ("1.txt", "http://foo/1.txt")
    fil = _utils.fetch_files(
        data_dir=str(tmp_path),
        verbose=0,
        files=[files + (dict(overwrite=True),)],
    )

    assert request_mocker.url_count == 1
    assert os.path.exists(fil[0])
    with open(fil[0]) as fp:
        assert fp.read() == ""

    # Modify content
    with open(fil[0], "w") as fp:
        fp.write("some content")

    # Don't overwrite existing file.
    fil = _utils.fetch_files(
        data_dir=str(tmp_path),
        verbose=0,
        files=[files + (dict(overwrite=False),)],
    )

    assert request_mocker.url_count == 1
    assert os.path.exists(fil[0])
    with open(fil[0]) as fp:
        assert fp.read() == "some content"

    # Overwrite existing file.
    fil = _utils.fetch_files(
        data_dir=str(tmp_path),
        verbose=0,
        files=[files + (dict(overwrite=True),)],
    )

    assert request_mocker.url_count == 2
    assert os.path.exists(fil[0])
    with open(fil[0]) as fp:
        assert fp.read() == ""


def test_naive_ftp_adapter():
    sender = _utils._NaiveFTPAdapter()
    resp = sender.send(requests.Request("GET", "ftp://example.com").prepare())
    resp.close()
    resp.raw.close.assert_called_with()
    urllib.request.OpenerDirector.open.side_effect = urllib.error.URLError(
        "timeout"
    )
    with pytest.raises(requests.RequestException, match="timeout"):
        resp = sender.send(
            requests.Request("GET", "ftp://example.com").prepare()
        )


# TODO remove for release 0.13.0
from nilearn.datasets import utils


def test_load_sample_motor_activation_image():
    with pytest.warns(
        DeprecationWarning,
        match="Please import this function from 'nilearn.datasets.func'",
    ):
        utils.load_sample_motor_activation_image()
