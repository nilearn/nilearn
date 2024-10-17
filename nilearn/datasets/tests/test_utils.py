"""Test the datasets module."""

# Author: Alexandre Abraham

import contextlib
import gzip
import os
import shutil
import tarfile
import urllib
from pathlib import Path
from tempfile import mkstemp
from unittest.mock import MagicMock
from zipfile import ZipFile

import numpy as np
import pytest
import requests

from nilearn.datasets import _utils

datadir = _utils.PACKAGE_DIRECTORY / "data"

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

    expected_base_dir = Path("~/nilearn_data").expanduser()
    data_dir = _utils.get_dataset_dir("test", verbose=0)

    assert data_dir == str(expected_base_dir / "test")
    assert os.path.exists(data_dir)

    shutil.rmtree(data_dir)

    expected_base_dir = tmp_path / "test_nilearn_data"
    os.environ["NILEARN_DATA"] = str(expected_base_dir)
    data_dir = _utils.get_dataset_dir("test", verbose=0)

    assert data_dir == str(expected_base_dir / "test")
    assert os.path.exists(data_dir)

    shutil.rmtree(data_dir)

    expected_base_dir = tmp_path / "nilearn_shared_data"
    os.environ["NILEARN_SHARED_DATA"] = str(expected_base_dir)
    data_dir = _utils.get_dataset_dir("test", verbose=0)

    assert data_dir == str(expected_base_dir / "test")
    assert os.path.exists(data_dir)

    shutil.rmtree(data_dir)

    # Verify exception for a path which exists and is a file
    test_file = tmp_path / "some_file"
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

    no_write = tmp_path / "no_write"
    no_write.mkdir(parents=True)
    os.chmod(no_write, 0o400)

    expected_base_dir = tmp_path / "nilearn_shared_data"
    os.environ["NILEARN_SHARED_DATA"] = str(expected_base_dir)
    data_dir = _utils.get_dataset_dir(
        "test", default_paths=[no_write], verbose=0
    )

    # Non writeable dir is returned because dataset may be in there.
    assert data_dir == str(no_write)
    assert os.path.exists(data_dir)

    os.chmod(no_write, 0o600)
    shutil.rmtree(data_dir)


def test_get_dataset_dir_symlink(tmp_path):
    expected_linked_dir = tmp_path / "linked"
    expected_linked_dir.mkdir(parents=True)
    expected_base_dir = tmp_path / "env_data"
    expected_base_dir.mkdir()
    symlink_dir = expected_base_dir / "test"
    symlink_dir.symlink_to(expected_linked_dir)

    assert symlink_dir.exists()

    data_dir = _utils.get_dataset_dir(
        "test", default_paths=[symlink_dir], verbose=0
    )

    assert data_dir == str(expected_linked_dir)
    assert os.path.exists(data_dir)


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


def test_tree(tmp_path):
    dir1 = tmp_path / "dir1"
    dir11 = dir1 / "dir11"
    dir12 = dir1 / "dir12"
    dir2 = tmp_path / "dir2"

    dir1.mkdir()
    dir11.mkdir()
    dir12.mkdir()
    dir2.mkdir()

    (tmp_path / "file1").touch()
    (tmp_path / "file2").touch()
    (dir1 / "file11").touch()
    (dir1 / "file12").touch()
    (dir11 / "file111").touch()
    (dir2 / "file21").touch()

    # test for list return value
    tree_ = _utils.tree(tmp_path)

    # Check the tree
    assert type(tree_[0]) is tuple
    assert type(tree_[0][1]) is list
    assert type(tree_[0][1][0]) is tuple
    assert type(tree_[1]) is tuple
    assert type(tree_[1][1]) is list
    assert tree_[0][1][0][1][0] == str(dir11 / "file111")
    assert len(tree_[0][1][1][1]) == 0
    assert tree_[0][1][2] == str(dir1 / "file11")
    assert tree_[0][1][3] == str(dir1 / "file12")
    assert tree_[1][1][0] == str(dir2 / "file21")
    assert tree_[2] == str(tmp_path / "file1")
    assert tree_[3] == str(tmp_path / "file2")

    # test for dictionary return value
    tree_ = _utils.tree(tmp_path, dictionary=True)

    # Check the tree
    assert type(tree_[dir1.name]) is dict
    assert type(tree_[dir1.name][dir11.name]) is list
    assert len(tree_[dir1.name][dir12.name]) == 0
    assert type(tree_[dir2.name]) is list
    assert type(tree_["."]) is list
    assert tree_[dir1.name][dir11.name][0] == str(dir11 / "file111")
    assert tree_[dir1.name]["."][0] == str(dir1 / "file11")
    assert tree_[dir1.name]["."][1] == str(dir1 / "file12")
    assert tree_[dir2.name][0] == str(dir2 / "file21")
    assert tree_["."] == [str(tmp_path / "file1"), str(tmp_path / "file2")]


def test_movetree(tmp_path):
    """Tests nilearn.dataset._utils.movetree."""
    dir1 = tmp_path / "dir1"
    dir111 = dir1 / "dir11"
    dir112 = dir1 / "dir12"
    dir2 = tmp_path / "dir2"
    dir212 = dir2 / "dir12"

    dir1.mkdir()
    dir111.mkdir()
    dir112.mkdir()
    dir2.mkdir()
    dir212.mkdir()

    (dir1 / "file11").touch()
    (dir1 / "file12").touch()
    (dir111 / "file1111").touch()
    (dir112 / "file1121").touch()
    (dir2 / "file21").touch()

    _utils.movetree(dir1, dir2)

    assert not dir111.exists()
    assert not dir112.exists()
    assert not (dir1 / "file11").exists()
    assert not (dir1 / "file12").exists()
    assert not (dir111 / "file1111").exists()
    assert not (dir112 / "file1121").exists()

    dir211 = dir2 / "dir11"
    dir212 = dir2 / "dir12"

    assert dir211.exists()
    assert dir212.exists()
    assert (dir2 / "file21").exists()
    assert (dir2 / "file11").exists()
    assert (dir2 / "file12").exists()
    assert (dir211 / "file1111").exists()
    assert (dir212 / "file1121").exists()


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


@pytest.mark.parametrize(
    "ext, mode", [("tar", "w"), ("tar.gz", "w:gz"), ("tgz", "w:gz")]
)
def test_uncompress_tar(tmp_path, ext, mode):
    """Tests nilearn.dataset._utils.uncompress_file for tar files."""
    # for each kind of compression, we create:
    # - a compressed object (ztemp)
    # - a temporary file-like object to compress into ztemp
    # we then uncompress the ztemp object into dtemp under the name ftemp
    # and check if ftemp exists
    ztemp = tmp_path / f"test.{ext}"
    ftemp = "test"
    with contextlib.closing(tarfile.open(ztemp, mode)) as testtar:
        temp = tmp_path / ftemp
        temp.write_text(ftemp)
        testtar.add(temp)

    _utils.uncompress_file(ztemp, verbose=0)
    assert (tmp_path / ftemp).exists()


def test_uncompress_zip(tmp_path):
    """Tests nilearn.dataset._utils.uncompress_file for zip files."""
    # for each kind of compression, we create:
    # - a compressed object (ztemp)
    # - a temporary file-like object to compress into ztemp
    # we then uncompress the ztemp object into dtemp under the name ftemp
    # and check if ftemp exists
    ztemp = tmp_path / "test.zip"
    ftemp = "test"
    with contextlib.closing(ZipFile(ztemp, "w")) as testzip:
        testzip.writestr(ftemp, " ")

    _utils.uncompress_file(ztemp, verbose=0)
    assert (tmp_path / ftemp).exists()


@pytest.mark.parametrize("ext", [".gz", ""])
def test_uncompress_gzip(tmp_path, ext):
    """Tests nilearn.dataset._utils.uncompress_file for gzip files."""
    # for each kind of compression, we create:
    # - a compressed object (ztemp)
    # - a temporary file-like object to compress into ztemp
    # we then uncompress the ztemp object into dtemp under the name ftemp
    # and check if ftemp exists
    ztemp = tmp_path / f"test{ext}"
    ftemp = "test"

    with gzip.open(ztemp, "wb") as testgzip:
        testgzip.write(ftemp.encode())

    _utils.uncompress_file(ztemp, verbose=0)
    assert (tmp_path / ftemp).exists()


def test_safe_extract(tmp_path):
    # Test vulnerability patch by mimicking path traversal
    ztemp = os.path.join(tmp_path, "test.tar")
    in_archive_file = tmp_path / "something.txt"
    in_archive_file.write_text("hello")
    with contextlib.closing(tarfile.open(ztemp, "w")) as tar:
        arcname = "../test.tar"
        tar.add(in_archive_file, arcname=arcname)

    with pytest.raises(
        Exception, match="Attempted Path Traversal in Tar File"
    ):
        _utils.uncompress_file(ztemp, verbose=0)


def test_fetch_file_part(tmp_path):
    url = "http://foo/temp.txt"
    file_full = tmp_path / "temp.txt"
    file_part = tmp_path / "temp.txt.part"
    file_part.touch()

    _utils.fetch_single_file(
        url=url, data_dir=tmp_path, verbose=0, resume=True
    )

    assert file_full.exists()

    file_full.unlink()
    assert not file_full.exists()
    assert not file_part.exists()

    # test for overwrite
    url = "http://foo/temp.txt"
    file_part.touch()

    _utils.fetch_single_file(
        url=url, data_dir=tmp_path, verbose=0, resume=True, overwrite=True
    )

    assert file_full.exists()


@pytest.mark.parametrize("should_cast_path_to_string", [False, True])
def test_fetch_file_overwrite(
    should_cast_path_to_string, tmp_path, request_mocker
):
    if should_cast_path_to_string:
        tmp_path = str(tmp_path)

    # overwrite non-exiting file.
    fil = _utils.fetch_single_file(
        url="http://foo/", data_dir=tmp_path, verbose=0, overwrite=True
    )

    assert request_mocker.url_count == 1
    assert fil.exists()
    with open(fil) as fp:
        assert fp.read() == ""

    # Modify content
    with open(fil, "w") as fp:
        fp.write("some content")

    # Don't overwrite existing file.
    fil = _utils.fetch_single_file(
        url="http://foo/", data_dir=tmp_path, verbose=0, overwrite=False
    )

    assert request_mocker.url_count == 1
    assert fil.exists()
    with open(fil) as fp:
        assert fp.read() == "some content"

    # Overwrite existing file.
    fil = _utils.fetch_single_file(
        url="http://foo/", data_dir=tmp_path, verbose=0, overwrite=True
    )

    assert request_mocker.url_count == 2
    assert fil.exists()
    with open(fil) as fp:
        assert fp.read() == ""


@pytest.mark.parametrize("should_cast_path_to_string", [False, True])
def test_fetch_files_use_session(
    should_cast_path_to_string,
    tmp_path,
    request_mocker,  # noqa: ARG001
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
        data_dir=tmp_path,
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
        data_dir=tmp_path,
        verbose=0,
        files=[(*files, {"overwrite": True})],
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
        data_dir=tmp_path,
        verbose=0,
        files=[(*files, {"overwrite": False})],
    )

    assert request_mocker.url_count == 1
    assert os.path.exists(fil[0])
    with open(fil[0]) as fp:
        assert fp.read() == "some content"

    # Overwrite existing file.
    fil = _utils.fetch_files(
        data_dir=tmp_path,
        verbose=0,
        files=[(*files, {"overwrite": True})],
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
