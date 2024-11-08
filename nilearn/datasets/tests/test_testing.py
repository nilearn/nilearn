import re
import tarfile
import zipfile
from pathlib import Path

import pytest
import requests

from nilearn import image
from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.datasets.tests import _testing


def test_sender_key_order(request_mocker):
    request_mocker.url_mapping["*message.txt"] = "message"
    resp = requests.get("https://example.org/message.txt")

    assert resp.text == "message"

    request_mocker.url_mapping["*.txt"] = "new message"
    resp = requests.get("https://example.org/message.txt")

    assert resp.text == "new message"

    request_mocker.url_mapping["*.csv"] = "other message"

    resp = requests.get("https://example.org/message.txt")

    assert resp.text == "new message"


def test_loading_from_archive_contents(tmp_path):
    expected_contents = sorted(
        [
            Path("README.txt"),
            Path("data"),
            Path("data", "img.nii.gz"),
            Path("data", "labels.csv"),
        ]
    )
    resp = requests.get("https://example.org/example_zip")
    file_path = tmp_path / "archive.zip"
    file_path.write_bytes(resp.content)
    zip_extract_dir = tmp_path / "extract_zip"
    zip_extract_dir.mkdir()

    with zipfile.ZipFile(str(file_path)) as zipf:
        assert sorted(map(Path, zipf.namelist())) == expected_contents
        zipf.extractall(str(zip_extract_dir))

    labels_file = zip_extract_dir / "data" / "labels.csv"

    assert labels_file.read_bytes() == b""

    for url_end in ["_default_format", "_tar_gz"]:
        resp = requests.get(f"https://example.org/example{url_end}")
        file_path = tmp_path / "archive.tar.gz"
        file_path.write_bytes(resp.content)
        tar_extract_dir = tmp_path / f"extract_tar{url_end}"
        tar_extract_dir.mkdir()

        with tarfile.open(str(file_path)) as tarf:
            assert sorted(map(Path, tarf.getnames())) == [
                Path(),
                *expected_contents,
            ]
            tarf.extractall(str(tar_extract_dir))

        labels_file = tar_extract_dir / "data" / "labels.csv"

        assert labels_file.read_bytes() == b""


def test_sender_regex(request_mocker):
    url = "https://example.org/info?key=value&name=nilearn"
    pattern = re.compile(
        r".*example.org/(?P<section>.*)\?.*name=(?P<name>[^&]+)"
    )
    request_mocker.url_mapping[pattern] = r"in \g<section>: hello \2"
    resp = requests.get(url)

    assert resp.text == "in info: hello nilearn"

    def f(match, request):
        return f"name: {match.group('name')}, url: {request.url}"

    request_mocker.url_mapping[pattern] = f
    resp = requests.get(url)

    assert resp.text == f"name: nilearn, url: {url}"

    def g(match, request):  # noqa: ARG001
        return 403

    request_mocker.url_mapping[pattern] = g
    resp = requests.get(url)
    with pytest.raises(requests.HTTPError, match="Error"):
        resp.raise_for_status()


def test_sender_status(request_mocker):
    request_mocker.url_mapping["*good"] = 200
    request_mocker.url_mapping["*forbidden"] = 403
    resp = requests.get("https://example.org/good")

    assert resp.status_code == 200
    assert resp.text == "OK"

    resp.raise_for_status()
    resp = requests.get("https://example.org/forbidden")

    assert resp.status_code == 403
    assert resp.text == "ERROR"
    with pytest.raises(requests.HTTPError, match="Error"):
        resp.raise_for_status()


class _MyError(Exception):
    pass


def test_sender_exception(request_mocker):
    request_mocker.url_mapping["*bad"] = _MyError("abc")
    with pytest.raises(_MyError, match="abc"):
        requests.get("ftp:example.org/bad")


def test_sender_img(request_mocker, tmp_path):
    request_mocker.url_mapping["*"] = generate_fake_fmri()[0]
    resp = requests.get("ftp:example.org/download")
    file_path = tmp_path / "img.nii.gz"
    file_path.write_bytes(resp.content)
    img = image.load_img(str(file_path))

    assert img.shape == (10, 11, 12, 17)


class _MyResponse(_testing.Response):
    def json(self):
        return '{"count": 1}'


def test_sender_response(request_mocker):
    request_mocker.url_mapping["*example.org/a"] = _MyResponse("", "")

    def f(match, request):  # noqa: ARG001
        resp = _testing.Response(b"hello", request.url)
        resp.headers["cookie"] = "abc"
        return resp

    request_mocker.url_mapping["*example.org/b"] = f
    resp = requests.get("https://example.org/a")

    assert resp.json() == '{"count": 1}'

    resp = requests.get("https://example.org/b")

    assert resp.headers["cookie"] == "abc"


def test_sender_path(request_mocker, tmp_path):
    file_path = tmp_path / "readme.txt"
    with file_path.open("w") as f:
        f.write("hello")
    request_mocker.url_mapping["*path"] = str(file_path)
    request_mocker.url_mapping["*content"] = file_path

    resp = requests.get("https://example.org/path")

    assert resp.text == str(file_path)

    resp = requests.get("https://example.org/content")

    assert resp.text == "hello"


def test_sender_bad_input(request_mocker):
    request_mocker.url_mapping["*"] = 2.5
    with pytest.raises(TypeError):
        requests.get("https://example.org")


def test_dict_to_archive(tmp_path):
    subdir = tmp_path / "tmp"
    subdir.mkdir()
    (subdir / "labels.csv").touch()
    (subdir / "img.nii.gz").touch()
    archive_spec = {
        "empty_data": subdir,
        "empty_data_path.txt": str(subdir),
        Path("data", "labels.csv"): "a,b,c",
        Path("data", "img.nii.gz"): generate_fake_fmri()[0],
        Path("a", "b", "c"): (100).to_bytes(
            length=1, byteorder="big", signed=False
        ),
    }
    targz = _testing.dict_to_archive(archive_spec)
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    archive_path = tmp_path / "archive"
    with archive_path.open("wb") as f:
        f.write(targz)
    with tarfile.open(str(archive_path)) as tarf:
        tarf.extractall(str(extract_dir))
    img = image.load_img(str(extract_dir / "data" / "img.nii.gz"))

    assert img.shape == (10, 11, 12, 17)
    with (extract_dir / "a" / "b" / "c").open("rb") as f:
        assert int.from_bytes(f.read(), byteorder="big", signed=False) == 100
    with (extract_dir / "empty_data" / "labels.csv").open() as f:
        assert f.read() == ""

    zip_archive = _testing.dict_to_archive(
        {"readme.txt": "hello", "archive": targz}, "zip"
    )
    with archive_path.open("wb") as f:
        f.write(zip_archive)

    with (
        zipfile.ZipFile(str(archive_path)) as zipf,
        zipf.open("archive", "r") as f,
    ):
        assert f.read() == targz

    from_list = _testing.list_to_archive(archive_spec.keys())
    with archive_path.open("wb") as f:
        f.write(from_list)

    with tarfile.open(str(archive_path)) as tarf:
        assert sorted(map(Path, tarf.getnames())) == sorted(
            [
                *list(map(Path, archive_spec.keys())),
                Path(),
                Path("a"),
                Path("a", "b"),
                Path("data"),
            ]
        )
