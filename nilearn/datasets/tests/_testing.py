"""Utilities for testing the dataset fetchers.

Unit tests should not depend on an internet connection nor on external
resources such as the servers from which we download datasets. Otherwise, tests
can fail spuriously when a web service is unavailable, and tests are slow
because downloading data takes a lot of time.

Therefore in the tests, we fake the downloads: the function from the requests
library that would normally download a file is replaced ("patched") by a
"mock", a function that mimics its interface but doesn't download anything and
returns fake data instead.

As we only patch functions from urllib and requests, nilearn code is unaware of
the mocking and can be tested as usual, as long as we provide fake responses
that look similar to those we would obtain from dataset providers if we
actually sent requests over the network.

This module provides the utilities for setting up this mocking: patching the
relevant requests and urllib functions, and creating the fake responses. The
function from the requests library that nilearn uses to send requests is
patched (replaced) by a `Sender` object defined in this module. The
corresponding docstring details how individual tests can configure what fake
responses it should return for specific URLs.

To make sure tests don't rely on previously existing data and don't write
outside of temporary directories, this module also adds fixtures to patch the
home directory and other default nilearn data directories.

"""

import fnmatch
import json
import os
import pickle
import re
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from requests.exceptions import HTTPError

from nilearn._utils.testing import serialize_niimg


@pytest.fixture(autouse=True)
def temp_nilearn_data_dir(tmp_path_factory, monkeypatch):
    """Monkeypatch user home directory and NILEARN_DATA env variable.

    This ensures that tests that use nilearn.datasets will not load datasets
    already present on the current machine, or write in the user's home or
    nilearn data directory.

    This fixture uses 'autouse' and is imported in conftest.py to make sure it
    is used by every test, even those that do not explicitly ask for it.

    """
    home_dir = tmp_path_factory.mktemp("temp_nilearn_home")
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("USERPROFILE", str(home_dir))
    data_dir = home_dir / "nilearn_data"
    data_dir.mkdir()
    monkeypatch.setenv("NILEARN_DATA", str(data_dir))
    shared_data_dir = home_dir / "nilearn_shared_data"
    monkeypatch.setenv("NILEARN_SHARED_DATA", str(shared_data_dir))


@pytest.fixture(autouse=True)
def request_mocker(monkeypatch):
    """Monkeypatch requests and urllib functions for sending requests.

    This ensures that test functions do not retrieve data from the network, but
    can still run using mock data.

    request.send is patched with a Sender object, whose responses can be
    configured -- see the docstring for Sender.

    urllib's open is simply patched with a MagicMock. As nilearn dataset
    fetchers use requests, most tests won't use this; it is patched to make
    sure network mocking is not worked around by using urllib directly instead
    of requests, and for testing the FTP adapter.

    This fixture uses 'autouse' and is imported in conftest.py to make sure it
    is used by every test, even those that do not explicitly ask for it.

    """
    sender = Sender()
    monkeypatch.setattr("requests.sessions.Session.send", sender)
    monkeypatch.setattr("urllib.request.OpenerDirector.open", MagicMock())
    return sender


class Response:
    """Response objects returned by Sender.

    This class mocks requests.Response. It does not implement the full
    interface; only the parts used by nilearn functions.

    """

    is_mock = True

    def __init__(self, content, url, status_code=200):
        self.content = content
        self.url = url
        self.status_code = status_code
        self.headers = {"Content-Length": len(self.content)}
        self.iter_start = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size=8):
        for i in range(self.iter_start, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]

    @property
    def text(self):
        return self.content.decode("utf-8")

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise HTTPError(f"{self.status_code} Error for url: {self.url}")


class Request:
    """A mock request class."""

    is_mock = True

    def __init__(self, url):
        self.url = url


class Sender:
    r"""Mock class used to patch requests.sessions.Session.send.

    In nilearn's tests this replaces the function used by requests to send
    requests over the network.

    Test functions can configure this object to specify what response is
    expected when sending requests to specific URLs. This is done by adding
    items to the ordered dictionary self.url_mapping

    When a Sender receives a request, it tries to match it against the keys in
    url_mapping, and then against urls found in
    nilearn/datasets/tests/data/archive_contents/ (details below). If a key
    matches, the corresponding value is used to compute the response. If no key
    matches, an response with empty content is returned.

    If several keys match, the one that was inserted most recently in
    url_mapping is used.

    Specifying keys
    ---------------
    Keys of url_mapping can be:

    - a `str`: it is used as a glob pattern, matched against the url with
      fnmatch (If special characters need to be matched literally they can be
      escaped with []). For example:
         '*' matches everything
         '*example.com/*' matches 'https://www.example.com/data'
                          but not 'https://example.org'

    - a `re.Pattern` (ie a compiled regex): it is matched against the url, and
      groups can be used to capture parts needed to construct the response. For
      example:
        re.compile(r'.*example.org/subject_(\d+)\.tar\.gz')
          matches 'https://example.org/subject_12.tar.gz' and captures '12'
          but does not match 'https://example.org/subject_12.csv'

    If none of the keys in url_mapping matches, the Sender turns to the
    contents of nilearn/datasets/tests/data/archive_contents. Files in this
    directory or any subdirectory are used to build responses that contain zip
    or tar archives containing a certain list of files. (.py and .pyc files are
    ignored)
    The first line of a file in archive_contents is a glob pattern stating to
    which urls it applies. If it matches, the subsequent lines are paths that
    will exist inside the archive. The files created in the archive are
    empty. For example, if a file looks like:
      https://example.org/subj_*.tar.gz
      README.txt
      data/img.nii.gz
      data/labels.csv
    the response will be a tar gzipped archive with this structure:
        .
        ├── data
        │   ├── img.nii.gz
        │   └── labels.csv
        └── README.txt

    Moreover, if the first line starts with 'format:' it is used to determine
    the archive format. For example: 'format: zip', 'format: gztar' (see
    `shutil` documentation for available formats). In this case the second line
    contains the url pattern and the rest of the file lists the contents.
    The paths for the archive contents must use '/' as path separator, it gets
    converted to the OS's separator when the file is read.
    A helper script is provided in
    nilearn/datasets/tests/data/list_archive_contents.sh to generate such files
    from a url.

    Finally, if no key and no file matches the request url, a response with an
    empty content is returned.

    Specifying values
    -----------------

    Once a key matches, the corresponding value is used to build a response.
    The value can be:
    - a callable: it is called as value(match, request), where request is the
      input `Request` object, and match is the url if the key was a string and
      the `re.Match` resulting from matching the key if it was a `re.Pattern`.
      The result of this call is then processed as described below.
    - an instance of the Response class: it used without modification.
    - a `bytes`: result is a Response with status 200 and these bytes as
      content.
    - a str: if the key was a `re.Pattern`, the value can contain
      backreferences that are replaced with groups matched in the url, e.g.
      \1, \g<groupname>. The resulting string is then encoded with UTF-8 to
      build the response content. For example:
        re.compile(r'.*example\.org/(.*)'): r'hello, \1'
        results in b'hello, nilearn' if the url is https://example.org/nilearn
    - an int: results in an response with this status code. The content is
      b"ERROR" if the status code is in [400, 600[ and b"OK" otherwise
    - an `Exception`: it is raised
    - a `pathlib.Path`: the contents of the response are the contents of that
      file. (can also be anything that has a `read_bytes` attribute,
      e.g a `pathlib2.Path`)
    - an object with a `to_filename` method, eg a Nifti1Image: it is serialized
      to .nii.gz to produce the response content.

    To help construct values that mock downloaded archives, this module
    provides `dict_to_archive` and `list_to_archive` helper functions; more
    details in their docstrings.

    Inspecting history
    ------------------
    Senders record all sent requests in `sent_requests`, the visited urls in
    `visited_urls`, and the number of sent requests in `url_count`

    """

    is_mock = True

    def __init__(self):
        self.url_mapping = OrderedDict()
        self.sent_requests = []
        self._archive_contents_index = _index_archive_contents()

    @property
    def visited_urls(self):
        return [request.url for request in self.sent_requests]

    @property
    def url_count(self):
        return len(self.visited_urls)

    def __call__(
        self,
        request,
        *args,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ):
        if isinstance(request, str):
            request = Request(request)
        self.sent_requests.append(request)
        for key, value in list(self.url_mapping.items())[::-1]:
            match = self.match(key, request.url)
            if match is not None:
                return self.get_response(value, match, request)
        for key, file_path in self._archive_contents_index.items():
            match = self.match(key, request.url)
            if match is not None:
                return Response(
                    _archive_from_contents_file(file_path), request.url
                )
        return self.default_response(request)

    def default_response(self, request):
        return Response(b"", request.url)

    def match(self, key, url):
        if isinstance(key, type(re.compile(r".*"))):
            return key.match(url)
        elif isinstance(key, str) and fnmatch.fnmatch(url, key):
            return url
        else:
            return None

    def get_response(self, response, match, request):
        if callable(response):
            response = response(match, request)

        if isinstance(response, Response):
            return response
        elif isinstance(response, Exception):
            raise response
        elif isinstance(response, int):
            if 400 <= response < 600:
                return Response(b"ERROR", request.url, status_code=response)
            else:
                return Response(b"OK", request.url, status_code=response)
        elif hasattr(response, "to_filename"):
            return Response(serialize_niimg(response), request.url)
        elif hasattr(response, "read_bytes"):
            return Response(response.read_bytes(), request.url)
        elif isinstance(response, str):
            if isinstance(match, type(re.match(r".*", ""))):
                response = match.expand(response)
            response = response.encode("utf-8")
            return Response(response, request.url)
        elif isinstance(response, bytes):
            return Response(response, request.url)
        else:
            raise TypeError(
                f"Don't know how to make a Response from: {response}"
            )


def _get_format_and_pattern(file_path):
    file_path = Path(file_path)
    with file_path.open() as f:
        first_line = f.readline().strip()
        match = re.match(r"format *: *(.+)", first_line)
        if match is None:
            return "gztar", first_line, 1
        return match[1], f.readline().strip(), 2


def _index_archive_contents():
    archive_contents_dir = (
        Path(__file__).parent.parent / "tests" / "data" / "archive_contents"
    )
    index = {}
    for file_path in sorted(archive_contents_dir.glob("**/*")):
        if file_path.is_file() and file_path.suffix not in [".py", ".pyc"]:
            fmt, url_pattern, n = _get_format_and_pattern(file_path)
            index[url_pattern] = str(file_path.resolve())
    return index


def _archive_from_contents_file(file_path):
    file_path = Path(file_path)
    fmt, pattern, n_skip = _get_format_and_pattern(file_path)
    with file_path.open() as f:
        contents = [p.strip().replace("/", os.sep) for p in f]
    return list_to_archive(list(filter(bool, contents))[n_skip:], fmt)


def _add_to_archive(path, content):
    path.parent.mkdir(exist_ok=True, parents=True)
    if hasattr(content, "to_filename"):
        content.to_filename(str(path))
    elif hasattr(content, "is_dir") and hasattr(content, "is_file"):
        if content.is_file():
            shutil.copy(str(content), str(path))
        elif content.is_dir():
            shutil.copytree(str(content), str(path))
        else:
            raise FileNotFoundError(
                f"Not found or not a regular file or a directory {content}"
            )
    elif isinstance(content, str):
        with path.open("w") as f:
            f.write(content)
    elif isinstance(content, bytes):
        with path.open("wb") as f:
            f.write(content)
    else:
        with path.open("wb") as f:
            pickle.dump(content, f)


def dict_to_archive(data, archive_format="gztar"):
    """Transform a {path: content} dict to an archive.

    Parameters
    ----------
    data : dict
        Keys are strings or `pathlib.Path` objects and specify paths inside the
        archive. (If strings, must use the system path separator.)
        Values determine the contents of these files and can be:
          - an object with a `to_filename` method (e.g. a Nifti1Image): it is
            serialized to .nii.gz
          - a `pathlib.Path`: the contents are copied inside the archive (can
            point to a file or a directory). (can also be anything that has
            `is_file` and `is_directory` attributes, e.g. a `pathlib2.Path`)
          - a `str` or `bytes`: the contents of the file
          - anything else is pickled.

    archive_format : str, optional (default="gztar")
        The archive format. See `shutil` documentation for available formats.

    Returns
    -------
    bytes : the contents of the resulting archive file, to be used for example
        as the contents of a mock response object (see Sender).

    Examples
    --------
    if `data` is `{"README.txt": "hello", Path("Data") / "labels.csv": "a,b"}`,
    the resulting archive has this structure:
        .
        ├── Data
        │   └── labels.csv
        └── README.txt

    where labels.csv and README.txt contain the corresponding values in `data`

    """
    with tempfile.TemporaryDirectory() as root_tmp_dir:
        root_tmp_dir = Path(root_tmp_dir)
        tmp_dir = root_tmp_dir / "tmp"
        tmp_dir.mkdir()
        for path, content in data.items():
            _add_to_archive(tmp_dir / path, content)
        archive_path = shutil.make_archive(
            str(root_tmp_dir / "archive"), archive_format, str(tmp_dir)
        )
        with Path(archive_path).open("rb") as f:
            return f.read()


def list_to_archive(sequence, archive_format="gztar", content=""):
    """Transform a list of paths to an archive.

    This invokes dict_to_archive with the `sequence` items as keys and
    `content` (by default '') as values.

    For example, if `sequence` is
    `["README.txt", Path("Data") / "labels.csv"]`,
    the resulting archive has this structure:
        .
        ├── Data
        │   └── labels.csv
        └── README.txt

    and "labels.csv" and "README.txt" contain the value of `content`.

    """
    return dict_to_archive(
        {item: content for item in sequence}, archive_format=archive_format
    )
