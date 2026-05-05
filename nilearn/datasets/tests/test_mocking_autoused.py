import os
from pathlib import Path
from urllib import request

import requests


def test_request_mocking_autoused_requests():
    """Check the request mocker is autoused."""
    assert requests.sessions.Session.send.__class__.__name__ == "Sender"
    assert requests.sessions.Session.send.is_mock

    resp = requests.get("https://example.com")

    assert resp.is_mock

    resp = requests.post("https://example.com", data={"key": "value"})

    assert resp.is_mock

    session = requests.Session()
    req = requests.Request("GET", "https://example.com")
    prepped = session.prepare_request(req)
    resp = session.send(prepped)

    assert resp.is_mock


def test_request_mocking_autoused_urllib():
    """Check the request mocker is autoused and works for a given URL."""
    resp = request.urlopen("https://example.com")

    assert resp.__class__.__name__ == "MagicMock"

    req = request.Request("https://example.com")
    opener = request.build_opener()
    resp = opener.open(req)

    assert resp.__class__.__name__ == "MagicMock"


def test_temp_nilearn_home_autoused():
    """Check that '~', NILEARN_DATA, NILEARN_SHARED_DATA \
       are properly expanded.
    """
    home_dir = Path("~").expanduser()

    assert home_dir.name.startswith("temp_nilearn_home")

    home_dir = Path.home()

    assert home_dir.name.startswith("temp_nilearn_home")

    home_dir = Path("~").expanduser()

    assert home_dir.name.startswith("temp_nilearn_home")

    nilearn_data = Path(os.environ.get("NILEARN_DATA"))

    assert nilearn_data.parent.name.startswith("temp_nilearn_home")

    nilearn_shared_data = Path(os.environ.get("NILEARN_SHARED_DATA"))

    assert nilearn_shared_data.parent.name.startswith("temp_nilearn_home")


def check_doctest_fixture():
    """Check doctest fixtures.

    >>> import requests
    >>> assert requests.get("https://example.com").is_mock
    """
