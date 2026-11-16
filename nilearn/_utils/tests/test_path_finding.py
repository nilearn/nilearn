from pathlib import Path

import pytest

from nilearn._utils.path_finding import resolve_globbing


def test_resolve_globbing(tmp_path):
    assert resolve_globbing(tmp_path) == [tmp_path]
    assert resolve_globbing([]) == []


def test_resolve_globbing_path_expanded():
    tmp_home = Path("~/some_user")
    tmp_home.expanduser().mkdir(parents=True, exist_ok=True)
    assert resolve_globbing(tmp_home) == [tmp_home.expanduser()]


def test_resolve_globbing_nested(tmp_path):
    (tmp_path / "spam.txt").touch()
    (tmp_path / "foo.txt").touch()
    (tmp_path / "spam").mkdir(parents=True, exist_ok=True)
    (tmp_path / "spam" / "foo.txt").touch()
    (tmp_path / "foo").mkdir(parents=True, exist_ok=True)
    (tmp_path / "foo" / "foo.txt").touch()

    results = resolve_globbing(tmp_path / "foo.txt")
    assert len(results) == 1
    assert all(isinstance(x, Path) for x in results)
    assert results == [tmp_path / "foo.txt"]

    results = resolve_globbing(tmp_path / "**/foo.txt")
    assert len(results) == 2
    assert all(isinstance(x, Path) for x in results)
    assert results == [
        tmp_path / "foo" / "foo.txt",
        tmp_path / "spam" / "foo.txt",
    ]


def test_resolve_globbing_error(tmp_path):
    with pytest.raises(ValueError, match="No files matching path"):
        assert resolve_globbing(tmp_path / "does_not_exist.txt")
