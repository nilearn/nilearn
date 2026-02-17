import pytest

try:
    from contextlib import chdir as _chdir
except ImportError:  # PY310
    import os
    from contextlib import contextmanager

    @contextmanager  # type: ignore[no-redef]
    def _chdir(path):
        cwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(cwd)


@pytest.fixture(autouse=True)
def chdir_tmpdir(request, tmp_path):
    if request.node.__class__.__name__ == "DoctestItem":
        with _chdir(tmp_path):
            yield
    else:
        yield
