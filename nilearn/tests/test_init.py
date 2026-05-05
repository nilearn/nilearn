import pytest

import nilearn


def test_version_number():
    """Check proper version number is returned."""
    try:
        assert nilearn.__version__ == nilearn._version.__version__
    except AttributeError:
        assert nilearn.__version__ == "0+unknown"


@pytest.mark.slow
def test_dummy_slow_test():
    """Slow test dummy.

    Ensure that running tests marked as slow always run at least 1 test.
    """
    assert isinstance(nilearn.__version__, str)
