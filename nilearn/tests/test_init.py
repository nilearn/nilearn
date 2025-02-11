import nilearn


def test_version_number():
    """Check proper version number is returned."""
    try:
        assert nilearn.__version__ == nilearn._version.__version__
    except AttributeError:
        assert nilearn.__version__ == "0+unknown"
