from importlib.metadata import version

try:
    __version__ = version("pytest-reporter-html1")
except Exception:
    # package is not installed
    __version__ = "unknown"

__pypi_url__ = "https://pypi.python.org/pypi/pytest-reporter-html1"
