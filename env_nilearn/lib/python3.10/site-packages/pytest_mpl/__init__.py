try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("pytest_mpl")
except PackageNotFoundError:
    __version__ = "unknown"
