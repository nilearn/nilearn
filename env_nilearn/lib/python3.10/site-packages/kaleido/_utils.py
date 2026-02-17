import asyncio
import traceback
import warnings
from functools import partial
from importlib.metadata import PackageNotFoundError, version

import logistro
from packaging.version import Version

_logger = logistro.getLogger(__name__)


async def to_thread(func, *args, **kwargs):
    _loop = asyncio.get_running_loop()
    fn = partial(func, *args, **kwargs)
    await _loop.run_in_executor(None, fn)


def warn_incompatible_plotly():
    """
    Check if installed Plotly version (if any) is compatible with this Kaleido version.

    If not, display a warning.
    """
    try:
        min_compatible_plotly_version = Version("6.1.1")
        installed_plotly_version = Version(version("plotly"))
        installed_kaleido_version = Version(version("kaleido"))
        if installed_plotly_version < min_compatible_plotly_version:
            warnings.warn(
                "\n\n"
                f"Warning: You have Plotly version {installed_plotly_version}, "
                "which is not compatible with this version of "
                f"Kaleido ({installed_kaleido_version}).\n\n"
                "This means that static image generation (e.g. `fig.write_image()`) "
                "will not work.\n\n"
                f"Please upgrade Plotly to version {min_compatible_plotly_version} "
                "or greater, or downgrade Kaleido to version 0.2.1."
                "\n",
                UserWarning,
                stacklevel=3,
            )
    except PackageNotFoundError:
        # If Plotly is not installed, there's nothing to worry about
        pass
    # ruff: noqa: BLE001
    except Exception as e:
        # If another error occurs, log it but do not raise
        # Since this compatibility check is just a convenience,
        # we don't want to block the whole library if there's an issue
        _logger.info("Error while checking Plotly version.", exc_info=e)


class ErrorEntry:
    """A simple object to record errors and context."""

    def __init__(self, name, error, javascript_log):
        """
        Construct an error entry.

        Args:
            name: the name of the image with the error
            error: the error object (from class BaseException)
            javascript_log: an array of entries from the javascript console

        """
        self.name = name
        self.error = error
        self.javascript_log = javascript_log

    def __str__(self):
        """Display the error object in a concise way."""
        ret = f"{self.name}:\n"
        e = self.error
        ret += " ".join(traceback.format_exception(type(e), e, e.__traceback__))
        ret += " javascript Log:\n"
        ret += "\n ".join(self.javascript_log)
        return ret
