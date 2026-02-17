"""cli provides some tools that are used on the commandline (and to download chrome)."""

from ._cli_utils import (
    get_chrome,
    get_chrome_sync,
)

__all__ = [
    "get_chrome",
    "get_chrome_sync",
]
