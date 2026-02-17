"""A list of errors available from the kaleido package."""

from choreographer.errors import (
    BrowserClosedError,
    BrowserFailedError,
    ChromeNotFoundError,
)

from ._kaleido_tab import JavascriptError, KaleidoError

__all__ = [
    "BrowserClosedError",
    "BrowserFailedError",
    "ChromeNotFoundError",
    "JavascriptError",
    "KaleidoError",
]
