"""
choreographer is a browser controller for python.

choreographer is natively async, so while there are two main entrypoints:
classes `Browser` and `BrowserSync`, the sync version is very limited, functioning
as a building block for more featureful implementations.

See the main README for a quickstart.
"""

import os

if os.getenv("CHOREO_ENABLE_DEBUG"):
    import sys

    import logistro

    logistro.betterConfig(level=1)
    print("DEBUG MODE!", file=sys.stderr)  # noqa: T201

from .browser_async import (
    Browser,
    Tab,
)
from .browser_sync import (
    BrowserSync,
    TabSync,
)

__all__ = [
    "Browser",
    "BrowserSync",
    "Tab",
    "TabSync",
]
