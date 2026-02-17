"""A modern skeleton for Sphinx themes."""

__version__ = "1.0.0.beta2"

from pathlib import Path
from typing import Any, Dict

from sphinx.application import Sphinx

_THEME_PATH = (Path(__file__).parent / "theme" / "basic-ng").resolve()


def setup(app: Sphinx) -> Dict[str, Any]:
    """Entry point for sphinx theming."""
    app.require_sphinx("4.0")

    app.add_html_theme("basic-ng", str(_THEME_PATH))

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
        "version": __version__,
    }
