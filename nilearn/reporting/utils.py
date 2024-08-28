"""Utility functions for the reporting module."""

import base64
import io
import urllib.parse
from pathlib import Path

TEMPLATE_ROOT_PATH = Path(__file__).parent / "data"

CSS_PATH = TEMPLATE_ROOT_PATH / "css"

HTML_TEMPLATE_PATH = TEMPLATE_ROOT_PATH / "html"

HTML_PARTIALS_PATH = HTML_TEMPLATE_PATH / "partials"


def _figure_to_bytes(fig, format):
    """Save figure as as certain format and return it as bytes."""
    with io.BytesIO() as io_buffer:
        fig.savefig(
            io_buffer, format=format, facecolor="white", edgecolor="white"
        )
        return io_buffer.getvalue()


def figure_to_svg_bytes(fig):
    """Save figure as svg and return it as bytes."""
    return _figure_to_bytes(fig, format="svg")


def figure_to_png_bytes(fig):
    """Save figure as png and return it as bytes."""
    return _figure_to_bytes(fig, format="png")


def figure_to_svg_base64(fig):
    """Save figure as svg and return it as 64 bytes."""
    return base64.b64encode(figure_to_svg_bytes(fig)).decode()


def figure_to_png_base64(fig):
    """Save figure as png and return it as 64 bytes."""
    return base64.b64encode(figure_to_png_bytes(fig)).decode()


def figure_to_svg_quoted(fig):
    """Save figure as svg and return it as quoted string."""
    return urllib.parse.quote(figure_to_svg_bytes(fig).decode("utf-8"))
