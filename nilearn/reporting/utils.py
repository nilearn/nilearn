"""Utility functions for the reporting module."""

import base64
import io
import urllib.parse


def figure_to_svg_bytes(fig):
    """Save figure as svg and return it as bytes."""
    with io.BytesIO() as io_buffer:
        fig.savefig(
            io_buffer, format="svg", facecolor="white", edgecolor="white"
        )
        return io_buffer.getvalue()


def figure_to_svg_base64(fig):
    """Save figure as svg and return it as 64 bytes."""
    return base64.b64encode(figure_to_svg_bytes(fig)).decode()


def figure_to_svg_quoted(fig):
    """Save figure as svg and return it as quoted string."""
    return urllib.parse.quote(figure_to_svg_bytes(fig).decode("utf-8"))
