"""
Channels are classes that choreo and the browser use to communicate.

This is a low-level part of the API.

"""

from ._errors import BlockWarning, ChannelClosedError, JSONError
from ._wire import register_custom_encoder
from .pipe import Pipe

__all__ = [
    "BlockWarning",
    "ChannelClosedError",
    "JSONError",
    "Pipe",
    "register_custom_encoder",
]
