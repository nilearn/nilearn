"""Provides the basic protocol class (the abstract interface) for a channel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from typing import Any, Mapping, Sequence

    from choreographer.protocol import BrowserResponse


class ChannelInterface(Protocol):
    """Defines the basic interface of a channel."""

    # Not sure I like the obj type
    def write_json(self, obj: Mapping[str, Any]) -> None:
        ...
        # """
        # Accept an object and send it doesnt the channel serialized.
        #
        # Args:
        #   obj: the object to send to the browser.
        #
        # """

    def read_jsons(self, *, blocking: bool = True) -> Sequence[BrowserResponse]:
        ...
        # """
        # Read all available jsons in the channel and returns a list of complete ones.
        #
        # Args:
        #   blocking: should this method block on read or return immediately.
        # """

    def close(self) -> None:
        ...
        # """Close the channel."""

    def open(self) -> None:
        ...
        # """Open the channel."""

    def is_ready(self) -> bool:
        ...
        # """Return true if comm channel is active."""


# Can't docstring protocols! EW!
