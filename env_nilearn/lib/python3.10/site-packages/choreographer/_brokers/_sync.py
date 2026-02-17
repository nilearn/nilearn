from __future__ import annotations

import json
from threading import Thread
from typing import TYPE_CHECKING

import logistro

from choreographer import protocol
from choreographer.channels import ChannelClosedError

if TYPE_CHECKING:
    from typing import Any

    from choreographer.browser_sync import BrowserSync
    from choreographer.channels._interface_type import ChannelInterface

_logger = logistro.getLogger(__name__)


class BrokerSync:
    """BrokerSync is a middleware implementation for synchronous browsers."""

    _browser: BrowserSync
    """Browser is a reference to the Browser object this broker is brokering for."""
    _channel: ChannelInterface
    """
    Channel will be the ChannelInterface implementation (pipe or websocket)
    that the broker communicates on.
    """

    def __init__(self, browser: BrowserSync, channel: ChannelInterface) -> None:
        """
        Construct a broker for a synchronous arragenment w/ both ends.

        Args:
            browser: The sync browser implementation.
            channel: The channel the browser uses to talk on.

        """
        self._browser = browser
        self._channel = channel

    def run_output_thread(self, **kwargs: Any) -> None:
        """
        Run a thread which dumps all browser messages. kwargs is passed to print.

        Raises:
            ChannelClosedError: When the channel is closed, this error is raised.

        """

        def run_print() -> None:
            try:
                while True:
                    responses = self._channel.read_jsons()
                    for response in responses:
                        print(json.dumps(response, indent=4), **kwargs)  # noqa: T201 print in the point
            except ChannelClosedError:
                print("ChannelClosedError caught.", **kwargs)  # noqa: T201 print is the point

        _logger.info("Starting thread to dump output to stdout.")
        Thread(target=run_print).start()

    def write_json(self, obj: protocol.BrowserCommand) -> protocol.MessageKey | None:
        """
        Send an object down the channel.

        Args:
            obj: An object to be serialized to json and written to the channel.

        """
        protocol.verify_params(obj)
        key = protocol.calculate_message_key(obj)
        self._channel.write_json(obj)
        return key

    def clean(self) -> None:
        pass
