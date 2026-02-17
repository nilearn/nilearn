"""Provides a channel based on operating system file pipes."""

from __future__ import annotations

import os
import platform
import sys
import warnings
from threading import Lock
from typing import TYPE_CHECKING

import logistro

from . import _wire as wire
from ._errors import BlockWarning, ChannelClosedError, JSONError

if TYPE_CHECKING:
    from typing import Any, Mapping, Sequence

    from choreographer.protocol import BrowserResponse

_with_block = bool(sys.version_info[:3] >= (3, 12) or platform.system() != "Windows")

_logger = logistro.getLogger(__name__)

# should be closing my ends from the start?


# if we're a pipe we expect these public attributes
class Pipe:
    """Defines an operating system pipe."""

    from_external_to_choreo: int
    """Consumers need this, it is the channel the browser uses to talk to choreo."""
    from_choreo_to_external: int
    """Consumers needs this, it is the channel choreo writes to the browser on."""
    shutdown_lock: Lock
    """Once this is locked, the pipe is closed and can't be reopened."""

    def __init__(self) -> None:
        """Construct a pipe using os functions."""
        # This is where pipe listens (from browser)
        # So pass the write to browser
        self._read_from_browser, self._write_from_browser = list(os.pipe())

        # This is where pipe writes (to browser)
        # So pass the read to browser
        self._read_to_browser, self._write_to_browser = list(os.pipe())

        # Popen will write stdout of wrapper to this (dupping 4)
        # Browser will write directly to this if not using wrapper
        self.from_external_to_choreo = self._write_from_browser
        # Popen will read this into stdin of wrapper (dupping 3)
        # Browser will read directly from this if not using wrapper
        # which dupes stdin to expected fd (4?)
        self.from_choreo_to_external = self._read_to_browser
        # These won't be used on windows directly, they'll be t-formed to
        # windows-style handles. But let another layer handle that.

        # this is just a convenience to prevent multiple shutdowns
        self.shutdown_lock = Lock()  # should be private
        self._open_lock = Lock()  # should be private

    def is_ready(self) -> bool:
        """Return true if pipe open."""
        return not self.shutdown_lock.locked() and self._open_lock.locked()

    def open(self) -> None:
        """
        Open the channel.

        In a sense, __init__ creates the pipe. The OS opens it.
        Here we're just marking it open for use, that said.

        We only use locks here for indications, we never actually lock,
        because the broker is in charge of all async/parallel stuff.
        """
        if not self._open_lock.acquire(blocking=False):
            raise RuntimeError("Cannot open same pipe twice.")

    def write_json(self, obj: Mapping[str, Any]) -> None:
        """
        Send one json down the pipe.

        Args:
            obj: any python object that serializes to json.

        """
        if not self.is_ready():
            raise ChannelClosedError(
                "The communication channel was either never "
                "opened or closed. Was .open() or .close() called?",
            )
        encoded_message = wire.serialize(obj) + b"\0"
        _logger.debug(
            f"Writing message {encoded_message[:15]!r}...{encoded_message[-15:]!r}, "
            f"size: {len(encoded_message)}.",
        )
        _logger.debug2(f"Full Message: {encoded_message!r}")
        try:
            ret = os.write(self._write_to_browser, encoded_message)
            _logger.debug(
                f"***Wrote {ret}/{len(encoded_message)}***",
            )
            if ret != len(encoded_message):
                _logger.critical(
                    f"***Did not write entire message. {ret}/{len(encoded_message)}***",
                )
        except OSError as e:
            self.close()
            raise ChannelClosedError from e

    def read_jsons(  # noqa: PLR0912, PLR0915, C901 branches, complexity
        self,
        *,
        blocking: bool = True,
    ) -> Sequence[BrowserResponse]:
        """
        Read from the pipe and return one or more jsons in a list.

        Args:
            blocking: The read option can be set to block or not.

        Returns:
            A list of jsons.

        """
        jsons: list[BrowserResponse] = []
        if not self.is_ready():
            raise ChannelClosedError(
                "The communication channel was either never "
                "opened or closed. Was .open() or .close() called?",
            )
        if not _with_block and not blocking:
            warnings.warn(  # noqa: B028
                "Windows python version < 3.12 does not support non-blocking",
                BlockWarning,
            )
        try:
            if _with_block:
                os.set_blocking(self._read_from_browser, blocking)
        except OSError as e:
            self.close()
            raise ChannelClosedError from e
        raw_buffer = None  # if we fail in read, we already defined
        loop_count = 1
        try:
            raw_buffer = os.read(
                self._read_from_browser,
                10000,
            )  # 10MB buffer, nbd, doesn't matter w/ this
            _logger.debug(
                f"First read in loop: {raw_buffer[:15]!r}...{raw_buffer[-15:]!r}. "
                f"size: {len(raw_buffer)}.",
            )
            _logger.debug2(f"Whole buffer: {raw_buffer!r}")
            if not raw_buffer or raw_buffer == b"{bye}\n":
                if raw_buffer:
                    _logger.debug(f"Received {raw_buffer!r}. is bye?")
                # we seem to need {bye} even if chrome closes NOTE
                self.close()
                raise ChannelClosedError
            while raw_buffer[-1] != 0:
                _logger.debug("Partial message from browser received.")
                loop_count += 1
                if _with_block:
                    os.set_blocking(self._read_from_browser, True)
                raw_buffer += os.read(self._read_from_browser, 10000)
        except BlockingIOError:
            _logger.debug("BlockingIOError")
            return jsons
        except OSError as e:
            _logger.debug("OSError")
            self.close()
            if not raw_buffer or raw_buffer == b"{bye}\n":
                raise ChannelClosedError from e
            # this could be hard to test as it is a real OS corner case
        finally:
            _logger.debug(
                f"Total loops: {loop_count}, "
                f"Final size: {len(raw_buffer) if raw_buffer else 0}.",
            )
            _logger.debug2(f"Whole buffer: {raw_buffer!r}")
        if raw_buffer is None:
            return jsons
        decoded_buffer = raw_buffer.decode("utf-8")
        raw_messages = decoded_buffer.split("\0")
        _logger.debug(f"Received {len(raw_messages)} raw_messages.")
        for raw_message in raw_messages:
            if raw_message:
                try:
                    jsons.append(wire.deserialize(raw_message))
                except JSONError:
                    _logger.exception("JSONError decoding message. Ignoring")
                except:
                    _logger.exception("Error in trying to decode JSON off our read.")
                    raise
        return jsons

    def _unblock_fd(self, fd: int) -> None:
        try:
            if _with_block:
                os.set_blocking(fd, False)
        except Exception:  # noqa: BLE001, S110 OS errors are not consistent, catch blind + pass
            pass

    def _close_fd(self, fd: int) -> None:
        try:
            os.close(fd)
        except Exception:  # noqa: BLE001, S110 OS errors are not consistent, catch blind + pass
            pass

    def _fake_bye(self) -> None:
        self._unblock_fd(self._write_from_browser)
        try:
            os.write(self._write_from_browser, b"{bye}\n")
        except Exception:  # noqa: BLE001, S110 OS errors are not consistent, catch blind + pass
            pass

    def close(self) -> None:
        """Close the pipe."""
        if self.shutdown_lock.acquire(blocking=False):
            if platform.system() == "Windows":
                self._fake_bye()
            self._unblock_fd(self._write_from_browser)
            self._unblock_fd(self._read_from_browser)
            self._unblock_fd(self._write_to_browser)
            self._unblock_fd(self._read_to_browser)
            self._close_fd(self._write_to_browser)  # no more writes
            self._close_fd(self._write_from_browser)  # we're done with writes
            self._close_fd(self._read_from_browser)  # no more attempts at read
            self._close_fd(self._read_to_browser)
