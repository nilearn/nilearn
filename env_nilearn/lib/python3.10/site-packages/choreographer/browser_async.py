"""Provides the async api: `Browser`, `Tab`."""

from __future__ import annotations

import asyncio
import os
import subprocess
import warnings
from asyncio import Lock
from typing import TYPE_CHECKING

import logistro

from choreographer import protocol

from ._brokers import Broker
from .browsers import BrowserClosedError, BrowserDepsError, BrowserFailedError, Chromium
from .channels import ChannelClosedError, Pipe
from .protocol.devtools_async import Session, Target
from .utils import TmpDirWarning, _manual_thread_pool
from .utils._kill import kill

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType
    from typing import Any, Generator, MutableMapping

    from typing_extensions import Self  # 3.9 needs this, could be from typing in 3.10

    from .browsers._interface_type import BrowserImplInterface
    from .channels._interface_type import ChannelInterface

_logger = logistro.getLogger(__name__)

# Since I added locks to pipes, do we need locks here?


class Tab(Target):
    """A wrapper for `Target`, so user can use `Tab`, not `Target`."""

    async def close(self) -> None:
        """Close the tab."""
        await self._broker._browser.close_tab(target_id=self.target_id)  # noqa: SLF001


class Browser(Target):
    """`Browser` is the async implementation of `Browser`."""

    subprocess: subprocess.Popen[bytes] | subprocess.Popen[str]
    """A reference to the `Popen` object."""

    tabs: MutableMapping[str, Tab]
    """A mapping by target_id of all the targets which are open tabs."""
    targets: MutableMapping[str, Target]
    """A mapping by target_id of ALL the targets."""
    # Don't init instance attributes with mutables
    _watch_dog_task: asyncio.Task[Any] | None = None

    def _make_lock(self) -> None:
        self._open_lock = Lock()

    async def _is_open(self) -> bool:
        # Did we acquire the lock? If so, return true, we locked open.
        # If we are open, we did not lock open.
        # fuck, go through this again
        if self._open_lock.locked():
            return True
        await self._open_lock.acquire()
        return False

    def _release_lock(self) -> bool:
        try:
            if self._open_lock.locked():
                self._open_lock.release()
                return True
            else:
                return False
        except RuntimeError:
            return False

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        browser_cls: type[BrowserImplInterface] = Chromium,
        channel_cls: type[ChannelInterface] = Pipe,
        **kwargs: Any,
    ) -> None:
        """
        Construct a new browser instance.

        Args:
            path: The path to the browser executable.
            browser_cls: The type of browser (default: `Chromium`).
            channel_cls: The type of channel to browser (default: `Pipe`).
            kwargs: The arguments that the browser_cls takes. For example,
                headless=True/False, enable_gpu=True/False, etc.

        """
        _logger.debug("Attempting to open new browser.")

        self._process_executor = _manual_thread_pool.ManualThreadExecutor(
            max_workers=3,
            name="checking_close",
        )
        self._make_lock()
        self.tabs = {}
        self.targets = {}

        # Compose Resources
        self._channel = channel_cls()
        self._broker = Broker(self, self._channel)
        self._browser_impl = browser_cls(self._channel, path, **kwargs)

    def is_isolated(self) -> bool:
        """Return if process is isolated."""
        return self._browser_impl.is_isolated()

    async def open(self) -> None:
        """Open the browser."""
        _logger.info("Opening browser.")
        if await self._is_open():
            raise RuntimeError("Can't re-open the browser")

        # asyncio's equiv doesn't work in all situations
        if hasattr(self._browser_impl, "logger_parser"):
            parser = self._browser_impl.logger_parser
        else:
            parser = None
        self._logger_pipe, _ = logistro.getPipeLogger(
            "browser_proc",
            parser=parser,
        )

        def run() -> subprocess.Popen[bytes] | subprocess.Popen[str]:  # depends on args
            self._browser_impl.pre_open()
            cli = self._browser_impl.get_cli()
            stderr = self._logger_pipe
            env = self._browser_impl.get_env()
            args = self._browser_impl.get_popen_args()
            return subprocess.Popen(  # noqa: S603
                cli,
                stderr=stderr,
                env=env,
                **args,
            )

        _logger.debug("Trying to open browser.")
        loop = asyncio.get_running_loop()
        self.subprocess = await loop.run_in_executor(
            self._process_executor,
            run,
        )

        super().__init__("0", self._broker)
        self._add_session(Session("", self._broker))

        try:
            _logger.debug("Starting watchdog")
            self._watch_dog_task = asyncio.create_task(self._watchdog())
            _logger.debug("Opening channel.")
            self._channel.open()  # should this and below be in a broker run
            _logger.debug("Running read loop")
            self._broker.run_read_loop()
            _logger.debug("Populating Targets")
            await asyncio.sleep(0)  # let watchdog start
            await self.populate_targets()
        except (BrowserClosedError, BrowserFailedError, asyncio.CancelledError) as e:
            if (
                hasattr(self._browser_impl, "missing_libs")
                and self._browser_impl.missing_libs  # type: ignore[reportAttributeAccessIssue]
            ):
                raise BrowserDepsError from e
            raise BrowserFailedError(
                "The browser seemed to close immediately after starting.",
                "You can set the `logging.Logger` level lower to see more output.",
                "You may try installing a known working copy of Chrome by running ",
                "`$ choreo_get_chrome`."
                ""
                "It may be your browser auto-updated and will now work upon "
                "restart. The browser we tried to start is located at "
                f"{self._browser_impl.path}.",
            ) from e

    async def __aenter__(self) -> Self:
        """Open browser as context to launch on entry and close on exit."""
        await self.open()
        return self

    # for use with `await Browser()`
    def __await__(self) -> Generator[Any, Any, Browser]:
        """If you await the `Browser()`, it will implicitly call `open()`."""
        return self.__aenter__().__await__()

    async def _is_closed(self, wait: int | None = 0) -> bool:
        if not hasattr(self, "subprocess"):
            return True
        if wait == 0:
            # poll returns None if its open
            _is_open = self.subprocess.poll() is None
            return not _is_open
        else:
            try:
                loop = asyncio.get_running_loop()

                await loop.run_in_executor(
                    self._process_executor,
                    self.subprocess.wait,
                    wait,
                )
            except subprocess.TimeoutExpired:
                return False
            except asyncio.CancelledError:
                return True
        return True

    # we encapsulate a portion of close that relates solely to browser shutdown
    # that's _close(), but close() handles everything around it
    async def _close(self) -> None:
        if await self._is_closed():
            _logger.debug("No _close(), already is closed")
            return

        try:
            _logger.debug("Trying Browser.close")
            await self.send_command("Browser.close")
        except (BrowserClosedError, BrowserFailedError):
            _logger.debug("Browser is closed trying to send Browser.close")
            return
        except ChannelClosedError:
            _logger.debug("Can't send Browser.close on close channel")
        except asyncio.CancelledError:
            _logger.debug("Close was cancelled, _broker must be shutting down.")

        self._channel.close()

        if await self._is_closed(wait=3):
            return

        if await self._is_closed():
            _logger.debug("Browser is closed after closing channel")
            return
        _logger.warning("Resorting to unclean kill browser.")

        kill(self.subprocess)
        if await self._is_closed(wait=6):
            return
        else:
            raise RuntimeError("Couldn't close or kill browser subprocess")

    async def close(self) -> None:
        """Close the browser."""
        _logger.info("Closing browser.")
        if self._watch_dog_task:
            _logger.debug("Cancelling watchdog.")
            self._watch_dog_task.cancel()
        if not self._release_lock():
            return
        # it can never be mid open here, because all of these must
        # run on the same thread. Do not push open or close to threads.
        try:
            _logger.debug("Starting browser close methods.")
            await self._close()
            _logger.debug("Browser close methods finished.")
        except ProcessLookupError:
            pass
        self._broker.clean()

        _logger.debug("Broker cleaned up.")
        if self._logger_pipe:
            os.close(self._logger_pipe)  # subprocess has it open anyway
            # could have closed this copy immediately
            _logger.debug("Logging pipe closed.")
        self._channel.close()  # was not blocky when comment written
        _logger.debug("Browser channel closed.")
        self._browser_impl.clean()  # os blocky/hangy across networks
        _logger.debug("Browser implementation cleaned up.")
        self._process_executor.shutdown(wait=False, cancel_futures=True)

    async def __aexit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Close the browser."""
        await self.close()
        return None

    async def _watchdog(self) -> None:
        _executor = _manual_thread_pool.ManualThreadExecutor(
            max_workers=1,
            name="watchdog_wait",
        )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=TmpDirWarning)
                _logger.debug("In watchdog")
                loop = asyncio.get_running_loop()
                _logger.debug2("Running wait.")
                await loop.run_in_executor(
                    _executor,
                    self.subprocess.wait,
                )

                _logger.warning("Wait expired, Browser is being closed by watchdog.")
                self._watch_dog_task = (
                    None  # no need for close to cancel, we're going to finish soon
                )
                await self.close()
                await asyncio.sleep(1)
                await loop.run_in_executor(
                    _executor,
                    self._browser_impl.clean,
                )  # this is a backup
        except asyncio.CancelledError:
            pass
        finally:
            _executor.shutdown(wait=False, cancel_futures=True)
            _logger.debug("Watchdog full shutdown (in finally:)")

    def _add_tab(self, tab: Tab) -> None:
        if not isinstance(tab, Tab):
            raise TypeError(f"tab must be an object of {self._tab_type}")
        self.tabs[tab.target_id] = tab

    def _remove_tab(self, target_id: str) -> None:
        if isinstance(target_id, Tab):
            target_id = target_id.target_id
        del self.tabs[target_id]

    def get_tab(self) -> Tab | None:
        """
        Get the first tab if there is one. Useful for default tabs.

        Returns:
            A tab object.

        """
        if self.tabs.values():
            return next(iter(self.tabs.values()))
        return None

    async def populate_targets(self) -> None:
        """Solicit the actual browser for all targets to add to the browser object."""
        if await self._is_closed():
            raise BrowserClosedError("populate_targets() called on a closed browser")
        response = await self.send_command("Target.getTargets")
        if "error" in response:
            raise RuntimeError("Could not get targets") from Exception(
                response["error"],
            )

        for json_response in response["result"]["targetInfos"]:
            if (
                json_response["type"] == "page"
                and json_response["targetId"] not in self.tabs
            ):
                target_id = json_response["targetId"]
                new_tab = Tab(target_id, self._broker)
                try:
                    await new_tab.create_session()
                except protocol.DevtoolsProtocolError as e:
                    if e.code == protocol.Ecode.TARGET_NOT_FOUND:
                        _logger.warning(
                            f"Target {target_id} not found (could be closed before)",
                        )
                        continue
                    else:
                        raise
                self._add_tab(new_tab)
                _logger.debug(f"The target {target_id} was added")

    async def create_session(self) -> Session:
        """
        Create a browser session. Only in supported browsers, is experimental.

        Returns:
            A session object.

        """
        if await self._is_closed():
            raise BrowserClosedError("create_session() called on a closed browser")
        warnings.warn(  # noqa: B028
            "Creating new sessions on Browser() only works with some "
            "versions of Chrome, it is experimental.",
            protocol.ExperimentalFeatureWarning,
        )
        response = await self.send_command("Target.attachToBrowserTarget")
        if "error" in response:
            raise RuntimeError(
                "Could not create session",
            ) from protocol.DevtoolsProtocolError(
                response,
            )
        session_id = response["result"]["sessionId"]
        new_session = Session(session_id, self._broker)
        self._add_session(new_session)
        return new_session

    async def create_tab(
        self,
        url: str = "",
        width: int | None = None,
        height: int | None = None,
        *,
        window: bool = False,
    ) -> Tab:
        """
        Create a new tab.

        Args:
            url: the url to navigate to, default ""
            width: the width of the tab (headless only)
            height: the height of the tab (headless only)
            window: default False, if true, create new window, not tab

        Returns:
            a tab.

        """
        if await self._is_closed():
            raise BrowserClosedError("create_tab() called on a closed browser.")
        params: MutableMapping[str, Any] = {"url": url}
        if width:
            params["width"] = width
        if height:
            params["height"] = height
        if window:
            params["newWindow"] = True

        response = await self.send_command("Target.createTarget", params=params)
        if "error" in response:
            raise RuntimeError(
                "Could not create tab",
            ) from protocol.DevtoolsProtocolError(
                response,
            )
        target_id = response["result"]["targetId"]
        new_tab = Tab(target_id, self._broker)
        self._add_tab(new_tab)
        await new_tab.create_session()
        return new_tab

    async def close_tab(self, target_id: str) -> protocol.BrowserResponse:
        """
        Close a tab by its id.

        Args:
            target_id: the targetId of the tab to close.

        """
        if await self._is_closed():
            raise BrowserClosedError("close_tab() called on a closed browser")
        if isinstance(target_id, Target):
            target_id = target_id.target_id
        # NOTE: we don't need to manually remove sessions because
        # sessions are intrinsically handled by events
        response = await self.send_command(
            command="Target.closeTarget",
            params={"targetId": target_id},
        )
        self._remove_tab(target_id)
        if "error" in response:
            raise RuntimeError(
                "Could not close tab",
            ) from protocol.DevtoolsProtocolError(
                response,
            )
        return response
