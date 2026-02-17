from __future__ import annotations

import asyncio
import warnings
from functools import partial
from typing import TYPE_CHECKING

import logistro

from choreographer import channels, protocol
from choreographer.utils import _manual_thread_pool

# afrom choreographer.channels import ChannelClosedError

if TYPE_CHECKING:
    from typing import Any, MutableMapping

    from choreographer.browser_async import Browser
    from choreographer.channels._interface_type import ChannelInterface
    from choreographer.protocol.devtools_async import Session, Target


_logger = logistro.getLogger(__name__)


class UnhandledMessageWarning(UserWarning):
    pass


class Broker:
    """Broker is a middleware implementation for asynchronous implementations."""

    _browser: Browser
    """Browser is a reference to the Browser object this broker is brokering for."""
    _channel: ChannelInterface
    """
    Channel will be the ChannelInterface implementation (pipe or websocket)
    that the broker communicates on.
    """
    futures: MutableMapping[protocol.MessageKey, asyncio.Future[Any]]
    """A mapping of all the futures for all sent commands."""

    _subscriptions_futures: MutableMapping[
        str,
        MutableMapping[
            str,
            list[asyncio.Future[Any]],
        ],
    ]
    """A mapping of session id: subscription: list[futures]"""

    def __init__(self, browser: Browser, channel: ChannelInterface) -> None:
        """
        Construct a broker for a synchronous arragenment w/ both ends.

        Args:
            browser: The sync browser implementation.
            channel: The channel the browser uses to talk on.

        """
        self._browser = browser
        self._channel = channel
        self._background_tasks: set[asyncio.Task[Any]] = set()
        # if its a task you dont want canceled at close (like the close task)
        self._background_tasks_cancellable: set[asyncio.Task[Any]] = set()
        # if its a user task, can cancel
        self._current_read_task: asyncio.Task[Any] | None = None
        self.futures = {}
        self._subscriptions_futures = {}

        self._write_lock = asyncio.Lock()
        self._executor = _manual_thread_pool.ManualThreadExecutor(
            max_workers=2,
            name="readwrite_thread",
        )

    def new_subscription_future(
        self,
        session_id: str,
        subscription: str,
    ) -> asyncio.Future[Any]:
        _logger.debug(
            f"Session {session_id} is subscribing to {subscription} one time.",
        )
        if session_id not in self._subscriptions_futures:
            self._subscriptions_futures[session_id] = {}
        if subscription not in self._subscriptions_futures[session_id]:
            self._subscriptions_futures[session_id][subscription] = []
        future = asyncio.get_running_loop().create_future()
        self._subscriptions_futures[session_id][subscription].append(future)
        return future

    def clean(self) -> None:
        _logger.debug("Cancelling message futures")
        for future in self.futures.values():
            if not future.done():
                _logger.debug2(f"Cancelling {future}")
                future.cancel()
        _logger.debug("Cancelling read task")
        if self._current_read_task and not self._current_read_task.done():
            _logger.debug2(f"Cancelling read: {self._current_read_task}")
            self._current_read_task.cancel()
        _logger.debug("Cancelling subscription-futures")
        for session in self._subscriptions_futures.values():
            for query in session.values():
                for future in query:
                    if not future.done():
                        _logger.debug2(f"Cancelling {future}")
                        future.cancel()
        _logger.debug("Cancelling background tasks")
        for task in self._background_tasks_cancellable:
            if not task.done():
                _logger.debug2(f"Cancelling {task}")
                task.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def run_read_loop(self) -> None:  # noqa: C901, PLR0915 complexity
        def check_read_loop_error(result: asyncio.Future[Any]) -> None:
            if result.cancelled():
                _logger.debug("Readloop cancelled")
                return
            e = result.exception()
            if e:
                _logger.debug("Error in readloop. Will post a close() task.")
                self._background_tasks.add(
                    asyncio.create_task(self._browser.close()),
                )
                if isinstance(e, channels.ChannelClosedError):
                    _logger.debug("PipeClosedError caught")
                    _logger.debug2("Full Error:", exc_info=e)
                elif isinstance(e, asyncio.CancelledError):
                    _logger.debug("CancelledError caught.")
                    _logger.debug2("Full Error:", exc_info=e)
                else:
                    _logger.error("Error in run_read_loop.", exc_info=e)
                    raise e

        async def read_loop() -> None:  # noqa: PLR0912, PLR0915, C901
            loop = asyncio.get_running_loop()
            fn = partial(self._channel.read_jsons, blocking=True)
            responses = await loop.run_in_executor(
                executor=self._executor,
                func=fn,
            )
            _logger.debug(f"Channel read found {len(responses)} json objects.")
            for response in responses:
                error = protocol.get_error_from_result(response)
                key = protocol.calculate_message_key(response)
                if not key and error:
                    raise protocol.DevtoolsProtocolError(response)

                # looks for event that we should handle internally
                self._check_for_closed_session(response)
                # surrounding lines overlap in idea
                if protocol.is_event(response):
                    event_session_id = response.get(
                        "sessionId",
                        "",
                    )
                    _logger.debug2(f"Is event for {event_session_id}")
                    x = self._get_target_session_by_session_id(
                        event_session_id,
                    )
                    if not x:
                        continue
                    _, event_session = x
                    if not event_session:
                        _logger.error("Found an event that returned no session.")
                        continue
                    _logger.debug(
                        f"Received event {response['method']} for "
                        f"{event_session_id} targeting {event_session}.",
                    )

                    session_futures = self._subscriptions_futures.get(
                        event_session_id,
                    )
                    _logger.debug2(
                        "Checking for event subscription future.",
                    )
                    if session_futures:
                        for query in session_futures:
                            match = (
                                query.endswith("*")
                                and response["method"].startswith(query[:-1])
                            ) or (response["method"] == query)
                            if match:
                                _logger.debug2(
                                    "Found event subscription future.",
                                )
                                for future in session_futures[query]:
                                    if not future.done():
                                        future.set_result(response)
                                session_futures[query] = []

                    _logger.debug2(
                        "Checking for event subscription callback.",
                    )
                    for query in list(event_session.subscriptions):
                        match = (
                            query.endswith("*")
                            and response["method"].startswith(query[:-1])
                        ) or (response["method"] == query)
                        _logger.debug2(
                            "Found event subscription callback.",
                        )
                        if match:
                            t: asyncio.Task[Any] = asyncio.create_task(
                                event_session.subscriptions[query][0](response),
                            )
                            self._background_tasks_cancellable.add(t)
                            if not event_session.subscriptions[query][1]:
                                event_session.unsubscribe(query)

                elif key:
                    _logger.debug(f"Have a response with key {key}")
                    if key in self.futures:
                        _logger.debug(f"Found future for key {key}")
                        future = self.futures.pop(key)
                    elif "error" in response:
                        raise protocol.DevtoolsProtocolError(response)
                    else:
                        raise RuntimeError(f"Couldn't find a future for key: {key}")
                    if not future.done():
                        future.set_result(response)
                else:
                    warnings.warn(
                        f"Unhandled message type:{response!s}",
                        UnhandledMessageWarning,
                        stacklevel=1,
                    )
            read_task = asyncio.create_task(read_loop())
            read_task.add_done_callback(check_read_loop_error)
            self._current_read_task = read_task

        read_task = asyncio.create_task(read_loop())
        read_task.add_done_callback(check_read_loop_error)
        self._current_read_task = read_task

    async def write_json(
        self,
        obj: protocol.BrowserCommand,
    ) -> protocol.BrowserResponse:
        protocol.verify_params(obj)
        key = protocol.calculate_message_key(obj)
        _logger.debug1(f"Broker writing {obj['method']} with key {key}")
        if not key:
            raise RuntimeError(
                "Message strangely formatted and "
                "choreographer couldn't figure it out why.",
            )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[protocol.BrowserResponse] = loop.create_future()
        self.futures[key] = future
        _logger.debug(f"Created future: {key} {future}")
        try:
            async with self._write_lock:  # this should be a queue not a lock
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    self._executor,
                    self._channel.write_json,
                    obj,
                )
        except (_manual_thread_pool.ExecutorClosedError, asyncio.CancelledError) as e:
            if not future.cancel() or not future.cancelled():
                await future  # it wasn't canceled, so listen to it before raising
            raise channels.ChannelClosedError("Executor is closed.") from e
        except Exception as e:  # noqa: BLE001
            future.set_exception(e)
            del self.futures[key]
            _logger.debug(f"Future for {key} deleted.")

        return await future

    def _get_target_session_by_session_id(
        self,
        session_id: str,
    ) -> tuple[Target, Session] | None:
        if session_id == "":
            return (self._browser, self._browser.sessions[session_id])
        for tab in self._browser.tabs.values():
            if session_id in tab.sessions:
                return (tab, tab.sessions[session_id])
        if session_id in self._browser.sessions:
            return (self._browser, self._browser.sessions[session_id])
        return None

    def _check_for_closed_session(self, response: protocol.BrowserResponse) -> bool:
        if "method" in response and response["method"] == "Target.detachedFromTarget":
            session_closed = response["params"].get(
                "sessionId",
                "",
            )
            if session_closed == "":
                _logger.debug2("Found closed session through events.")
                return True

            x = self._get_target_session_by_session_id(session_closed)
            if x:
                target_closed, _ = x
            else:
                return False

            if target_closed:
                target_closed._remove_session(session_closed)  # noqa: SLF001
                _logger.debug(
                    "Using intern subscription key: "
                    "'Target.detachedFromTarget'. "
                    f"Session {session_closed} was closed.",
                )
                _logger.debug2("Found closed session through events.")
                return True
            return False
        else:
            return False
