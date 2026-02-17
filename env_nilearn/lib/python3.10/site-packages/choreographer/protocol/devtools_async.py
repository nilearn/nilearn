"""Provide a lower-level async interface to the Devtools Protocol."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import logistro

from choreographer import protocol

if TYPE_CHECKING:
    import asyncio
    from typing import Any, Callable, Coroutine, MutableMapping

    from choreographer._brokers import Broker

_logger = logistro.getLogger(__name__)


class Session:
    """A session is a single conversation with a single target."""

    session_id: str
    """The id of the session given by the browser."""
    message_id: int
    """All messages are counted per session and this is the current message id."""
    subscriptions: MutableMapping[
        str,
        tuple[
            Callable[[protocol.BrowserResponse], Coroutine[Any, Any, Any]],
            bool,
        ],
    ]

    def __init__(self, session_id: str, broker: Broker) -> None:
        """
        Construct a session from the browser as an object.

        A session is like an open conversation with a target.
        All commands are sent on sessions.

        Args:
            broker:  a reference to the browser's broker
            session_id:  the id given by the browser

        """
        if not isinstance(session_id, str):
            raise TypeError("session_id must be a string")
        # Resources
        self._broker = broker

        # State
        self.session_id = session_id
        _logger.debug(f"New session: {session_id}")
        self.message_id = 0
        self.subscriptions = {}

    async def send_command(
        self,
        command: str,
        params: MutableMapping[str, Any] | None = None,
    ) -> protocol.BrowserResponse:
        """
        Send a devtools command on the session.

        https://chromedevtools.github.io/devtools-protocol/

        Args:
            command: devtools command to send
            params: the parameters to send

        Returns:
            A message key (session, message id) tuple or None

        """
        current_id = self.message_id
        self.message_id += 1
        json_command = protocol.BrowserCommand(
            {
                "id": current_id,
                "method": command,
            },
        )

        if self.session_id:
            json_command["sessionId"] = self.session_id
        if params:
            json_command["params"] = params
        _logger.debug(
            f"Cmd '{command}', param keys '{params.keys() if params else ''}', "
            f"sessionId '{self.session_id}'",
        )
        _logger.debug2(f"Full params: {str(params).replace('%', '%%')}")
        return await self._broker.write_json(json_command)

    def subscribe(
        self,
        string: str,
        callback: Callable[[protocol.BrowserResponse], Coroutine[Any, Any, Any]],
        *,
        repeating: bool = True,
    ) -> None:
        """
        Subscribe to an event on this session.

        Args:
            string: the name of the event. Can use * wildcard at the end.
            callback: the callback (which takes a message dict and returns nothing)
            repeating: default True, should the callback execute more than once

        """
        if not inspect.iscoroutinefunction(callback):
            raise TypeError(
                "Call back must be be `async def` type function.",
            )
        if string in self.subscriptions:
            raise ValueError(
                "You are already subscribed to this string, "
                "duplicate subscriptions are not allowed.",
            )
        else:
            # so this should be per session
            # and that means we need a list of all sessions
            self.subscriptions[string] = (callback, repeating)

    def unsubscribe(self, string: str) -> None:
        """
        Remove a subscription.

        Args:
            string: the subscription to remove.

        """
        if string not in self.subscriptions:
            return
        del self.subscriptions[string]

    def subscribe_once(self, string: str) -> asyncio.Future[Any]:
        """
        Return a future for a browser event.

        Generally python asyncio doesn't recommend futures.

        But in this case, one must call subscribe_once and await it later,
        generally because they must subscribe and then provoke the event.

        Args:
            string: the event to subscribe to

        Returns:
            A future to be awaited later, the complete event.

        """
        return self._broker.new_subscription_future(self.session_id, string)


class Target:
    """A target like a browser, tab, or others. It sends commands. It has sessions."""

    target_id: str
    """The browser's ID of the target."""
    sessions: MutableMapping[str, Session]
    """A list of all the sessions for this target."""

    def __init__(self, target_id: str, broker: Broker):
        """
        Create a target after one ahs been created by the browser.

        Args:
            broker:  a reference to the browser's broker
            target_id:  the id given by the browser

        """
        if not isinstance(target_id, str):
            raise TypeError("target_id must be string")
        # Resources
        self._broker = broker

        # States
        self.sessions = {}
        self.target_id = target_id
        _logger.debug(f"Created new target {target_id}.")

    def _add_session(self, session: Session) -> None:
        if not isinstance(session, Session):
            raise TypeError("session must be a session type class")
        self.sessions[session.session_id] = session

    def _remove_session(self, session_id: str) -> None:
        if isinstance(session_id, Session):
            session_id = session_id.session_id
        _ = self.sessions.pop(session_id, None)

    def get_session(self) -> Session:
        """Retrieve the first session of the target, if it exists."""
        if not self.sessions.values():
            raise RuntimeError(
                "Cannot use this method without at least one valid session",
            )
        session = next(iter(self.sessions.values()))
        return session

    async def send_command(
        self,
        command: str,
        params: MutableMapping[str, Any] | None = None,
    ) -> protocol.BrowserResponse:
        """
        Send a command to the first session in a target.

        https://chromedevtools.github.io/devtools-protocol/

        Args:
            command: devtools command to send
            params: the parameters to send

        """
        if not self.sessions.values():
            raise RuntimeError("Cannot send_command without at least one valid session")
        session = self.get_session()
        return await session.send_command(command, params)

    async def create_session(self) -> Session:
        """Create a new session on this target."""
        response = await self._broker._browser.send_command(  # noqa: SLF001 yeah we need the browser :-(
            "Target.attachToTarget",
            params={"targetId": self.target_id, "flatten": True},
        )
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

    # async only
    async def close_session(
        self,
        session_id: str,
    ) -> protocol.BrowserResponse:
        """
        Close a session by session_id.

        Args:
            session_id: the session to close

        """
        if isinstance(session_id, Session):
            session_id = session_id.session_id
        response = await self._broker._browser.send_command(  # noqa: SLF001 we need browser
            command="Target.detachFromTarget",
            params={"sessionId": session_id},
        )

        self._remove_session(session_id)
        if "error" in response:
            raise RuntimeError(
                "Could not close session",
            ) from protocol.DevtoolsProtocolError(
                response,
            )
        _logger.debug(f"The session {session_id} has been closed.")
        return response
        # kinda hate, why do we need this again?

    def subscribe(
        self,
        string: str,
        callback: Callable[[protocol.BrowserResponse], Coroutine[Any, Any, Any]],
        *,
        repeating: bool = True,
    ) -> None:
        """
        Subscribe to an event on the main session of this target.

        Args:
            string: the name of the event. Can use * wildcard at the end.
            callback: the callback (which takes a message dict and returns nothing)
            repeating: default True, should the callback execute more than once

        """
        session = self.get_session()
        session.subscribe(string, callback, repeating=repeating)

    def unsubscribe(self, string: str) -> None:
        """
        Remove a subscription.

        Args:
            string: the subscription to remove.

        """
        session = self.get_session()
        session.unsubscribe(string)

    def subscribe_once(self, string: str) -> asyncio.Future[Any]:
        """
        Return a future for a browser event for the first session of this target.

        Generally python asyncio doesn't recommend futures.

        But in this case, one must call subscribe_once and await it later,
        generally because they must subscribe and then provoke the event.

        Args:
            string: the event to subscribe to

        Returns:
            A future to be awaited later, the complete event.

        """
        session = self.get_session()
        return session.subscribe_once(string)
