from __future__ import annotations

import asyncio
import atexit
import warnings
from functools import partial
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING, NamedTuple

from .kaleido import Kaleido

if TYPE_CHECKING:
    from typing import Any


class Task(NamedTuple):
    fn: str
    args: Any
    kwargs: Any


class _BadFunctionName(BaseException):
    """For use when programmed poorly."""


class GlobalKaleidoServer:
    _instance = None

    async def _server(self, *args, **kwargs):
        async with Kaleido(*args, **kwargs) as k:  # multiple processor? Enable GPU?
            while True:
                task = self._task_queue.get()  # thread dies if main thread dies
                if task is None:
                    self._task_queue.task_done()
                    return
                if not hasattr(k, task.fn):
                    raise _BadFunctionName(f"Kaleido has no attribute {task.fn}")
                try:
                    self._return_queue.put(
                        await getattr(k, task.fn)(*task.args, **task.kwargs),
                    )
                except Exception as e:  # noqa: BLE001
                    self._return_queue.put(e)

                self._task_queue.task_done()

    def __new__(cls):
        # Create the singleton on first instantiation
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False  # noqa: SLF001
        return cls._instance

    def is_running(self):
        return self._initialized

    def open(self, *args: Any, silence_warnings=False, **kwargs: Any) -> None:
        """Initialize the singleton with three values."""
        if self.is_running():
            if not silence_warnings:
                warnings.warn(
                    "Server already open.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return
        coroutine = self._server(*args, **kwargs)
        self._thread: Thread = Thread(
            target=asyncio.run,
            args=(coroutine,),
            daemon=True,
        )
        self._task_queue: Queue[Task | None] = Queue()
        self._return_queue: Queue[Any] = Queue()
        self._thread.start()
        self._initialized = True
        close = partial(self.close, silence_warnings=True)
        atexit.register(close)

    def close(self, *, silence_warnings=False):
        """Reset the singleton back to an uninitialized state."""
        if not self.is_running():
            if not silence_warnings:
                warnings.warn(
                    "Server already closed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return
        self._task_queue.put(None)
        self._thread.join()
        del self._thread
        del self._task_queue
        del self._return_queue
        self._initialized = False

    def call_function(self, cmd: str, *args, **kwargs):
        if not self.is_running():
            raise RuntimeError("Can't call function on stopped server.")
        if kwargs.pop("kopts", None):
            warnings.warn(
                "The kopts argument is ignored if using a server.",
                UserWarning,
                stacklevel=3,
            )
        self._task_queue.put(Task(cmd, args, kwargs))
        self._task_queue.join()
        res = self._return_queue.get()
        if isinstance(res, BaseException):
            raise res
        else:
            return res


def oneshot_async_run(func, args: tuple[Any, ...], kwargs: dict):
    q: Queue[Any] = Queue(maxsize=1)

    def run(func, q, *args, **kwargs):
        # func is a closure
        try:
            q.put(asyncio.run(func(*args, **kwargs)))
        except BaseException as e:  # noqa: BLE001
            q.put(e)

    t = Thread(target=run, args=(func, q, *args), kwargs=kwargs)
    t.start()
    t.join()
    res = q.get()
    if isinstance(res, BaseException):
        raise res
    else:
        return res
