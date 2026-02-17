import queue
import threading
from concurrent.futures import Executor, Future

import logistro

_logger = logistro.getLogger(__name__)


class ExecutorClosedError(RuntimeError):
    """Raise if submitting when executor is closed."""


class ManualThreadExecutor(Executor):
    def __init__(self, *, max_workers=2, daemon=True, name="manual-exec"):
        self._q = queue.Queue()
        self._stop = False
        self._threads = []
        self.name = name
        for i in range(max_workers):
            t = threading.Thread(
                target=self._worker,
                name=f"{name}-{i}",
                daemon=daemon,
            )
            t.start()
            self._threads.append(t)

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:  # sentinel
                return
            fn, args, kwargs, fut = item
            if fut.set_running_or_notify_cancel():
                try:
                    res = fn(*args, **kwargs)
                except BaseException as e:  # noqa: BLE001 yes we catch and set
                    fut.set_exception(e)
                else:
                    fut.set_result(res)
            self._q.task_done()

    def submit(self, fn, *args, **kwargs):
        fut = Future()
        if self._stop:
            fut.set_exception(ExecutorClosedError("Cannot submit tasks."))
            return fut
        self._q.put((fn, args, kwargs, fut))
        return fut

    def shutdown(self, wait=True, *, cancel_futures=False):  # noqa: FBT002 overriding, can't change args
        self._stop = True
        if cancel_futures:
            # Drain queue and cancel pending
            try:
                while True:
                    _, _, _, fut = self._q.get_nowait()
                    fut.cancel()
                    self._q.task_done()
            except queue.Empty:
                pass
        for _ in self._threads:
            self._q.put(None)
        if wait:
            for t in self._threads:
                t.join(timeout=2.0)
