"""
_unix_pipe_chromium_wrapper.py provides proper fds to chrome.

By running chromium in a new process (this wrapper), we guarantee
the user hasn't stolen one of our desired file descriptors, which
the OS gives away first-come-first-serve everytime someone opens a
file. chromium demands we use 3 and 4.
"""

from __future__ import annotations

import os

# importing modules has side effects, so we do this before imports
# ruff: noqa: E402

# chromium reads on 3, writes on 4
os.dup2(0, 3)  # make our stdin their input
os.dup2(1, 4)  # make our stdout their output

_inheritable = True
os.set_inheritable(4, _inheritable)
os.set_inheritable(3, _inheritable)

import signal
import subprocess
import sys
from functools import partial
from typing import TYPE_CHECKING

import logistro

if TYPE_CHECKING:
    from types import FrameType

_logger = logistro.getLogger("chrome_wrapper")

# we're a wrapper, the cli is everything that came after us
cli = sys.argv[1:]

print(f"wrapper CLI: {cli}", file=sys.stderr)  # noqa: T201 goes to pipe/logger anyway

process = subprocess.Popen(cli, pass_fds=(3, 4))  # noqa: S603 untrusted input


def kill_proc(
    process: subprocess.Popen[bytes],
    _sig_num: int,
    _frame: FrameType | None,
) -> None:
    process.terminate()
    process.wait(5)  # 5 seconds to clean up nicely, it's a lot
    process.kill()


kp = partial(kill_proc, process)
signal.signal(signal.SIGTERM, kp)
signal.signal(signal.SIGINT, kp)

process.wait()

# not great but it seems that
# pipe isn't always closed when chrome closes
# so we pretend to be chrome and send a bye instead
# also, above depends on async/sync, platform, etc
print("{bye}")  # noqa: T201 we need print here
