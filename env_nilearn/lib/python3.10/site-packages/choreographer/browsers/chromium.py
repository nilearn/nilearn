"""Provides a class proving tools for running chromium browsers."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import logistro

if platform.system() == "Windows":
    import msvcrt

from choreographer.channels import Pipe
from choreographer.utils import TmpDirectory, get_browser_path

from ._chrome_constants import chromium_based_browsers

if TYPE_CHECKING:
    import logging
    from typing import Any, Mapping, MutableMapping, Sequence

    from choreographer.channels._interface_type import ChannelInterface

_chromium_wrapper_path = (
    Path(__file__).resolve().parent / "_unix_pipe_chromium_wrapper.py"
)

_packaged_chromium_libs = Path(__file__).resolve().parent / "packaged_chromium_libs"

_logger = logistro.getLogger(__name__)


def _is_exe(path: str | Path) -> bool:
    try:
        return os.access(path, os.X_OK)
    except:  # noqa: E722 bare except ok, weird errors, best effort.
        return False


_logs_parser_regex = re.compile(r"\d*:\d*:\d*\/\d*\.\d*:")


class ChromeNotFoundError(RuntimeError):
    """Raise when browser path can't be determined."""


class Chromium:
    """
    Chromium represents an implementation of the chromium browser.

    It also includes chromium-like browsers (chrome, edge, and brave).
    """

    path: str | Path | None
    """The path to the chromium executable."""
    gpu_enabled: bool
    """True if we should use the gpu. False by default for compatibility."""
    headless: bool
    """True if we should not show the browser, true by default."""
    sandbox_enabled: bool
    """True to enable the sandbox. False by default."""
    skip_local: bool
    """True if we want to avoid looking for our local download when searching path."""
    tmp_dir: TmpDirectory
    """A reference to a temporary directory object the chromium needs to store data."""

    @classmethod
    def find_browser(
        cls,
        *,
        skip_local: bool,
        skip_typical: bool = False,
    ) -> str | None:
        """Find a chromium based browser."""
        for name, browser_data in chromium_based_browsers.items():
            _logger.debug(f"Looking for a {name} browser.")
            path = get_browser_path(
                executable_names=browser_data.exe_names,
                skip_local=skip_local,
                ms_prog_id=browser_data.ms_prog_id,
            )
            if not path and not skip_typical:
                for candidate in browser_data.typical_paths:
                    if _is_exe(candidate):
                        path = candidate
                        break
            if path:
                return path
        return None

    @classmethod
    def logger_parser(
        cls,
        record: logging.LogRecord,
        _old: MutableMapping[str, Any],
    ) -> bool:
        """
        Remove chromium timestamp from chromium's logs.

        This method will be used as the `filter()` method on the `logging.Filter()`
        attached to all incoming logs from the browser process.

        Args:
            record: the `logging.LogRecord` object to read/modify
            _old: data that was already stripped out.

        """
        # replace the chromium timestamp because we do our own
        record.msg = _logs_parser_regex.sub("", record.msg)

        return True

    def _libs_ok(self) -> bool:
        """Return true if libs ok."""
        if self.skip_local:
            _logger.debug(
                "If we HAVE to skip local.",
            )
            return True
        _logger.debug("Checking for libs needed.")
        if platform.system() != "Linux":
            _logger.debug("We're not in linux, so no need for check.")
            return True
        p = None
        try:
            _logger.debug(f"Trying ldd {self.path}")
            p = subprocess.run(  # noqa: S603, validating run with variables
                [  # noqa: S607 path is all we have
                    "ldd",
                    str(self.path),
                ],
                capture_output=True,
                timeout=5,
                check=True,
            )
        except Exception as e:  # noqa: BLE001
            msg = "ldd failed."
            stderr = p.stderr.decode() if p and p.stderr else None
            # Log failure as INFO rather than WARNING so that it's hidden by default,
            # since browser may succeed even if ldd fails
            _logger.info(
                msg  # noqa: G003 + in log
                + f" e: {e}, stderr: {stderr}",
            )
            return False
        if b"not found" in p.stdout:
            msg = "Found deps missing in chrome"
            _logger.debug2(msg + f" {p.stdout.decode()}")
            return False
        _logger.debug("No problems found with dependencies")
        return True

    def __init__(
        self,
        channel: ChannelInterface,
        path: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Construct a chromium browser implementation.

        Args:
            channel: the `choreographer.Channel` we'll be using (WebSockets? Pipe?)
            path: path to the browser
            kwargs:
                gpu_enabled (default False): Turn on GPU? Doesn't work in all envs.
                headless (default True): Actually launch a browser?
                sandbox_enabled (default False): Enable sandbox-
                    a persnickety thing depending on environment, OS, user, etc
                tmp_dir (default None): Manually set the temporary directory

        Raises:
            RuntimeError: Too many kwargs, or browser not found.
            NotImplementedError: Pipe is the only channel type it'll accept right now.

        """
        _logger.info(f"Chromium init'ed with kwargs {kwargs}")
        self.path = path
        self.gpu_enabled = kwargs.pop("enable_gpu", False)
        self.headless = kwargs.pop("headless", True)
        self.sandbox_enabled = kwargs.pop("enable_sandbox", False)
        self._tmp_dir_path = kwargs.pop("tmp_dir", None)
        if kwargs:
            raise RuntimeError(
                f"Chromium.get_cli() received invalid args: {kwargs.keys()}",
            )
        self.skip_local = bool(
            "ubuntu" in platform.version().lower() and self.sandbox_enabled,
        )

        if self.skip_local:
            _logger.warning(
                "Forced skipping local. Ubuntu sandbox requires package manager.",
            )

        if not self.path:
            self.path = Chromium.find_browser(skip_local=self.skip_local)
        if not self.path:
            raise ChromeNotFoundError(
                "Browser not found. You can use get_chrome() or "
                "choreo_get_chrome from bash. please see documentation. "
                f"Local copy ignored: {self.skip_local}.",
            )
        _logger.info(f"Found chromium path: {self.path}")

        self._channel = channel
        if not isinstance(channel, Pipe):
            raise NotImplementedError("Websocket style channels not implemented yet.")

        self._is_isolated = "snap" in str(self.path)

    def pre_open(self) -> None:
        """Prepare browser for opening."""
        self.tmp_dir = TmpDirectory(
            path=self._tmp_dir_path,
            sneak=self._is_isolated,
        )
        self.missing_libs = not self._libs_ok()
        _logger.info(f"Temporary directory at: {self.tmp_dir.path}")

    def is_isolated(self) -> bool:
        """
        Return if /tmp directory is isolated by OS.

        Returns:
            bool indicating if /tmp is isolated.

        """
        return self._is_isolated

    def get_popen_args(self) -> Mapping[str, Any]:
        """Return the args needed to runc chromium with `subprocess.Popen()`."""
        args = {}
        # need to check pipe
        if platform.system() == "Windows":
            args["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore [attr-defined]
            args["close_fds"] = False
        else:
            args["close_fds"] = True
            if isinstance(self._channel, Pipe):
                args["stdin"] = self._channel.from_choreo_to_external
                args["stdout"] = self._channel.from_external_to_choreo
        _logger.debug(f"Returning args: {args}")
        return args

    def get_cli(self) -> Sequence[str]:
        """Return the CLI command for chromium."""
        if platform.system() != "Windows":
            cli = [
                str(sys.executable),
                str(_chromium_wrapper_path),
                str(self.path),
            ]
        else:
            cli = [
                str(self.path),
            ]

        if not self.gpu_enabled:
            cli.append("--disable-gpu")
        if self.headless:
            cli.append("--headless")
        if not self.sandbox_enabled:
            cli.append("--no-sandbox")

        cli.extend(
            [
                "--disable-breakpad",
                "--allow-file-access-from-files",
                "--enable-logging=stderr",
                f"--user-data-dir={self.tmp_dir.path}",
                "--no-first-run",
                "--enable-unsafe-swiftshader",
                "--disable-dev-shm-usage",
                "--disable-background-media-suspend",
                "--disable-lazy-loading",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-component-update",
                "--disable-hang-monitor",
                "--disable-popup-blocking",
                "--disable-prompt-on-repost",
                "--disable-ipc-flooding-protection",
                "--disable-sync",
                "--metrics-recording-only",
                "--password-store=basic",
                "--use-mock-keychain",
                "--no-default-browser-check",
                "--no-process-per-site",
                "--disable-web-security",
            ],
        )
        if isinstance(self._channel, Pipe):
            cli.append("--remote-debugging-pipe")
            if platform.system() == "Windows":
                # its gonna read on 3
                # its gonna write on 4
                r_handle = msvcrt.get_osfhandle(self._channel.from_choreo_to_external)  # type: ignore [attr-defined]
                w_handle = msvcrt.get_osfhandle(self._channel.from_external_to_choreo)  # type: ignore [attr-defined]
                _inheritable = True
                os.set_handle_inheritable(r_handle, _inheritable)  # type: ignore [attr-defined]
                os.set_handle_inheritable(w_handle, _inheritable)  # type: ignore [attr-defined]
                cli += [
                    f"--remote-debugging-io-pipes={r_handle!s},{w_handle!s}",
                ]
        _logger.debug(f"Returning cli: {cli}")
        return cli

    def get_env(self) -> MutableMapping[str, str]:
        """Return the env needed for chromium."""
        env = os.environ.copy()
        return env

    def clean(self) -> None:
        """Clean up any leftovers form browser, like tmp files."""
        if hasattr(self, "tmp_dir"):
            self.tmp_dir.clean()

    def __del__(self) -> None:
        """Delete the temporary file and run `clean()`."""
        self.clean()
