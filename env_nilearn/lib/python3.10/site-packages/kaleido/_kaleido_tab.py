from __future__ import annotations

import base64
import json
import time
from typing import TYPE_CHECKING

import logistro
from choreographer.errors import DevtoolsProtocolError

from ._utils import ErrorEntry, to_thread

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    import choreographer as choreo

_logger = logistro.getLogger(__name__)

_TEXT_FORMATS = ("svg", "json")  # eps


class JavascriptError(RuntimeError):  # TODO(A): process better # noqa: TD003, FIX002
    """Used to report errors from javascript."""


### Error definitions ###
class KaleidoError(Exception):
    """An error to interpret errors from Kaleido's JS side."""

    def __init__(self, code, message):
        """
        Construct an error object.

        Args:
            code: the number code of the error.
            message: the message of the error.

        """
        super().__init__(message)
        self._code = code
        self._message = message

    def __str__(self):
        """Display the KaleidoError nicely."""
        return f"Error {self._code}: {self._message}"


def _check_error(result):
    e = _check_error_ret(result)
    if e:
        raise e


def _check_error_ret(result):  # Utility
    """Check browser response for errors. Helper function."""
    if "error" in result:
        return DevtoolsProtocolError(result)
    if result.get("result", {}).get("result", {}).get("subtype", None) == "error":
        return JavascriptError(str(result.get("result")))
    return None


def _make_console_logger(name, log):
    """Create printer specifically for console events. Helper function."""

    async def console_printer(event):
        _logger.debug2(f"{name}:{event}")  # TODO(A): parse # noqa: TD003, FIX002
        log.append(str(event))

    return console_printer


class _KaleidoTab:
    """
    A Kaleido tab is a wrapped choreographer tab providing the functions we need.

    The choreographer tab can be access through the `self.tab` attribute.
    """

    tab: choreo.Tab
    """The underlying choreographer tab."""

    javascript_log: list[Any]
    """A list of console outputs from the tab."""

    def __init__(self, tab, *, _stepper=False):
        """
        Create a new _KaleidoTab.

        Args:
            tab: the choreographer tab to wrap.

        """
        self.tab = tab
        self.javascript_log = []
        self._stepper = _stepper

    def _regenerate_javascript_console(self):
        tab = self.tab
        self.javascript_log = []
        _logger.debug2("Subscribing to all console prints for tab {tab}.")
        tab.unsubscribe("Runtime.consoleAPICalled")
        tab.subscribe(
            "Runtime.consoleAPICalled",
            _make_console_logger("tab js console", self.javascript_log),
        )

    async def navigate(self, url: str | Path = ""):
        """
        Navigate to the kaleidofier script. This is effectively the real initialization.

        Args:
            url: Override the location of the kaleidofier script if necessary.

        """
        tab = self.tab
        javascript_ready = tab.subscribe_once("Runtime.executionContextCreated")
        while javascript_ready.done():
            _logger.debug2("Clearing an old Runtime.executionContextCreated")
            javascript_ready = tab.subscribe_once("Runtime.executionContextCreated")
        page_ready = tab.subscribe_once("Page.loadEventFired")
        while page_ready.done():
            _logger.debug2("Clearing a old Page.loadEventFired")
            page_ready = tab.subscribe_once("Page.loadEventFired")

        _logger.debug2(f"Calling Page.navigate on {tab}")
        _check_error(await tab.send_command("Page.navigate", params={"url": url}))
        # Must enable after navigating.
        _logger.debug2(f"Calling Page.enable on {tab}")
        _check_error(await tab.send_command("Page.enable"))
        _logger.debug2(f"Calling Runtime.enable on {tab}")
        _check_error(await tab.send_command("Runtime.enable"))

        await javascript_ready
        self._current_js_id = (
            javascript_ready.result()
            .get("params", {})
            .get("context", {})
            .get("id", None)
        )
        if not self._current_js_id:
            raise RuntimeError(
                "Refresh sequence didn't work for reload_tab_with_javascript."
                "Result {javascript_ready.result()}.",
            )
        await page_ready
        self._regenerate_javascript_console()

    async def reload(self):
        """Reload the tab, and set the javascript runtime id."""
        tab = self.tab
        _logger.debug(f"Reloading tab {tab} with javascript.")
        javascript_ready = tab.subscribe_once("Runtime.executionContextCreated")
        while javascript_ready.done():
            _logger.debug2("Clearing an old Runtime.executionContextCreated")
            javascript_ready = tab.subscribe_once("Runtime.executionContextCreated")
        is_loaded = tab.subscribe_once("Page.loadEventFired")
        while is_loaded.done():
            _logger.debug2("Clearing an old Page.loadEventFired")
            is_loaded = tab.subscribe_once("Page.loadEventFired")
        _logger.debug2(f"Calling Page.reload on {tab}")
        _check_error(await tab.send_command("Page.reload"))
        await javascript_ready
        self._current_js_id = (
            javascript_ready.result()
            .get("params", {})
            .get("context", {})
            .get("id", None)
        )
        if not self._current_js_id:
            raise RuntimeError(
                "Refresh sequence didn't work for reload_tab_with_javascript."
                "Result {javascript_ready.result()}.",
            )
        await is_loaded
        self._regenerate_javascript_console()

    async def console_print(self, message: str) -> None:
        """
        Print something to the javascript console.

        Args:
            message: The thing to print.

        """
        jsfn = r"function()" r"{" f"console.log('{message}')" r"}"
        params = {
            "functionDeclaration": jsfn,
            "returnByValue": False,
            "userGesture": True,
            "awaitPromise": True,
            "executionContextId": self._current_js_id,
        }

        # send request to run script in chromium
        _logger.debug("Calling js function")
        result = await self.tab.send_command("Runtime.callFunctionOn", params=params)
        _logger.debug(f"Sent javascript got result: {result}")
        _check_error(result)

    def _finish_profile(self, profile, state, error=None):
        _logger.debug("Finishing profile")
        profile["duration"] = float(f"{time.perf_counter() - profile['start']:.6f}")
        del profile["start"]
        profile["state"] = state
        if self.javascript_log:
            profile["js_console"] = self.javascript_log
        if error:
            profile["error"] = error

    async def _write_fig(
        self,
        spec,
        full_path,
        *,
        topojson=None,
        error_log=None,
        profiler=None,
    ):
        """Calculate and write figure to file. Wraps _calc_fig, and writes a file."""
        img, profile = await self._calc_fig(
            spec,
            full_path,
            topojson=topojson,
            error_log=error_log,
            profiler=profiler,
        )

        def write_image(binary):
            with full_path.open("wb") as file:
                file.write(binary)

        _logger.info(f"Starting write of {full_path.name}")
        await to_thread(write_image, img)
        _logger.info(f"Wrote {full_path.name}")

        if profile is not None:
            profile["megabytes"] = full_path.stat().st_size / 1000000
            profile["state"] = "WROTE"

    async def _calc_fig(  # noqa: C901, PLR0912, complexity, branches
        self,
        spec,
        full_path,
        *,
        topojson=None,
        error_log=None,
        profiler=None,
    ):
        """
        Call the plotly renderer via javascript.

        Args:
            spec: the processed plotly figure
            full_path: the path to write the image too. if its a directory, we will try
                to generate a name. If the path contains an extension,
                "path/to/my_image.png", that extension will be the format used if not
                overridden in `opts`.
            opts: dictionary describing format, width, height, and scale of image
            topojson: topojsons are used to customize choropleths
            error_log: A supplied list, will be populated with `ErrorEntry`s
                       which can be converted to strings. Note, this is for
                       collections errors that have to do with plotly. They will
                       not be thrown. Lower level errors (kaleido, choreographer)
                       will still be thrown. If not passed, all errors raise.
            profiler: a supplied dictionary to collect stats about the operation

        """
        tab = self.tab
        execution_context_id = self._current_js_id
        if profiler is not None:
            profile = {
                "name": full_path.name,
                "start": time.perf_counter(),
                "state": "INIT",
            }
        else:
            profile = None

        _logger.debug(f"In tab {tab.target_id[:4]} calc_fig for {full_path.name}.")

        _logger.info(f"Processing {full_path.name}")
        # js script
        kaleido_jsfn = (
            r"function(spec, ...args)"
            r"{"
            r"return kaleido_scopes.plotly(spec, ...args).then(JSON.stringify);"
            r"}"
        )

        # params
        arguments = [{"value": spec}]
        arguments.append({"value": topojson if topojson else None})
        arguments.append({"value": self._stepper})
        params = {
            "functionDeclaration": kaleido_jsfn,
            "arguments": arguments,
            "returnByValue": False,
            "userGesture": True,
            "awaitPromise": True,
            "executionContextId": execution_context_id,
        }

        _logger.info(f"Sending big command for {full_path.name}.")
        if profile:
            profile["state"] = "SENDING"
        result = await tab.send_command("Runtime.callFunctionOn", params=params)
        if profile:
            profile["state"] = "SENT"
        _logger.info(f"Sent big command for {full_path.name}.")
        e = _check_error_ret(result)
        if e:
            if profiler is not None:
                self._finish_profile(profile, "ERROR", e)
                profiler[tab.target_id].append(profile)
            if error_log is not None:
                error_log.append(ErrorEntry(full_path.name, e, self.javascript_log))
                _logger.error(f"Failed {full_path.name}", exc_info=e)
            else:
                _logger.error(f"Raising error on {full_path.name}")
                raise e
        _logger.debug2(f"Result of function call: {result}")
        if self._stepper:
            print(f"Image {full_path.name} was sent to browser")  # noqa: T201
            input("Press Enter to continue...")
        if e:
            return None, None

        img = await self._img_from_response(result)
        if isinstance(img, BaseException):
            if profiler is not None:
                self._finish_profile(profile, "ERROR", img)
                profiler[tab.target_id].append(profile)
            if error_log is not None:
                error_log.append(
                    ErrorEntry(full_path.name, img, self.javascript_log),
                )
                _logger.info(f"Failed {full_path.name}")
                return None, None
            else:
                raise img
        if profile:
            self._finish_profile(profile, "CALCULATED", None)
            profiler[tab.target_id].append(profile)
        return img, profile

    async def _img_from_response(self, response):
        js_response = json.loads(response.get("result").get("result").get("value"))

        if js_response["code"] != 0:
            return KaleidoError(js_response["code"], js_response["message"])

        response_format = js_response.get("format")
        img = js_response.get("result")
        if response_format == "pdf":
            pdf_params = {
                "printBackground": True,
                "marginTop": 0.1,
                "marginBottom": 0.1,
                "marginLeft": 0.1,
                "marginRight": 0.1,
                "preferCSSPageSize": True,
                "pageRanges": "1",
            }
            pdf_response = await self.tab.send_command(
                "Page.printToPDF",
                params=pdf_params,
            )
            e = _check_error_ret(pdf_response)
            if e:
                return e
            img = pdf_response.get("result").get("data")
        # Base64 decode binary types
        if response_format not in _TEXT_FORMATS:
            img = base64.b64decode(img)
        else:
            img = str.encode(img)
        return img
