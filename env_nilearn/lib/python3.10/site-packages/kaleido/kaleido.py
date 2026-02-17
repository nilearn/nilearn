"""the kaleido module kaleido.py provides the main classes for the kaleido package."""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncIterable, Iterable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

import choreographer as choreo
import logistro
from choreographer.errors import ChromeNotFoundError
from choreographer.utils import TmpDirectory

from ._fig_tools import _is_figurish, build_fig_spec
from ._kaleido_tab import _KaleidoTab
from ._page_generator import PageGenerator
from ._utils import ErrorEntry, warn_incompatible_plotly

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Any, Callable, Coroutine

    from . import _fig_tools

_logger = logistro.getLogger(__name__)

try:
    from plotly.utils import PlotlyJSONEncoder  #  noqa: I001
    from choreographer import channels

    channels.register_custom_encoder(PlotlyJSONEncoder)
    _logger.debug("Successfully registered PlotlyJSONEncoder.")
except ImportError as e:
    _logger.debug(f'Couldn\'t import plotly due to "{e!s}" - skipping.')

# Show a warning if the installed Plotly version
# is incompatible with this version of Kaleido
warn_incompatible_plotly()


def _make_printer(name: str) -> Callable[[Any], Coroutine[Any, Any, None]]:
    """Create event printer for generic events. Helper function."""

    async def print_all(response: Any) -> None:
        _logger.debug2(f"{name}:{response}")

    return print_all


class Kaleido(choreo.Browser):
    """
    Kaleido manages a set of image processors.

    It can be used as a context (`async with Kaleido(...)`), but can
    also be used like:

    ```
    k = Kaleido(...)
    k = await Kaleido.open()
    ... # do stuff
    k.close()
    ```
    """

    tabs_ready: asyncio.Queue[_KaleidoTab]
    """A queue of ready tabs."""
    _background_render_tasks: set[asyncio.Task]
    # not really render tasks
    _main_tasks: set[asyncio.Task]

    async def close(self) -> None:
        """Close the browser."""
        await super().close()
        if self._tmp_dir:
            self._tmp_dir.clean()
        _logger.info("Cancelling tasks.")
        for task in self._main_tasks:
            if not task.done():
                task.cancel()
        for task in self._background_render_tasks:
            if not task.done():
                task.cancel()
        _logger.info("Exiting Kaleido/Choreo")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Close the browser."""
        _logger.info("Waiting for all cleanups to finish.")
        await asyncio.gather(*self._background_render_tasks, return_exceptions=True)
        _logger.info("Exiting Kaleido")
        return await super().__aexit__(exc_type, exc_value, exc_tb)

    def __init__(  # noqa: D417, PLR0913 no args/kwargs in description
        self,
        *args: Any,
        page_generator: None | PageGenerator | str | Path = None,
        n: int = 1,
        timeout: int | None = 90,
        width: int | None = None,  # deprecate
        height: int | None = None,  # deprecate
        stepper: bool = False,
        plotlyjs: str | None = None,
        mathjax: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Kaleido, a `choreo.Browser` wrapper adding kaleido functionality.

        It takes all `choreo.Browser` args, plus some extra. The extra
        are listed, see choreographer for more documentation.

        Note: Chrome will throttle background tabs and windows, so non-headless
        multi-process configurations don't work well.

        For argument `page`, if it is a string, it must be passed as a fully-qualified
        URI, like `file://` or `https://`.
        If it is a `Path`, `Path`'s `as_uri()` will be called.
        If it is a string or path, its expected to be an HTML file, one will not
        be generated.

        Args:
            n: the number of separate processes (windows, not seen) to use.
            timeout: limit on any single render (default 90 seconds).
            width: width of window (headless only)
            height: height of window (headless only)
            page: This can be a `kaleido.PageGenerator`, a `pathlib.Path`, or a string.

        """
        self._background_render_tasks = set()
        self._main_tasks = set()
        self.tabs_ready = asyncio.Queue(maxsize=0)
        self._total_tabs = 0
        self._tmp_dir = None

        page = page_generator
        self._timeout = timeout
        self._n = n
        self._height = height  # deprecate
        self._width = width  # deprecate
        self._stepper = stepper
        self._plotlyjs = plotlyjs
        self._mathjax = mathjax
        if not kwargs.get("headless", True) and (self._height or self._width):
            warnings.warn(
                "Height and Width can only be used if headless=True, "
                "ignoring both sizes.",
                stacklevel=1,
            )
            self._height = None
            self._width = None
        _logger.debug(f"Timeout: {self._timeout}")

        try:
            super().__init__(*args, **kwargs)
        except ChromeNotFoundError:
            raise ChromeNotFoundError(
                "Kaleido v1 and later requires Chrome to be installed. "
                "To install Chrome, use the CLI command `kaleido_get_chrome`, "
                "or from Python, use either `kaleido.get_chrome()` "
                "or `kaleido.get_chrome_sync()`.",
            ) from ChromeNotFoundError

        # do this during open because it requires close
        self._saved_page_arg = page

    async def open(self):
        """Build temporary file if we need one."""
        page = self._saved_page_arg
        del self._saved_page_arg

        if isinstance(page, str):
            if page.startswith(r"file://") and Path(unquote(urlparse(page).path)):
                self._index = page
            elif Path(page).is_file():
                self._index = Path(page).as_uri()
            else:
                raise FileNotFoundError(f"{page} does not exist.")
        elif isinstance(page, Path):
            if page.is_file():
                self._index = page.as_uri()
            else:
                raise FileNotFoundError(f"{page!s} does not exist.")
        else:
            self._tmp_dir = TmpDirectory(sneak=self.is_isolated())
            index = self._tmp_dir.path / "index.html"
            self._index = index.as_uri()
            if not page:
                page = PageGenerator(plotly=self._plotlyjs, mathjax=self._mathjax)
            page.generate_index(index)
        await super().open()

    async def _conform_tabs(self, tabs: list[choreo.Tab] | None = None) -> None:
        if not tabs:
            tabs = list(self.tabs.values())
        _logger.info(f"Conforming {len(tabs)} to {self._index}")

        for i, tab in enumerate(tabs):
            n = f"tab-{i!s}"
            _logger.debug2(f"Subscribing * to tab: {tab}.")
            tab.subscribe("*", _make_printer(n + " event"))

        _logger.debug("Navigating all tabs")

        kaleido_tabs = [_KaleidoTab(tab, _stepper=self._stepper) for tab in tabs]

        # A little hard to read because we don't have TaskGroup in this version
        tasks = [asyncio.create_task(tab.navigate(self._index)) for tab in kaleido_tabs]
        _logger.info("Waiting on all navigates")
        await asyncio.gather(*tasks)
        _logger.info("All navigates done, putting them all in queue.")

        for ktab in kaleido_tabs:
            await self.tabs_ready.put(ktab)
        self._total_tabs = len(kaleido_tabs)
        _logger.debug("Tabs fully navigated/enabled/ready")

    async def populate_targets(self) -> None:
        """
        Override the browser populate_targets to ensure the correct page.

        Is called automatically during initialization, and should only be called
        once ever per object.
        """
        await super().populate_targets()
        await self._conform_tabs()
        needed_tabs = self._n - len(self.tabs)
        if needed_tabs < 0:
            raise RuntimeError("Did you set 0 or less tabs?")
        if not needed_tabs:
            return
        tasks = [
            asyncio.create_task(self._create_kaleido_tab()) for _ in range(needed_tabs)
        ]

        await asyncio.gather(*tasks)
        for tab in self.tabs.values():
            _logger.info(f"Tab ready: {tab.target_id}")

    async def _create_kaleido_tab(
        self,
    ) -> None:
        """
        Create a tab with the kaleido script.

        Returns:
            The kaleido-tab created.

        """
        tab = await super().create_tab(
            url="",
            width=self._width,
            height=self._height,
            window=True,
        )
        await self._conform_tabs([tab])

    async def _get_kaleido_tab(self) -> _KaleidoTab:
        """
        Retrieve an available tab from queue.

        Returns:
            A kaleido-tab from the queue.

        """
        _logger.info(f"Getting tab from queue (has {self.tabs_ready.qsize()})")
        if not self._total_tabs:
            raise RuntimeError(
                "Before generating a figure, you must await `k.open()`.",
            )
        tab = await self.tabs_ready.get()
        _logger.info(f"Got {tab.tab.target_id[:4]}")
        return tab

    async def _return_kaleido_tab(self, tab: _KaleidoTab) -> None:
        """
        Refresh tab and put it back into the available queue.

        Args:
            tab: the kaleido tab to return.

        """
        _logger.info(f"Reloading tab {tab.tab.target_id[:4]} before return.")
        await tab.reload()
        _logger.info(
            f"Putting tab {tab.tab.target_id[:4]} back (queue size: "
            f"{self.tabs_ready.qsize()}).",
        )
        await self.tabs_ready.put(tab)
        _logger.debug(f"{tab.tab.target_id[:4]} put back.")

    def _clean_tab_return_task(
        self,
        main_task: asyncio.Task,
        task: asyncio.Task,
    ) -> None:
        _logger.info("Cleaning out background tasks.")
        self._background_render_tasks.remove(task)
        e = task.exception()
        if e:
            _logger.error("Clean tab return task found exception", exc_info=e)
            if not main_task.done():
                main_task.cancel()
            raise e

    def _check_render_task(
        self,
        name: str,
        tab: _KaleidoTab,
        main_task: asyncio.Task,
        error_log: None | list[ErrorEntry],
        task: asyncio.Task,
    ) -> None:
        if task.cancelled():
            _logger.info(f"Something cancelled {name}.")
            if error_log:
                error_log.append(
                    ErrorEntry(name, asyncio.CancelledError, tab.javascript_log),
                )
        elif e := task.exception():
            _logger.error(f"Render Task Error In {name}- ", exc_info=e)
            if isinstance(e, (asyncio.TimeoutError, TimeoutError)) and error_log:
                error_log.append(
                    ErrorEntry(name, e, tab.javascript_log),
                )
            else:
                _logger.error("Cancelling all.")
                if not main_task.done():
                    main_task.cancel()
                raise e
        _logger.info(f"Returning {name} tab after render.")
        t = asyncio.create_task(self._return_kaleido_tab(tab))
        self._background_render_tasks.add(t)
        t.add_done_callback(partial(self._clean_tab_return_task, main_task))

    async def _render_task(
        self,
        tab: _KaleidoTab,
        args: Any,
        error_log: None | list[ErrorEntry] = None,
        profiler: None | list = None,
    ):
        _logger.info(f"Posting a task for {args['full_path'].name}")
        if self._timeout:
            try:
                await asyncio.wait_for(
                    tab._write_fig(  # noqa: SLF001 I don't want it documented, too complex for user
                        **args,
                        error_log=error_log,
                        profiler=profiler,
                    ),
                    self._timeout,  # timeout can be None, no need for branches
                )
            except BaseException as e:
                if error_log:
                    error_log.append(
                        ErrorEntry(
                            args["full_path"].name,
                            e,
                            tab.javascript_log
                            if hasattr(
                                tab,
                                "javascript_log",
                            )
                            else [],
                        ),
                    )
                else:
                    raise

        else:
            await tab._write_fig(  # noqa: SLF001 I don't want it documented, too complex for user
                **args,
                error_log=error_log,
                profiler=profiler,
            )
        _logger.info(f"Posted task ending for {args['full_path'].name}")

    async def calc_fig(
        self,
        fig: _fig_tools.Figurish,
        path: str | Path | None = None,
        opts: None | _fig_tools.LayoutOpts = None,
        *,
        topojson: str | None = None,
    ):
        """
        Calculate the bytes for a figure.

        This function does not support parallelism or multi-image processing like
        `write_fig` does, although its arguments are a subset of those of `write_fig`.
        This function is currently just meant to bridge the old and new API.
        """
        if not _is_figurish(fig) and isinstance(fig, Iterable):
            raise TypeError("Calc fig can not process multiple images at a time.")
        spec, full_path = build_fig_spec(fig, path, opts)
        tab = await self._get_kaleido_tab()
        args = {
            "spec": spec,
            "full_path": full_path,
            "topojson": topojson,
        }
        data = None
        timeout = self._timeout if self._timeout else None
        data = await asyncio.wait_for(
            tab._calc_fig(  # noqa: SLF001 I don't want it documented, too complex for user
                **args,
            ),
            timeout,
        )
        await self._return_kaleido_tab(tab)
        return data[0]

    async def write_fig(  # noqa: PLR0913, PLR0912, C901 (too many args, complexity)
        self,
        fig: _fig_tools.Figurish,
        path: str | Path | None = None,
        opts: _fig_tools.LayoutOpts | None = None,
        *,
        topojson: str | None = None,
        error_log: None | list[ErrorEntry] = None,
        profiler: None | list = None,
    ):
        """
        Call the plotly renderer via javascript on first available tab.

        Args:
            fig: the plotly figure or an iterable of plotly figures
            path: the path to write the images to. if its a directory, we will try to
                generate a name. If the path contains an extension,
                "path/to/my_image.png", that extension will be the format used if not
                overridden in `opts`. If you pass a complete path (filename), for
                multiple figures, you will overwrite every previous figure.
            opts: dictionary describing format, width, height, and scale of image
            topojson: a link ??? TODO
            error_log: a supplied list, will be populated with `ErrorEntry`s
                       which can be converted to strings. Note, this is for
                       collections errors that have to do with plotly. They will
                       not be thrown. Lower level errors (kaleido, choreographer)
                       will still be thrown. If not passed, all errors raise.
            profiler: a supplied dictionary to collect stats about the operation
                      about tabs, runtimes, etc.

        """
        if error_log is not None:
            _logger.info("Using error log.")
        if profiler is not None:
            _logger.info("Using profiler.")

        if _is_figurish(fig) or not isinstance(fig, Iterable):
            fig = [fig]
        else:
            _logger.debug(f"Is iterable {type(fig)}")

        if main_task := asyncio.current_task():
            self._main_tasks.add(main_task)
        tasks = set()

        async def _loop(f):
            spec, full_path = build_fig_spec(f, path, opts)
            tab = await self._get_kaleido_tab()
            if profiler is not None and tab.tab.target_id not in profiler:
                profiler[tab.tab.target_id] = []
            t = asyncio.create_task(
                self._render_task(
                    tab,
                    args={
                        "spec": spec,
                        "full_path": full_path,
                        "topojson": topojson,
                    },
                    error_log=error_log,
                    profiler=profiler,
                ),
            )
            t.add_done_callback(
                partial(
                    self._check_render_task,
                    full_path.name,
                    tab,
                    main_task,
                    error_log,
                ),
            )
            tasks.add(t)

        try:
            if hasattr(fig, "__aiter__"):  # is async iterable
                _logger.debug("Is async for")
                async for f in fig:
                    await _loop(f)
            else:
                _logger.debug("Is sync for")
                for f in fig:
                    await _loop(f)
            _logger.debug("awaiting tasks")
            await asyncio.gather(*tasks, return_exceptions=True)
        except:
            _logger.exception("Cleaning tasks after error.")
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise
        finally:
            if main_task:
                self._main_tasks.remove(main_task)

    async def write_fig_from_object(  # noqa: C901 too complex
        self,
        generator: Iterable | AsyncIterable,
        *,
        error_log: None | list[ErrorEntry] = None,
        profiler: None | list = None,
    ):
        """
        Equal to `write_fig` but allows the user to generate all arguments.

        Generator must yield dictionaries with keys:
        - fig: the plotly figure
        - path: (optional, string or pathlib.Path) the path
        - opts: (optional) dictionary with:
            - format (string)
            - scale (number)
            - height (number)
            - and width (number)
        - topojson: (optional) topojsons are used to customize choropleths

        Generators are good because, if rendering many images, one doesn't need to
        prerender them all. They can be rendered and yielded asynchronously.

        While `write_fig` can also take generators, but only for the figure.
        In this case, the generator will specify all render-related arguments.

        Args:
            generator: an iterable or generator which supplies a dictionary
                       of arguments to pass to tab.write_fig.
            error_log: A supplied list, will be populated with `ErrorEntry`s
                       which can be converted to strings. Note, this is for
                       collections errors that have to do with plotly. They will
                       not be thrown. Lower level errors (kaleido, choreographer)
                       will still be thrown.
            profiler: A supplied dictionary, will be populated with information
                      about tabs, runtimes, etc.

        """
        if error_log is not None:
            _logger.info("Using error log.")
        if profiler is not None:
            _logger.info("Using profiler.")

        if main_task := asyncio.current_task():
            self._main_tasks.add(main_task)
        tasks = set()

        async def _loop(args):
            spec, full_path = build_fig_spec(
                args.pop("fig"),
                args.pop("path", None),
                args.pop("opts", None),
            )
            args["spec"] = spec
            args["full_path"] = full_path
            tab = await self._get_kaleido_tab()
            if profiler is not None and tab.tab.target_id not in profiler:
                profiler[tab.tab.target_id] = []
            t = asyncio.create_task(
                self._render_task(
                    tab,
                    args=args,
                    error_log=error_log,
                    profiler=profiler,
                ),
            )
            t.add_done_callback(
                partial(
                    self._check_render_task,
                    full_path.name,
                    tab,
                    main_task,
                    error_log,
                ),
            )
            tasks.add(t)

        try:
            if hasattr(generator, "__aiter__"):  # is async iterable
                _logger.debug("Is async for")
                async for args in generator:
                    await _loop(args)
            else:
                _logger.debug("Is sync for")
                for args in generator:
                    await _loop(args)
            _logger.debug("awaiting tasks")
            await asyncio.gather(*tasks, return_exceptions=True)
        except:
            _logger.exception("Cleaning tasks after error.")
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise
        finally:
            if main_task:
                self._main_tasks.remove(main_task)
