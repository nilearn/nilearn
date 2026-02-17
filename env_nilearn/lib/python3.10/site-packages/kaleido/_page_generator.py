from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import logistro

_logger = logistro.getLogger(__name__)

DEFAULT_PLOTLY = "https://cdn.plot.ly/plotly-2.35.2.js"
DEFAULT_MATHJAX = (
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js"
    "?config=TeX-AMS-MML_SVG"
)

KJS_PATH = Path(__file__).resolve().parent / "vendor" / "kaleido_scopes.js"


def _ensure_path(path: Path | str | tuple[str | Path, str]) -> None:
    if isinstance(path, tuple):
        path = path[0]
    _logger.debug(f"Ensuring path {path!s}")
    parsed = urlparse(str(path))
    _logger.debug(f"Parsed file path: {parsed}")
    if parsed.scheme.startswith("http"):  # is url
        return
    elif parsed.scheme.startswith("file"):
        if (_p := Path(url2pathname(parsed.path))).exists():
            return
        _logger.error(f"File parsed to: {_p}")
    elif Path(path).exists():
        return
    raise FileNotFoundError(f"{path!s} does not exist.")


class PageGenerator:
    """
    A page generator can set the versions of the js libraries used to render.

    It does this by outputting the HTML used to render the plotly figures.
    """

    header = """
<!DOCTYPE html>
<html>
    <head>
        <style id="head-style"></style>
        <title>Kaleido-fier</title>
        <script>
          window.PlotlyConfig = {MathJaxConfig: 'local'}
        </script>
        <script type="text/x-mathjax-config">
          MathJax.Hub.Config({ "SVG": { blacker: 0 }})
        </script>
"""
    """The header is the HTML that always goes at the top. Rarely needs changing."""

    footer = f"""
        <script src="{KJS_PATH.as_uri()}"></script>
    </head>
    <body style="{{margin: 0; padding: 0;}}"><img id="kaleido-image" /></body>
</html>
"""
    """The footer is the HTML that always goes on the bottom. Rarely needs changing."""

    def __init__(  # noqa: C901
        self,
        *,
        plotly: None | Path | str | tuple[Path | str, str] = None,
        mathjax: None | Path | str | bool | tuple[Path | str, str] = None,
        others: None | list[Path | str | tuple[Path | str, str]] = None,
        force_cdn: bool = False,
    ):
        """
        Create a PageGenerator.

        Args:
            plotly: The url to the plotly.js to use. Defaults to plotly.js
                present in plotly.py, if installed. Otherwise fallback to
                value of DEFAULT_PLOTLY.
            mathjax: The url to the mathjax script. Defaults to values of
                DEFAULT_MATHJAX. Can be set to false to disable mathjax.
            others: A list of other script urls to include. Usually strings, but
                can be (str, str) where it's (url, encoding).
            force_cdn: Set True to force CDN use, defaults to False.

        """
        self._scripts = []
        if mathjax is not False:
            if mathjax is None or mathjax is True:
                mathjax = DEFAULT_MATHJAX
            elif mathjax:
                _ensure_path(mathjax)
            self._scripts.append(mathjax)
        if force_cdn:
            plotly = (DEFAULT_PLOTLY, "utf-8")
        elif not plotly:
            try:
                # ruff: noqa: PLC0415
                # is this the best way to do this? can't we use importlib?
                import plotly as pltly  # type: ignore[import-untyped]

                plotly_path = (
                    Path(pltly.__file__).parent / "package_data" / "plotly.min.js"
                )
                plotly = (
                    plotly_path.as_uri(),
                    "utf-8",
                )
                if not plotly_path.is_file():
                    _logger.warning(
                        f"Found plotly but path to js is wrong? {plotly[0]}",
                    )
                    plotly = (DEFAULT_PLOTLY, "utf-8")
                else:
                    plotly[0]
            except ImportError:
                _logger.info("Plotly not installed. Using CDN.")
                plotly = (DEFAULT_PLOTLY, "utf-8")
        elif isinstance(plotly, (str, Path)):
            _ensure_path(plotly)
            plotly = (plotly, "utf-8")
        _logger.debug(f"Plotly script: {plotly}")
        self._scripts.append(plotly)
        if others:
            for o in others:
                _ensure_path(o)
            self._scripts.extend(others)

    def generate_index(self, path=None):
        """
        Generate the page.

        Args:
            path: If specified, page is written to path. Otherwise it is returned.

        """
        page = self.header
        script_tag = '\n        <script src="%s"></script>'
        script_tag_charset = '\n        <script src="%s" charset="%s"></script>'
        for script in self._scripts:
            if isinstance(script, (str, Path)):
                page += script_tag % str(script)
            else:
                page += script_tag_charset % script
        page += self.footer
        _logger.debug2(page)
        if not path:
            return page
        with (path).open("w") as f:
            f.write(page)
        return path.as_uri()
