import hashlib
import itertools
import mimetypes
import re
import shutil
from base64 import b64encode, b64decode
from datetime import datetime, timedelta
from inspect import cleandoc
from pathlib import Path
import warnings

import htmlmin
from ansi2html import Ansi2HTMLConverter
from ansi2html.style import get_styles
from docutils.core import publish_parts
from jinja2 import (
    Environment,
    FileSystemLoader,
    TemplateNotFound,
    ChainableUndefined,
    select_autoescape,
)
from markupsafe import Markup

from . import __version__

TEMPLATE_PATH = Path(__file__).parent / "templates"
# category/style: background-color, color
COLORS = {
    "passed": ("#43A047", "#FFFFFF"),
    "failed": ("#F44336", "#FFFFFF"),
    "error": ("#B71C1C", "#FFFFFF"),
    "xfailed": ("#EF9A9A", "#222222"),
    "xpassed": ("#A5D6A7", "#222222"),
    "skipped": ("#9E9E9E", "#FFFFFF"),
    "notrun": ("#9E9E9E", "#FFFFFF"),
    "rerun": ("#FBC02D", "#222222"),
    "warning": ("#FBC02D", "#222222"),
    "green": ("#43A047", "#FFFFFF"),
    "red": ("#E53935", "#FFFFFF"),
    "yellow": ("#FBC02D", "#222222"),
}


def pytest_addoption(parser):
    group = parser.getgroup("report generation")
    group.addoption(
        "--split-report",
        action="store_true",
        help="store CSS and image files separately from the HTML.",
    )


def pytest_configure(config):
    config.pluginmanager.register(TemplatePlugin(config))


def css_minify(s):
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"/\*.*?\*/", "", s)
    return s


class TemplatePlugin:

    def __init__(self, config):
        self.self_contained = not config.getoption("--split-report")
        self._css = None
        self._assets = []
        self._dirs = []

    def pytest_reporter_loader(self, dirs, config):
        self._dirs = dirs + [str(TEMPLATE_PATH)]
        conv = Ansi2HTMLConverter(escaped=False)
        self.env = env = Environment(
            loader=FileSystemLoader(self._dirs),
            autoescape=select_autoescape(["html", "htm", "xml"]),
            undefined=ChainableUndefined,
        )
        env.globals["get_ansi_styles"] = get_styles
        env.globals["self_contained"] = self.self_contained
        env.globals["__version__"] = __version__
        env.filters["css"] = self._cssfilter
        env.filters["asset"] = self._assetfilter
        env.filters["repr"] = repr
        env.filters["chain"] = itertools.chain.from_iterable
        env.filters["strftime"] = lambda ts, fmt: datetime.fromtimestamp(ts).strftime(fmt)
        env.filters["timedelta"] = lambda ts: timedelta(seconds=ts)
        env.filters["ansi"] = lambda s: conv.convert(s, full=False)
        env.filters["cleandoc"] = cleandoc
        env.filters["rst"] = lambda s: publish_parts(source=s, writer_name="html5")["body"]
        env.filters["css_minify"] = css_minify
        return env

    def pytest_reporter_context(self, context, config):
        context.setdefault("colors", COLORS)
        context.setdefault("time_format", "%Y-%m-%d %H:%M:%S")
        metadata = context.setdefault("metadata", {})

        if config.pluginmanager.getplugin("metadata"):
            from pytest_metadata.plugin import metadata_key

            metadata.update(config.stash[metadata_key])

    def _cssfilter(self, css):
        if self.self_contained:
            return Markup("<style>") + css + Markup("</style>")
        else:
            self._css = css
            return Markup('<link rel="stylesheet" type="text/css" href="style.css">')

    def _assetfilter(self, src, extension=None, inline=None):
        path = None
        b64_content = None
        raw_content = None
        if inline is None:
            inline = self.self_contained

        if isinstance(src, bytes):
            raw_content = src
        elif len(src) > 255:
            # Probably not a path
            b64_content = src
        else:
            try:
                for parent in [".", *self._dirs]:
                    maybe_file = Path(parent) / src
                    if maybe_file.is_file():
                        path = maybe_file
                        break
                else:
                    b64_content = src
            except ValueError:
                b64_content = src

        if not path and not b64_content and not raw_content:
            warnings.warn("Could not find file")
            path = src

        if inline:
            if path:
                fname = str(path)
            elif extension:
                fname = "temp." + extension
            mimetype, _ = mimetypes.guess_type(fname)
            if not mimetype:
                mimetype = "application/octet-stream"
            if path:
                raw_content = path.read_bytes()
            if raw_content:
                b64_content = b64encode(raw_content).decode("utf-8")
            return "data:" + mimetype + ";base64," + b64_content
        else:
            m = hashlib.sha1()
            if path:
                with path.open("rb") as fp:
                    while True:
                        data = fp.read(16384)
                        if not data:
                            break
                        m.update(data)
                content = path
            if b64_content:
                raw_content = b64decode(b64_content.encode("utf-8"))
            if raw_content:
                m.update(raw_content)
                content = raw_content
            if extension:
                suffix = "." + extension
            else:
                suffix = path.suffix
            fname = m.hexdigest() + suffix
            self._assets.append((fname, content))
            return "assets/" + fname

    def pytest_reporter_render(self, template_name, dirs, context):
        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            return
        html = template.render(context)
        minified = htmlmin.minify(html, remove_comments=True)
        return minified

    def pytest_reporter_finish(self, path, context, config):
        assets = path.parent / "assets"
        if not self.self_contained:
            assets.mkdir(parents=True, exist_ok=True)
        if self._css:
            style_css = path.parent / "style.css"
            style_css.write_text(self._css)
        for fname, content in self._assets:
            if isinstance(content, bytes):
                with open(assets / fname, "wb") as fp:
                    fp.write(content)
            else:
                shutil.copy(content, assets / fname)
