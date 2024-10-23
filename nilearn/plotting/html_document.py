"""Handle HTML plotting."""

import warnings
import weakref
import webbrowser
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from queue import Empty, Queue
from socketserver import TCPServer
from threading import Thread

from nilearn._utils import remove_parameters

MAX_IMG_VIEWS_BEFORE_WARNING = 10
BROWSER_TIMEOUT_SECONDS = 3.0


def set_max_img_views_before_warning(new_value):
    """Set the number of open views which triggers a warning.

    If `None` or a negative number, disable the memory warning.
    """
    global MAX_IMG_VIEWS_BEFORE_WARNING
    MAX_IMG_VIEWS_BEFORE_WARNING = new_value


def _open_in_browser(content):
    """Open a page in the user's web browser.

    This function starts a local server in a separate thread, opens the page
    with webbrowser, and shuts down the server once it has served one request.
    """
    queue = Queue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            del args

        def do_GET(self):  # noqa: N802
            if not self.path.endswith("index.html"):
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            queue.put("done")

    server = TCPServer(("", 0), Handler)
    _, port = server.server_address

    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://localhost:{port}/index.html"
    webbrowser.open(url)
    try:
        queue.get(timeout=BROWSER_TIMEOUT_SECONDS)
    except Empty:
        raise RuntimeError(
            "Failed to open nilearn plot or report in a web browser."
        )
    server.shutdown()
    server_thread.join()


class HTMLDocument:
    """Embeds a plot in a web page.

    If you are running a Jupyter notebook, the plot will be displayed
    inline if this object is the output of a cell.
    Otherwise, use ``open_in_browser()`` to open it in a web browser (or
    ``save_as_html("filename.html")`` to save it as an html file).

    Use ``str(document)`` or ``document.html`` to get the content of the
    web page, and ``document.get_iframe()`` to have it wrapped in an iframe.

    """

    _all_open_html_repr = weakref.WeakSet()

    def __init__(self, html, width=600, height=400):
        self.html = html
        self.width = width
        self.height = height
        self._temp_file = None
        self._check_n_open()
        self._temp_file_removing_proc = None

    def _check_n_open(self):
        HTMLDocument._all_open_html_repr.add(self)
        if MAX_IMG_VIEWS_BEFORE_WARNING is None:
            return
        if MAX_IMG_VIEWS_BEFORE_WARNING < 0:
            return
        if (
            len(HTMLDocument._all_open_html_repr)
            > MAX_IMG_VIEWS_BEFORE_WARNING - 1
        ):
            warnings.warn(
                "It seems you have created "
                f"more than {MAX_IMG_VIEWS_BEFORE_WARNING} "
                "nilearn views. As each view uses dozens "
                "of megabytes of RAM, you might want to "
                "delete some of them."
            )

    def resize(self, width, height):
        """Resize the plot displayed in a Jupyter notebook.

        Parameters
        ----------
        width : :obj:`int`
            New width of the plot.

        height : :obj:`int`
            New height of the plot.

        """
        self.width, self.height = width, height
        return self

    def get_iframe(self, width=None, height=None):
        """Get the document wrapped in an inline frame.

        For inserting in another HTML page of for display in a Jupyter
        notebook.

        Parameters
        ----------
        width : :obj:`int` or ``None``, default=None
            Width of the inline frame.

        height : :obj:`int` or ``None``, default=None
            Height of the inline frame.

        Returns
        -------
        wrapped : :obj:`str`
            Raw HTML code for the inline frame.

        """
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        escaped = escape(self.html, quote=True)
        wrapped = (
            f'<iframe srcdoc="{escaped}" '
            f'width="{width}" height="{height}" '
            'frameBorder="0"></iframe>'
        )
        return wrapped

    def get_standalone(self):
        """Return the plot in an HTML page."""
        return self.html

    def _repr_html_(self):
        """Return html representation of the plot.

        Used by the Jupyter notebook.

        Users normally won't call this method explicitly.

        See the jupyter documentation:
        https://ipython.readthedocs.io/en/stable/config/integrating.html
        """
        return self.get_iframe()

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Return html representation of the plot.

        Used by the Jupyter notebook.

        Users normally won't call this method explicitly.

        See the jupyter documentation:
        https://ipython.readthedocs.io/en/stable/config/integrating.html
        """
        del include, exclude
        return {"text/html": self.get_iframe()}

    def __str__(self):
        return self.html

    def save_as_html(self, file_name):
        """Save the plot in an HTML file, that can later be opened \
        in a browser.

        Parameters
        ----------
        file_name : :obj:`str`
            Path to the HTML file used for saving.

        """
        with Path(file_name).open("wb") as f:
            f.write(self.get_standalone().encode("utf-8"))

    @remove_parameters(
        removed_params=["temp_file_lifetime"],
        reason=(
            "this function does not use a temporary file anymore "
            "and 'temp_file_lifetime' has no effect."
        ),
        end_version="0.13.0",
    )
    def open_in_browser(
        self,
        file_name=None,
        temp_file_lifetime="deprecated",  # noqa: ARG002
    ):
        """Save the plot to a temporary HTML file and open it in a browser.

        Parameters
        ----------
        file_name : :obj:`str` or ``None``, default=None
            HTML file to use as a temporary file.

        temp_file_lifetime : :obj:`float`, default=30

            .. deprecated:: 0.10.3

                The parameter is kept for backward compatibility and will be
                removed in a future version. It has no effect.
        """
        if file_name is None:
            _open_in_browser(self.get_standalone().encode("utf-8"))
        else:
            self.save_as_html(file_name)
            webbrowser.open(f"file://{file_name}")
